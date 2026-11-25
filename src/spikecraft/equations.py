"""Parse neuron equations into SymPy and compile them to PyTorch callables."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy as sp
import torch

# SymPy function name -> torch implementation
_TORCH_FUNCS = {
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "tanh": torch.tanh,
    "sin": torch.sin,
    "cos": torch.cos,
    "Abs": torch.abs,
    "sigmoid": torch.sigmoid,
    "Max": lambda *a: _reduce(torch.maximum, a),
    "Min": lambda *a: _reduce(torch.minimum, a),
    "Heaviside": lambda x, *_: (torch.as_tensor(x) >= 0).to(torch.get_default_dtype()),
}

# Names users may write in equations that SymPy should treat as functions
_PARSE_LOCALS = {
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "tanh": sp.tanh,
    "sin": sp.sin, "cos": sp.cos, "abs": sp.Abs, "max": sp.Max, "min": sp.Min,
    "sigmoid": sp.Function("sigmoid"), "heaviside": sp.Heaviside,
}

_ODE_RE = re.compile(r"^d\s*([A-Za-z_]\w*)\s*/\s*dt\s*=\s*(.+)$")
_THRESH_RE = re.compile(r"^(.+?)\s*(>=|<=|>|<)\s*(.+)$")
_RESET_RE = re.compile(r"^([A-Za-z_]\w*)\s*(\+=|-=|\*=|=)\s*(.+)$")


def _reduce(fn, args):
    out = args[0]
    for a in args[1:]:
        out = fn(torch.as_tensor(out), torch.as_tensor(a))
    return out


class EquationError(ValueError):
    pass


def _sym(expr_str: str, where: str) -> sp.Expr:
    try:
        # Use a local symbol table so names like "I", "E", "S", "N", "beta" stay plain symbols
        names = set(re.findall(r"[A-Za-z_]\w*", expr_str)) - set(_PARSE_LOCALS)
        local = {n: sp.Symbol(n) for n in names}
        local.update(_PARSE_LOCALS)
        return sp.sympify(expr_str.replace("^", "**"), locals=local)
    except Exception as e:  # noqa: BLE001
        raise EquationError(f"Could not parse {where}: '{expr_str}' ({e})") from e


def _split(text: str) -> list[str]:
    lines = []
    for raw in re.split(r"[;\n]", text or ""):
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


@dataclass
class ParsedModel:
    states: list[str]
    odes: dict[str, sp.Expr]
    threshold: sp.Expr | None          # spike when threshold >= 0
    resets: list[tuple[str, str, sp.Expr]] = field(default_factory=list)  # (var, op, expr)
    input_name: str = "I"
    params: list[str] = field(default_factory=list)


def parse(equations: str, threshold: str | None, reset: str | None,
          params: dict, input_name: str = "I") -> ParsedModel:
    odes: dict[str, sp.Expr] = {}
    for line in _split(equations):
        m = _ODE_RE.match(line)
        if not m:
            raise EquationError(f"Expected 'dX/dt = expression', got: '{line}'")
        var, rhs = m.group(1), m.group(2)
        if var in odes:
            raise EquationError(f"State '{var}' defined twice")
        odes[var] = _sym(rhs, f"equation for d{var}/dt")
    if not odes:
        raise EquationError("No equations given")
    states = list(odes)

    thr = None
    if threshold:
        m = _THRESH_RE.match(threshold.strip())
        if not m:
            raise EquationError(f"Threshold must look like 'v > v_th', got '{threshold}'")
        lhs, op, rhs = _sym(m.group(1), "threshold"), m.group(2), _sym(m.group(3), "threshold")
        thr = lhs - rhs if op in (">", ">=") else rhs - lhs

    resets = []
    for line in _split(reset):
        m = _RESET_RE.match(line)
        if not m:
            raise EquationError(f"Reset must look like 'v = v_reset' or 'w += b', got '{line}'")
        var, op, rhs = m.groups()
        if var not in states:
            raise EquationError(f"Reset assigns unknown state '{var}' (states: {states})")
        resets.append((var, op, _sym(rhs, f"reset for {var}")))

    # Symbol check
    known = set(states) | set(params) | {input_name, "dt"}
    exprs = list(odes.values()) + ([thr] if thr is not None else []) + [r[2] for r in resets]
    used = set().union(*(e.free_symbols for e in exprs))
    unknown = sorted(str(s) for s in used if str(s) not in known)
    if unknown:
        raise EquationError(
            f"Unknown symbols {unknown}. Add them to params=..., or rename the input "
            f"(current input name: '{input_name}')."
        )
    return ParsedModel(states, odes, thr, resets, input_name, list(params))


class Compiled:
    """Torch callables for one parsed model. Argument order: states, input, params, dt."""

    def __init__(self, pm: ParsedModel, solver: str):
        self.pm = pm
        self.arg_names = pm.states + [pm.input_name] + pm.params + ["dt"]
        syms = [sp.Symbol(n) for n in self.arg_names]
        mods = [_TORCH_FUNCS]
        lam = lambda e: sp.lambdify(syms, e, modules=mods)  # noqa: E731

        self.f = {s: lam(e) for s, e in pm.odes.items()}
        # Exponential Euler: dx/dt = A x + B where A = df/dx (if A is free of x)
        self.linear_A = {}
        self.static_A = set()  # states whose A depends only on params/dt -> cache per sequence
        static_syms = {sp.Symbol(n) for n in pm.params} | {sp.Symbol("dt")}
        if solver == "exp_euler":
            for s, e in pm.odes.items():
                A = sp.diff(e, sp.Symbol(s))
                if sp.Symbol(s) in A.free_symbols:
                    A = sp.simplify(A)
                if sp.Symbol(s) not in A.free_symbols and A != 0:
                    self.linear_A[s] = lam(A)
                    if A.free_symbols <= static_syms:
                        self.static_A.add(s)
        self.threshold = lam(pm.threshold) if pm.threshold is not None else None
        self.resets = [(v, op, lam(e)) for v, op, e in pm.resets]

    def describe(self) -> dict:
        return {
            "states": self.pm.states,
            "exact_exponential": sorted(self.linear_A),
            "euler": sorted(set(self.pm.states) - set(self.linear_A)),
        }
