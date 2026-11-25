"""Neuron: a differentiable spiking neuron defined by ODE equations."""
from __future__ import annotations

import copy
import textwrap

import sympy as sp
import torch
from torch import nn

from .equations import Compiled, EquationError, parse, _sym, _TORCH_FUNCS
from .surrogate import get_surrogate

SOLVERS = ("exp_euler", "euler")


class Neuron(nn.Module):
    """A spiking neuron built from equations.

    Example
    -------
    >>> lif = Neuron(
    ...     equations="dv/dt = (-(v - v_rest) + R*I) / tau",
    ...     threshold="v > v_th", reset="v = v_reset",
    ...     params=dict(tau=10.0, R=1.0, v_rest=0.0, v_reset=0.0, v_th=1.0),
    ...     learnable=["tau"],
    ... )

    Parameters
    ----------
    equations : one ``dX/dt = ...`` per line (or separated by ``;``). Time unit is
        whatever ``dt`` is in (ms by convention).
    threshold : spike condition, e.g. ``"v > v_th"`` or ``"v > v_th + beta*a"``.
        ``None`` gives a non-spiking (purely dynamical) unit.
    reset : statements applied on spike, e.g. ``"v = v_reset; w += b"``.
        Supports ``=``, ``+=``, ``-=``, ``*=``.
    params : name -> initial value.
    init : state -> number or expression of params (default 0), e.g. ``{"v": "v_rest"}``.
    learnable : parameter names trained by backprop.
    per_neuron : learnable params get one value per neuron (set when used in a layer).
    surrogate : ``fast_sigmoid`` | ``sigmoid`` | ``atan`` | ``triangle`` | ``gaussian``
        or a :class:`Surrogate`.
    solver : ``exp_euler`` (exact for states linear in themselves, Euler otherwise)
        or ``euler``.
    threshold_mode : ``level`` (spike while condition holds; use with a reset) or
        ``rising`` (spike only on the step the condition becomes true; for models
        without reset such as Hodgkin-Huxley).
    detach_reset : stop gradients through the reset path (usually trains better).
    input : name of the input-current symbol in the equations (default ``I``).
    """

    def __init__(self, equations: str, threshold: str | None = None, reset: str | None = None,
                 params: dict | None = None, init: dict | None = None,
                 learnable=(), per_neuron: bool = False, surrogate="fast_sigmoid",
                 solver: str = "exp_euler", dt: float = 1.0, threshold_mode: str = "level",
                 detach_reset: bool = True, input: str = "I", name: str | None = None):
        super().__init__()
        params = dict(params or {})
        init = dict(init or {})
        learnable = list(learnable)
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}")
        if threshold_mode not in ("level", "rising"):
            raise ValueError("threshold_mode must be 'level' or 'rising'")
        bad = [p for p in learnable if p not in params]
        if bad:
            raise EquationError(f"learnable {bad} not found in params")

        self.spec = dict(equations=equations, threshold=threshold, reset=reset, params=params,
                         init=init, learnable=learnable, per_neuron=per_neuron,
                         surrogate=surrogate, solver=solver, dt=dt,
                         threshold_mode=threshold_mode, detach_reset=detach_reset,
                         input=input, name=name)
        self.name = name or "Neuron"
        self._pm = parse(equations, threshold, reset, params, input)
        bad = [s for s in init if s not in self._pm.states]
        if bad:
            raise EquationError(f"init given for unknown states {bad}")
        self.states = self._pm.states
        self.param_names = list(params)
        self.solver = solver
        self.threshold_mode = threshold_mode
        self.detach_reset = detach_reset
        self.per_neuron = per_neuron
        self.surrogate = get_surrogate(surrogate)

        self.p = nn.ParameterDict()
        for k, v in params.items():
            t = torch.as_tensor(float(v), dtype=torch.get_default_dtype())
            if k in learnable:
                self.p[k] = nn.Parameter(t)
            else:
                self.p[k] = nn.Parameter(t, requires_grad=False)
        self.register_buffer("dt", torch.as_tensor(float(dt), dtype=torch.get_default_dtype()))

        psyms = [sp.Symbol(n) for n in self.param_names]
        self._init_exprs = {}
        for s in self.states:
            e = _sym(str(init.get(s, 0.0)), f"init for {s}")
            extra = [str(x) for x in e.free_symbols if str(x) not in self.param_names]
            if extra:
                raise EquationError(f"init for '{s}' uses unknown symbols {extra}")
            self._init_exprs[s] = e
        self._init_fn = {s: sp.lambdify(psyms, e, modules=[_TORCH_FUNCS])
                         for s, e in self._init_exprs.items()}
        self._compiled = None

    # ------------------------------------------------------------------ setup
    @property
    def compiled(self) -> Compiled:
        if self._compiled is None:
            self._compiled = Compiled(self._pm, self.solver)
        return self._compiled

    def __getstate__(self):  # allow torch.save / pickling (lambdas are rebuilt)
        st = self.__dict__.copy()
        st["_compiled"] = None
        st["_init_fn"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        psyms = [sp.Symbol(n) for n in self.param_names]
        self._init_fn = {s: sp.lambdify(psyms, e, modules=[_TORCH_FUNCS])
                         for s, e in self._init_exprs.items()}

    def materialize(self, shape) -> "Neuron":
        """Give learnable params one value per neuron (called by layers)."""
        if not self.per_neuron:
            return self
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        for k in self.spec["learnable"]:
            old = self.p[k]
            if old.dim() == 0:
                self.p[k] = nn.Parameter(old.detach().expand(shape).clone())
        return self

    def replace(self, **changes) -> "Neuron":
        """Copy with modified spec, e.g. ``lif.replace(params={'tau': 20.0})``.
        ``params`` / ``init`` are merged into the existing dicts."""
        spec = copy.deepcopy(self.spec)
        for k in ("params", "init"):
            if k in changes:
                spec[k] = {**spec[k], **changes.pop(k)}
        spec.update(changes)
        return Neuron(**spec)

    # -------------------------------------------------------------- dynamics
    def _pvals(self):
        return [self.p[k] for k in self.param_names]

    def init_state(self, shape, device=None, dtype=None) -> dict:
        dtype = dtype or torch.get_default_dtype()
        pv = [p.to(device=device, dtype=dtype) for p in self._pvals()]
        out = {}
        for s in self.states:
            v = torch.as_tensor(self._init_fn[s](*pv), device=device, dtype=dtype)
            out[s] = torch.zeros(shape, device=device, dtype=dtype) + v.detach()
        return out

    @staticmethod
    def _exp_coeffs(A, dt):
        small = A.abs() < 1e-8
        A_safe = torch.where(small, torch.full_like(A, 1e-8), A)
        phi = torch.where(small, dt.expand_as(A), torch.expm1(A_safe * dt) / A_safe)
        return A, torch.exp(A * dt), phi

    def prepare(self, ref: torch.Tensor) -> dict:
        """Per-sequence cache of exponential-step coefficients that depend only on
        parameters (e.g. exp(-dt/tau) for LIF). Pass the result to ``step(..., cache=)``."""
        c = self.compiled
        if not c.static_A:
            return {}
        pv = self._pvals()
        zero = torch.zeros((), dtype=ref.dtype, device=ref.device)
        args = [zero] * (len(self.states) + 1) + pv + [self.dt]
        return {s: self._exp_coeffs(torch.as_tensor(c.linear_A[s](*args), dtype=ref.dtype,
                                                    device=ref.device), self.dt)
                for s in c.static_A}

    def step(self, I: torch.Tensor, state: dict | None = None, cache: dict | None = None):
        """Advance one time step. Returns ``(spikes, new_state)``."""
        if state is None:
            state = self.init_state(I.shape, I.device, I.dtype)
        c = self.compiled
        pv = self._pvals()
        dt = self.dt
        args = [state[s] for s in self.states] + [I] + pv + [dt]

        new = {}
        for s in self.states:
            x = state[s]
            f = c.f[s](*args)
            if s in c.linear_A:
                if cache is not None and s in cache:
                    A, decay, phi = cache[s]
                else:
                    A, decay, phi = self._exp_coeffs(
                        torch.as_tensor(c.linear_A[s](*args), dtype=x.dtype, device=x.device), dt)
                y = x * decay + phi * (f - A * x)
            else:
                y = x + dt * f
            new[s] = y if y.shape == x.shape else y + torch.zeros_like(x)  # full shape

        new_args = [new[s] for s in self.states] + [I] + pv + [dt]
        if c.threshold is None:
            spk = torch.zeros_like(new[self.states[0]])
            return spk, new
        thr = c.threshold(*new_args)
        if not (torch.is_tensor(thr) and thr.shape == I.shape):
            thr = thr + torch.zeros_like(I)
        spk = self.surrogate(thr)
        if self.threshold_mode == "rising":
            prev = c.threshold(*args)
            spk = spk * (torch.as_tensor(prev) < 0).to(spk.dtype).detach()

        sr = spk.detach() if self.detach_reset else spk
        targets = [(v, op, fn(*new_args)) for v, op, fn in c.resets]
        for v, op, val in targets:
            x = new[v]
            if op == "=":
                new[v] = x + sr * (val - x)
            elif op == "+=":
                new[v] = x + sr * val
            elif op == "-=":
                new[v] = x - sr * val
            elif op == "*=":
                new[v] = x + sr * (x * val - x)
        return spk, new

    def forward(self, I_seq: torch.Tensor, state: dict | None = None, return_states: bool = False):
        """Run over time. ``I_seq``: ``[T, ...]``. Returns spikes ``[T, ...]``
        (and a dict of state traces ``[T, ...]`` if ``return_states``)."""
        spikes, traces = [], {s: [] for s in self.states}
        cache = self.prepare(I_seq)
        for t in range(I_seq.shape[0]):
            s, state = self.step(I_seq[t], state, cache)
            spikes.append(s)
            if return_states:
                for k in self.states:
                    traces[k].append(state[k])
        out = torch.stack(spikes)
        if return_states:
            return out, {k: torch.stack(v) for k, v in traces.items()}
        return out

    # ------------------------------------------------------------------- info
    def equations(self) -> str:
        sp_ = self.spec
        lines = [f"{self.name}"]
        lines += [f"  {l.strip()}" for l in sp_["equations"].strip().splitlines() if l.strip()]
        if sp_["threshold"]:
            lines.append(f"  spike : {sp_['threshold']}  ({self.threshold_mode})")
        if sp_["reset"]:
            lines.append(f"  reset : {sp_['reset']}")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        info = self.compiled.describe()
        ps = ", ".join(f"{k}={float(self.p[k].detach().mean()):g}"
                       + ("*" if self.p[k].requires_grad else "") for k in self.param_names)
        return (f"states={self.states}, solver={self.solver} "
                f"(exact: {info['exact_exponential']}), surrogate={self.surrogate.kind}, "
                f"dt={float(self.dt):g}\n  params: {ps}   (* = learnable)\n"
                + textwrap.indent(self.equations(), "  "))
