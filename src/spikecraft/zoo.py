"""Model zoo. Every model is just an equation preset for :class:`Neuron`.

Use defaults, override anything, or copy the spec and edit an equation::

    zoo.LIF(tau=20.0, learnable=["tau"])            # override a param, train it
    zoo.Izhikevich(firing="chattering")              # named parameter set
    spec = zoo.spec("ALIF"); spec["reset"] += "; v -= 0.1"; Neuron(**spec)

Time unit: ms. Voltages: dimensionless for LIF-family, mV for biophysical models.
"""
from __future__ import annotations

import copy

from .neuron import Neuron

_SPEC_KEYS = {"equations", "threshold", "reset", "init", "learnable", "per_neuron",
              "surrogate", "solver", "dt", "threshold_mode", "detach_reset", "input", "name"}

_MODELS: dict[str, dict] = {}


def _register(name, **spec):
    spec["name"] = name
    _MODELS[name] = spec


_register(
    "LIF",
    equations="dv/dt = (-(v - v_rest) + R*I) / tau",
    threshold="v > v_th", reset="v = v_reset",
    params=dict(tau=10.0, R=1.0, v_rest=0.0, v_reset=0.0, v_th=1.0),
    init=dict(v="v_rest"), dt=1.0,
)

_register(
    "SynapticLIF",  # current-based LIF with a synaptic current filter
    equations="""
        di/dt = (-i + I) / tau_syn
        dv/dt = (-(v - v_rest) + R*i) / tau
    """,
    threshold="v > v_th", reset="v = v_reset",
    params=dict(tau=10.0, tau_syn=5.0, R=1.0, v_rest=0.0, v_reset=0.0, v_th=1.0),
    init=dict(v="v_rest"), dt=1.0,
)

_register(
    "ALIF",  # adaptive threshold (Bellec et al. 2018 style)
    equations="""
        dv/dt = (-(v - v_rest) + R*I) / tau
        da/dt = -a / tau_a
    """,
    threshold="v > v_th + beta*a", reset="v = v_reset; a += 1",
    params=dict(tau=10.0, tau_a=100.0, beta=0.2, R=1.0, v_rest=0.0, v_reset=0.0, v_th=1.0),
    init=dict(v="v_rest"), dt=1.0,
)

_register(
    "QIF",  # quadratic integrate-and-fire
    equations="dv/dt = (k*(v - v_rest)*(v - v_c) + R*I) / tau",
    threshold="v > v_peak", reset="v = v_reset",
    params=dict(tau=10.0, k=1.0, R=1.0, v_rest=0.0, v_c=0.5, v_peak=1.0, v_reset=0.0),
    init=dict(v="v_rest"), dt=0.5,
)

_register(
    "EIF",  # exponential integrate-and-fire
    equations="dv/dt = (-(v - v_rest) + delta_T*exp(min((v - v_T)/delta_T, 10)) + R*I) / tau",
    threshold="v > v_peak", reset="v = v_reset",
    params=dict(tau=10.0, R=1.0, v_rest=-65.0, v_T=-50.0, delta_T=2.0,
                v_peak=-30.0, v_reset=-68.0),
    init=dict(v="v_rest"), dt=0.1,
)

_register(
    "Izhikevich",  # Izhikevich 2003; I ~ 10 drives regular spiking
    equations="""
        dv/dt = 0.04*v**2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
    """,
    threshold="v >= v_peak", reset="v = c; u += d",
    params=dict(a=0.02, b=0.2, c=-65.0, d=8.0, v_peak=30.0),
    init=dict(v="c", u="b*c"), dt=0.5,
)

_IZH_FIRING = {
    "regular_spiking": dict(a=0.02, b=0.2, c=-65.0, d=8.0),
    "intrinsically_bursting": dict(a=0.02, b=0.2, c=-55.0, d=4.0),
    "chattering": dict(a=0.02, b=0.2, c=-50.0, d=2.0),
    "fast_spiking": dict(a=0.1, b=0.2, c=-65.0, d=2.0),
    "low_threshold": dict(a=0.02, b=0.25, c=-65.0, d=2.0),
    "resonator": dict(a=0.1, b=0.26, c=-65.0, d=2.0),
}

_register(
    "AdEx",  # adaptive exponential IF (Brette & Gerstner 2005); I in pA, v in mV
    equations="""
        dv/dt = (-gL*(v - EL) + gL*DT*exp(min((v - VT)/DT, 20)) - w + I) / C
        dw/dt = (a*(v - EL) - w) / tau_w
    """,
    threshold="v > v_peak", reset="v = v_reset; w += b",
    params=dict(C=281.0, gL=30.0, EL=-70.6, VT=-50.4, DT=2.0, tau_w=144.0,
                a=4.0, b=80.5, v_reset=-70.6, v_peak=0.0),
    init=dict(v="EL"), dt=0.1,
)

_register(
    "HodgkinHuxley",  # squid axon, rest -65 mV; I in uA/cm^2 (~10 for tonic firing)
    equations="""
        dv/dt = (I - gNa*m**3*h*(v - ENa) - gK*n**4*(v - EK) - gL*(v - EL)) / C
        dm/dt = 0.1*(v + 40)/(1 - exp(-(v + 40)/10)) * (1 - m) - 4*exp(-(v + 65)/18) * m
        dh/dt = 0.07*exp(-(v + 65)/20) * (1 - h) - 1/(1 + exp(-(v + 35)/10)) * h
        dn/dt = 0.01*(v + 55)/(1 - exp(-(v + 55)/10)) * (1 - n) - 0.125*exp(-(v + 65)/80) * n
    """,
    threshold="v > v_spike", reset=None, threshold_mode="rising",
    params=dict(C=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.387,
                v_spike=0.0),
    init=dict(v=-65.0, m=0.0529, h=0.5961, n=0.3177), dt=0.025,
)

_register(
    "ResonateAndFire",  # Izhikevich 2001 damped oscillator; omega in rad/ms
    equations="""
        dx/dt = b*x - omega*y + I
        dy/dt = omega*x + b*y
    """,
    threshold="y > y_th", reset="x = 0; y = 0",
    params=dict(b=-0.1, omega=0.6, y_th=1.0),
    dt=0.1,
)

_register(
    "TwoCompartment",  # dendrite receives input, soma spikes
    equations="""
        dvd/dt = (-(vd - E_L) + g_c*(vs - vd) + R*I) / tau_d
        dvs/dt = (-(vs - E_L) + g_c*(vd - vs)) / tau_s
    """,
    threshold="vs > v_th", reset="vs = v_reset",
    params=dict(tau_d=20.0, tau_s=10.0, g_c=1.0, R=1.0, E_L=0.0, v_th=1.0, v_reset=0.0),
    dt=1.0,
)


def list_models() -> list[str]:
    return sorted(_MODELS)


def spec(name: str) -> dict:
    """Editable copy of a preset's full spec (pass to ``Neuron(**spec)``)."""
    if name not in _MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return copy.deepcopy(_MODELS[name])


def get(model: str, **overrides) -> Neuron:
    """Build a preset. Keyword args that match a param name override that param;
    spec keys (``learnable``, ``dt``, ``surrogate``, ``solver``, ...) override the spec;
    ``params=`` / ``init=`` dicts are merged."""
    s = spec(model)
    for k, v in overrides.items():
        if k in ("params", "init"):
            s.setdefault(k, {}).update(v)
        elif k in _SPEC_KEYS:
            s[k] = v
        elif k in s["params"]:
            s["params"][k] = v
        else:
            raise KeyError(f"'{k}' is neither a parameter of {model} "
                           f"{sorted(s['params'])} nor a spec option {sorted(_SPEC_KEYS)}")
    return Neuron(**s)


def LIF(**kw) -> Neuron: return get("LIF", **kw)                        # noqa: E704
def SynapticLIF(**kw) -> Neuron: return get("SynapticLIF", **kw)        # noqa: E704
def ALIF(**kw) -> Neuron: return get("ALIF", **kw)                      # noqa: E704
def QIF(**kw) -> Neuron: return get("QIF", **kw)                        # noqa: E704
def EIF(**kw) -> Neuron: return get("EIF", **kw)                        # noqa: E704
def AdEx(**kw) -> Neuron: return get("AdEx", **kw)                      # noqa: E704
def HodgkinHuxley(**kw) -> Neuron: return get("HodgkinHuxley", **kw)    # noqa: E704
def ResonateAndFire(**kw) -> Neuron: return get("ResonateAndFire", **kw)  # noqa: E704
def TwoCompartment(**kw) -> Neuron: return get("TwoCompartment", **kw)  # noqa: E704


def Izhikevich(firing: str = "regular_spiking", **kw) -> Neuron:
    """``firing``: one of ``regular_spiking``, ``intrinsically_bursting``, ``chattering``,
    ``fast_spiking``, ``low_threshold``, ``resonator``."""
    if firing not in _IZH_FIRING:
        raise KeyError(f"firing must be one of {sorted(_IZH_FIRING)}")
    kw = {**_IZH_FIRING[firing], **kw}
    kw.setdefault("name", f"Izhikevich[{firing}]")
    return get("Izhikevich", **kw)
