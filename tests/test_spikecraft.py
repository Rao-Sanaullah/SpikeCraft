import io
import math

import pytest
import torch

import spikecraft as sf
from spikecraft import zoo

DRIVE = {"LIF": 1.5, "SynapticLIF": 1.5, "ALIF": 1.5, "QIF": 0.5, "EIF": 20.0,
         "Izhikevich": 10.0, "AdEx": 800.0, "HodgkinHuxley": 10.0,
         "ResonateAndFire": 2.0, "TwoCompartment": 4.0}


@pytest.mark.parametrize("name", zoo.list_models())
def test_every_zoo_model_spikes(name):
    n = zoo.get(name)
    T = int(300 / float(n.dt))
    s = n(torch.full((T, 2), DRIVE[name]))
    assert s.shape == (T, 2)
    assert s.sum() > 0
    assert torch.isfinite(s).all()


@pytest.mark.parametrize("name", zoo.list_models())
def test_every_zoo_model_silent_without_input(name):
    if name == "ResonateAndFire":
        pytest.skip("starts at rest anyway")
    n = zoo.get(name)
    s = n(torch.zeros(int(100 / float(n.dt)), 1))
    assert s.sum() == 0


def test_exp_euler_matches_analytic_lif():
    n = zoo.LIF(tau=10.0, v_th=100.0)  # never spikes
    T, I = 50, 0.7
    _, tr = n(torch.full((T, 1), I), return_states=True)
    t = torch.arange(1, T + 1, dtype=torch.float32)
    expected = I * (1 - torch.exp(-t / 10.0))
    assert torch.allclose(tr["v"][:, 0], expected, atol=1e-5)


def test_euler_differs_from_exact():
    exact = zoo.LIF(v_th=100.0)
    euler = zoo.LIF(v_th=100.0, solver="euler")
    I = torch.full((5, 1), 1.0)
    _, a = exact(I, return_states=True)
    _, b = euler(I, return_states=True)
    assert not torch.allclose(a["v"], b["v"])


def test_custom_equation_and_gradients():
    neuron = sf.Neuron(
        equations="dv/dt = (-v + I) / tau",
        threshold="v > 1", reset="v = 0",
        params={"tau": 5.0}, learnable=["tau"], per_neuron=True, surrogate="atan",
    )
    layer = sf.SpikingLinear(neuron, 4, 3)
    x = torch.rand(20, 8, 4) * 3
    out = layer(x)
    assert out.shape == (20, 8, 3)
    out.sum().backward()
    assert layer.neuron.p["tau"].shape == (3,)
    assert layer.neuron.p["tau"].grad is not None
    assert layer.synapse.weight.grad.abs().sum() > 0
    # original spec neuron untouched (layers copy)
    assert neuron.p["tau"].dim() == 0


def test_unknown_symbol_error():
    with pytest.raises(sf.EquationError, match="Unknown symbols"):
        sf.Neuron("dv/dt = (-v + I) / tau_m", params={"tau": 1.0})


def test_bad_equation_format():
    with pytest.raises(sf.EquationError):
        sf.Neuron("v' = -v")


def test_reset_operators():
    n = sf.Neuron(
        equations="dv/dt = I\nda/dt = 0*a\ndb/dt = 0*b\ndc/dt = 0*c",
        threshold="v > 0.5", reset="v = 0; a += 2; b -= 1; c *= 3",
        params={}, init={"c": 1.0},
    )
    s, st = n.step(torch.ones(1))
    assert s.item() == 1
    assert st["v"].item() == 0 and st["a"].item() == 2
    assert st["b"].item() == -1 and st["c"].item() == 3


def test_rising_threshold_one_spike_per_crossing():
    n = sf.Neuron("dv/dt = I", threshold="v > 1", threshold_mode="rising", params={})
    s = n(torch.ones(10, 1))
    assert s.sum() == 1


def test_hh_spike_count_reasonable():
    n = zoo.HodgkinHuxley()
    s = n(torch.full((int(1000 / 0.025), 1), 10.0))
    assert 50 <= s.sum() <= 80  # ~68 Hz at 10 uA/cm^2


def test_alif_adapts():
    n = zoo.ALIF()
    s = n(torch.full((400, 1), 1.5))
    assert s[:100].sum() > s[300:].sum()


def test_overrides_and_replace():
    n = zoo.LIF(tau=20.0, learnable=["tau"])
    assert float(n.p["tau"].detach()) == 20.0 and n.p["tau"].requires_grad
    m = n.replace(params={"v_th": 2.0})
    assert float(m.p["v_th"]) == 2.0 and float(m.p["tau"]) == 20.0
    with pytest.raises(KeyError):
        zoo.LIF(not_a_param=1)


def test_spec_editing():
    spec = zoo.spec("LIF")
    spec["equations"] = "dv/dt = (-(v - v_rest) + R*I + noise) / tau"
    spec["params"]["noise"] = 0.1
    n = sf.Neuron(**spec)
    assert "noise" in n.param_names


def test_conv_layer_and_save():
    layer = sf.SpikingConv2d(zoo.LIF(learnable=["tau"], per_neuron=True), 2, 4, 3, padding=1)
    x = torch.rand(5, 2, 2, 8, 8)
    y = layer(x)
    assert y.shape == (5, 2, 4, 8, 8)
    buf = io.BytesIO()
    torch.save(layer, buf)
    buf.seek(0)
    layer2 = torch.load(buf, weights_only=False)
    torch.manual_seed(0)
    assert torch.equal(layer2(x), y)


def test_recurrent_linear():
    layer = sf.SpikingLinear(zoo.ALIF(), 3, 5, recurrent=True)
    y = layer(torch.rand(10, 2, 3) * 2)
    assert y.shape == (10, 2, 5)


@pytest.mark.parametrize("kind", ["fast_sigmoid", "sigmoid", "atan", "triangle", "gaussian"])
def test_surrogates(kind):
    x = torch.linspace(-1, 1, 11, requires_grad=True)
    y = sf.Surrogate(kind)(x)
    y.sum().backward()
    assert torch.equal(y, (x >= 0).float())
    assert x.grad[5] == x.grad.max()  # peak at threshold


def test_encoders():
    x = torch.rand(4, 6)
    assert sf.encoding.rate(x, 7).shape == (7, 4, 6)
    lat = sf.encoding.latency(x, 9)
    assert lat.shape == (9, 4, 6) and lat.sum(0).max() <= 1
    assert math.isclose(sf.encoding.repeat(x, 3)[2].sum().item(), x.sum().item(), rel_tol=1e-6)
