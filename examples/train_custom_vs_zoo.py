"""Train the same small SNN with a zoo neuron and with a custom-equation neuron.

Task (synthetic, no downloads): 3 classes of 30-channel Poisson spike trains; each class
drives a different channel group. Run: python examples/train_custom_vs_zoo.py
"""
import torch
from torch import nn

import spikecraft as sf
from spikecraft import zoo

torch.manual_seed(0)
T, C, K, N = 40, 30, 3, 600


def make_data(n):
    y = torch.randint(0, K, (n,))
    p = torch.full((n, C), 0.05)
    for k in range(K):
        p[y == k, k * 10:(k + 1) * 10] = 0.4
    return sf.encoding.rate(p, T), y  # [T, n, C]


def build(neuron):
    return nn.Sequential(sf.SpikingLinear(neuron, C, 64), sf.SpikingLinear(neuron, 64, K))


def train(model, name, epochs=15):
    xtr, ytr = make_data(N)
    xte, yte = make_data(300)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        for i in range(0, N, 50):
            out = sf.encoding.spike_count(model(xtr[:, i:i + 50]))
            loss = nn.functional.cross_entropy(out, ytr[i:i + 50])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (sf.encoding.spike_count(model(xte)).argmax(1) == yte).float().mean()
    taus = [round(float(l.neuron.p["tau"].detach().mean()), 2) for l in model]
    print(f"{name:28s} test acc={acc:.2%}  mean learned tau per layer={taus}")


# 1) zoo preset, membrane time constant learned per neuron
train(build(zoo.ALIF(learnable=["tau", "beta"], per_neuron=True)), "zoo.ALIF")

# 2) your own equation: LIF with a learnable leak nonlinearity
custom = sf.Neuron(
    equations="dv/dt = (-v + gain*tanh(I)) / tau",
    threshold="v > 0.5", reset="v = 0",
    params=dict(tau=8.0, gain=1.5), learnable=["tau", "gain"], per_neuron=True,
    surrogate="atan", name="TanhLIF",
)
train(build(custom), "custom TanhLIF")
