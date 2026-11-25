"""Layers that combine a synapse (any nn.Module) with a Neuron, run over time."""
from __future__ import annotations

import copy

import torch
from torch import nn

from .neuron import Neuron


class Spiking(nn.Module):
    """Wrap any synapse module (Linear, Conv2d, ...) with a spiking neuron.

    Input ``[T, B, ...]`` -> spikes ``[T, B, ...]``. The synapse is applied to all
    time steps at once (folded into the batch), then the neuron is stepped in time.

    ``param_shape`` sets the shape of per-neuron parameters, e.g. ``(C, 1, 1)`` for conv.
    ``recurrent=True`` adds a learnable recurrent weight (feature dim must be last).
    """

    def __init__(self, synapse: nn.Module, neuron: Neuron, param_shape=None,
                 recurrent: bool = False, n_out: int | None = None):
        super().__init__()
        self.synapse = synapse
        self.neuron = copy.deepcopy(neuron)
        if param_shape is not None:
            self.neuron.materialize(param_shape)
        self.rec = nn.Linear(n_out, n_out, bias=False) if recurrent else None

    def forward(self, x: torch.Tensor, state: dict | None = None, return_states: bool = False):
        T, B = x.shape[:2]
        cur = self.synapse(x.reshape(T * B, *x.shape[2:]))
        cur = cur.reshape(T, B, *cur.shape[1:])
        spikes, traces = [], {k: [] for k in self.neuron.states}
        s = None
        cache = self.neuron.prepare(cur)
        for t in range(T):
            I = cur[t]
            if self.rec is not None and s is not None:
                I = I + self.rec(s)
            s, state = self.neuron.step(I, state, cache)
            spikes.append(s)
            if return_states:
                for k in traces:
                    traces[k].append(state[k])
        out = torch.stack(spikes)
        if return_states:
            return out, {k: torch.stack(v) for k, v in traces.items()}
        return out


class SpikingLinear(Spiking):
    """``nn.Linear`` + neuron. ``[T, B, in] -> [T, B, out]``."""

    def __init__(self, neuron: Neuron, in_features: int, out_features: int,
                 bias: bool = True, recurrent: bool = False):
        super().__init__(nn.Linear(in_features, out_features, bias=bias), neuron,
                         param_shape=(out_features,), recurrent=recurrent, n_out=out_features)


class SpikingConv2d(Spiking):
    """``nn.Conv2d`` + neuron. ``[T, B, C, H, W] -> [T, B, C_out, H', W']``."""

    def __init__(self, neuron: Neuron, in_channels: int, out_channels: int,
                 kernel_size, **conv_kwargs):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs),
                         neuron, param_shape=(out_channels, 1, 1))
