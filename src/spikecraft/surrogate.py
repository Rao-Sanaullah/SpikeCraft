"""Surrogate-gradient spike functions.

Forward pass is a Heaviside step; backward pass uses a smooth derivative.
"""
from __future__ import annotations

import math

import torch


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, kind):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.kind = kind
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        a, kind = ctx.alpha, ctx.kind
        if kind == "fast_sigmoid":
            g = 1.0 / (a * x.abs() + 1.0) ** 2
        elif kind == "sigmoid":
            s = torch.sigmoid(a * x)
            g = a * s * (1 - s)
        elif kind == "atan":
            g = (a / 2) / (1 + (math.pi / 2 * a * x) ** 2)
        elif kind == "triangle":
            g = torch.clamp(1.0 - a * x.abs(), min=0.0) * a
        elif kind == "gaussian":
            g = torch.exp(-((a * x) ** 2) / 2) * a / math.sqrt(2 * math.pi)
        else:  # pragma: no cover
            raise ValueError(kind)
        return grad_out * g, None, None


SURROGATES = ("fast_sigmoid", "sigmoid", "atan", "triangle", "gaussian")
_DEFAULT_ALPHA = {"fast_sigmoid": 10.0, "sigmoid": 5.0, "atan": 2.0, "triangle": 1.0, "gaussian": 1.0}


class Surrogate:
    """Callable spike function: ``Surrogate("atan", alpha=2.0)(x)``."""

    def __init__(self, kind: str = "fast_sigmoid", alpha: float | None = None):
        if kind not in SURROGATES:
            raise ValueError(f"Unknown surrogate '{kind}'. Choose from {SURROGATES}.")
        self.kind = kind
        self.alpha = float(alpha if alpha is not None else _DEFAULT_ALPHA[kind])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _Spike.apply(x, self.alpha, self.kind)

    def __repr__(self) -> str:
        return f"Surrogate('{self.kind}', alpha={self.alpha})"


def get_surrogate(s) -> Surrogate:
    if isinstance(s, Surrogate):
        return s
    if isinstance(s, str):
        return Surrogate(s)
    raise TypeError("surrogate must be a name or a Surrogate instance")
