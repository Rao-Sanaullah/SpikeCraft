"""Input encoders and small readout helpers."""
from __future__ import annotations

import torch


def rate(x: torch.Tensor, steps: int, gain: float = 1.0) -> torch.Tensor:
    """Poisson/Bernoulli rate coding. ``x`` in [0, 1] -> spikes ``[steps, *x.shape]``."""
    p = (x * gain).clamp(0, 1)
    return torch.bernoulli(p.unsqueeze(0).expand(steps, *p.shape))


def latency(x: torch.Tensor, steps: int, eps: float = 1e-6) -> torch.Tensor:
    """Time-to-first-spike: stronger input spikes earlier. One spike per element."""
    x = x.clamp(0, 1)
    t = ((1 - x) * (steps - 1)).round().long()
    out = torch.zeros(steps, *x.shape, device=x.device, dtype=torch.get_default_dtype())
    out.scatter_(0, t.unsqueeze(0), (x > eps).to(out.dtype).unsqueeze(0))
    return out


def repeat(x: torch.Tensor, steps: int) -> torch.Tensor:
    """Constant current: repeat analog input over time."""
    return x.unsqueeze(0).expand(steps, *x.shape)


def spike_count(spikes: torch.Tensor) -> torch.Tensor:
    """Sum over time (dim 0); a common classification readout."""
    return spikes.sum(0)
