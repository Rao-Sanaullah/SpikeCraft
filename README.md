# SpikeCraft

**Write a spiking neuron as equations; get a trainable PyTorch layer.**

[![PyPI](https://img.shields.io/pypi/v/spikecraft.svg)](https://pypi.org/project/spikecraft/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Project page](https://img.shields.io/badge/project-page-1a6b4f.svg)](https://rao-sanaullah.github.io/spikecraft/)

SpikeCraft compiles a neuron's differential equations, spike condition and reset rules into a differentiable PyTorch module, with automatic exact-exponential integration, surrogate gradients and learnable time constants. Ten neuron models (LIF to Hodgkin–Huxley) ship as editable equation presets.

> Full documentation, results and figures are on the **[project page](https://rao-sanaullah.github.io/spikecraft/)**.

## Install

```bash
pip install spikecraft
```

## Example

```python
import spikecraft as sf

neuron = sf.Neuron(
    equations="dv/dt = (-v + I) / tau",
    threshold="v > 1", reset="v = 0",
    params={"tau": 2.0}, learnable=["tau"], per_neuron=True,
)
layer = sf.SpikingLinear(neuron, 128, 64)   # [T, B, 128] -> [T, B, 64]
```

## Repository

| Folder | Contents |
|---|---|
| `src/spikecraft/` | the library |
| `tests/` | test suite (`pytest`) |
| `examples/` | minimal training example |
| `benchmarks/` | DVS128 Gesture, SHD and library benchmarks used in the paper |
| `paper/make_results.py` | regenerates every figure and table from the experiment logs |
| `docs/` | project page (GitHub Pages) |

## Citation

```bibtex
@misc{spikecraft2026,
  title  = {SpikeCraft Enables Fast Exploration of Efficient Spiking Neuron Models for Embedded and Robotic AI},
  author = {Sanaullah},
  year   = {2026},
  url    = {https://github.com/Rao-Sanaullah/spikecraft}
}
```

## License

MIT
