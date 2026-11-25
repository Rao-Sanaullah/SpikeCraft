"""spikecraft: write spiking-neuron equations, get trainable PyTorch layers."""
from . import encoding, zoo
from .equations import EquationError
from .layers import Spiking, SpikingConv2d, SpikingLinear
from .neuron import Neuron
from .surrogate import Surrogate

__version__ = "0.1.0"
__all__ = ["Neuron", "Spiking", "SpikingLinear", "SpikingConv2d", "Surrogate",
           "EquationError", "zoo", "encoding", "__version__"]
