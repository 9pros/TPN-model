"""TPN Engine - Temporal Packet Network Inference Engine."""

from .phase import PhaseWeight, PhaseEncoding
from .attention import TPNAttention, tpn_softmax, standard_softmax
from .hadamard import HadamardTransform
from .evolution import EvolvablePhaseNetwork
from .inference import TPNModel, TPNConfig
from .trajectory import TrajectoryPredictor, Trajectory
from .intent import IntentPredictor, IntentState
from .gguf_loader import GGUFLoader
from .safetensors_loader import SafetensorsLoader
from .binetic import BineticFormat, BineticGraph, BineticVersion
from .binetic_converter import BineticConverter
from .tool_calling import Tool, ToolCall, ToolRegistry, ToolExecutor
from .openai_api import OpenAIChatCompletions, ChatMessage, ChatCompletionRequest
from .bitslicing import BitslicedGate, BitslicedOperation, ParallelExecutor

__all__ = [
    "PhaseWeight",
    "PhaseEncoding",
    "TPNAttention",
    "tpn_softmax",
    "standard_softmax",
    "HadamardTransform",
    "EvolvablePhaseNetwork",
    "TPNModel",
    "TPNConfig",
    "TrajectoryPredictor",
    "Trajectory",
    "IntentPredictor",
    "IntentState",
    "GGUFLoader",
    "SafetensorsLoader",
    "BineticFormat",
    "BineticGraph",
    "BineticVersion",
    "BineticConverter",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolExecutor",
    "OpenAIChatCompletions",
    "ChatMessage",
    "ChatCompletionRequest",
    "BitslicedGate",
    "BitslicedOperation",
    "ParallelExecutor",
]
