"""TPN Engine - Temporal Packet Network Inference Engine."""

from .phase import PhaseWeight, PhaseEncoding, PhaseMatrix, numpy_available
from .attention import TPNAttention, tpn_softmax, standard_softmax
from .attention_fast import (
    TPNAttentionFast,
    tpn_softmax_fast,
    standard_softmax_fast,
    attention_fast,
)
from .hadamard import HadamardTransform
from .hadamard_fast import FastHadamardTransform, fwht, hadamard_transform
from .evolution import EvolvablePhaseNetwork
from .inference import TPNModel, TPNConfig, BLOCK_WEIGHT_NAMES
from .trajectory import TrajectoryPredictor, Trajectory
from .intent import IntentPredictor, IntentState
from .gguf_loader import GGUFLoader
from .safetensors_loader import SafetensorsLoader
from .binetic import BineticFormat, BineticGraph, BineticVersion
from .binetic_converter import BineticConverter
from .tool_calling import Tool, ToolCall, ToolRegistry, ToolExecutor
from .openai_api import OpenAIChatCompletions, ChatMessage, ChatCompletionRequest
from .bitslicing import BitslicedGate, BitslicedOperation, ParallelExecutor
from .bitslice_inference import BitslicedLinear, BitslicedAttention

__all__ = [
    "PhaseWeight",
    "PhaseEncoding",
    "PhaseMatrix",
    "numpy_available",
    "TPNAttention",
    "tpn_softmax",
    "standard_softmax",
    "TPNAttentionFast",
    "tpn_softmax_fast",
    "standard_softmax_fast",
    "attention_fast",
    "HadamardTransform",
    "FastHadamardTransform",
    "fwht",
    "hadamard_transform",
    "EvolvablePhaseNetwork",
    "TPNModel",
    "TPNConfig",
    "BLOCK_WEIGHT_NAMES",
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
    "BitslicedLinear",
    "BitslicedAttention",
]
