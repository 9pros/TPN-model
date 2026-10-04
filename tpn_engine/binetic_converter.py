"""
Universal model to .binetic converter.

Converts any neural network model (transformer, CNN, RNN, etc.)
to .binetic format with graph structure and phase-encoded weights.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from .binetic import (
    BineticGraph, BineticFormat, BineticIdentity, GenerationRecord,
    BINETIC_IDENTITY,
)
from .phase import PhaseEncoding


class BineticConverter:
    """
    Universal converter from any model format to .binetic.
    
    Supports:
    - Transformer models (GPT, BERT, T5, etc.)
    - CNN models (ResNet, VGG, etc.)
    - RNN models (LSTM, GRU, etc.)
    - Quantized models (ternary, binary, etc.)
    - Custom architectures
    """
    
    def __init__(self):
        self._metadata: Dict[str, Any] = {}
    
    def convert(self, model_data: Dict[str, Any]) -> BineticGraph:
        """
        Convert any model to .binetic graph format.
        
        Args:
            model_data: Dictionary containing model information.
                Expected keys:
                - "layers": List of layer dictionaries with "name" and "weights"
                - "connections": List of (from_node, to_node) tuples
                - "architecture": Optional architecture type
                - "metadata": Optional metadata dictionary
        
        Returns:
            BineticGraph with nodes, edges, and metadata.
        """
        graph = BineticGraph()
        
        # Extract metadata
        architecture = model_data.get("architecture", "unknown")
        model_name = model_data.get("model_name", "converted_model")
        quantization = model_data.get("quantization", {})
        
        graph.metadata = {
            "architecture": architecture,
            "model_name": model_name,
            "quantization": quantization,
            "converted_at": self._get_timestamp(),
        }
        
        # Add any additional top-level metadata
        for key, value in model_data.items():
            if key not in ["layers", "connections", "architecture", "model_name", "quantization", "tensors"]:
                graph.metadata[key] = value

        # --- Binetic identity: ALWAYS asserted on Binetic-authored models ---
        graph.set_binetic_identity(
            owner="binetic.ai",
            partners=["Max Yeremenko", "binetic-partner"],
            derived_from=model_name,
            tool="binetic_converter",
            generation_info={
                "mode": "conversion",
                "architecture": architecture,
            },
        )
        graph.metadata["binetic"] = graph.identity.as_dict()

        # Add nodes (layers)
        layers = model_data.get("layers", [])
        for layer in layers:
            name = layer.get("name", f"layer_{len(graph.nodes)}")
            layer_type = layer.get("type", "unknown")
            weights = layer.get("weights")
            
            node_data = {
                "type": layer_type,
                "weights": weights,
            }
            
            # Add any additional layer metadata
            for key, value in layer.items():
                if key not in ["name", "type", "weights"]:
                    node_data[key] = value
            
            graph.add_node(name, node_data)
        
        # Add edges (connections)
        connections = model_data.get("connections", [])
        for conn in connections:
            if isinstance(conn, (list, tuple)) and len(conn) == 2:
                from_node, to_node = conn
                graph.add_edge(from_node, to_node, {"type": "sequential"})
        
        # If no connections specified, create sequential connections
        if not connections and len(layers) > 1:
            for i in range(len(layers) - 1):
                from_name = layers[i].get("name", f"layer_{i}")
                to_name = layers[i + 1].get("name", f"layer_{i + 1}")
                graph.add_edge(from_name, to_name, {"type": "sequential"})
        
        return graph
    
    def convert_from_gguf(self, gguf_loader) -> BineticGraph:
        """
        Convert from GGUF format.
        
        Args:
            gguf_loader: GGUFLoader instance.
        
        Returns:
            BineticGraph with extracted weights.
        """
        # Get model config
        config = gguf_loader.get_model_config()
        
        # Extract weights
        weights = gguf_loader.extract_weights()
        
        # Build model data
        model_data = {
            "architecture": config.get("architecture", "unknown"),
            "model_name": "converted_from_gguf",
            "quantization": {"source": "gguf"},
            "layers": [],
            "connections": [],
        }
        
        # Group weights by layer
        layer_weights = {}
        for name, data in weights.items():
            if name.startswith("blk."):
                parts = name.split(".")
                if len(parts) >= 2:
                    layer_name = parts[1]
                    if layer_name not in layer_weights:
                        layer_weights[layer_name] = {}
                    layer_weights[layer_name][name] = data
        
        # Create layers
        prev_layer = None
        for layer_name in sorted(layer_weights.keys()):
            layer_data = {
                "name": f"layer_{layer_name}",
                "type": "transformer",
                "weights": layer_weights[layer_name],
            }
            model_data["layers"].append(layer_data)
            
            if prev_layer is not None:
                model_data["connections"].append((prev_layer, f"layer_{layer_name}"))
            prev_layer = f"layer_{layer_name}"
        
        return self.convert(model_data)
    
    def convert_from_safetensors(self, safetensors_loader) -> BineticGraph:
        """
        Convert from safetensors format.
        
        Args:
            safetensors_loader: SafetensorsLoader instance.
        
        Returns:
            BineticGraph with extracted weights.
        """
        # Get model config
        config = safetensors_loader.get_model_config()
        
        # Extract first layer weights (memory efficient)
        layer_weights = safetensors_loader.extract_layer_weights(0)
        
        # Build model data
        model_data = {
            "architecture": config.get("model_type", "unknown"),
            "model_name": "converted_from_safetensors",
            "quantization": config.get("quantization", {}),
            "layers": [],
            "connections": [],
        }
        
        # Create layers from safetensors
        for name, data in layer_weights.items():
            layer_data = {
                "name": name,
                "type": "unknown",
                "weights": data.tolist() if hasattr(data, 'tolist') else data,
            }
            model_data["layers"].append(layer_data)
        
        # Create sequential connections
        for i in range(len(model_data["layers"]) - 1):
            model_data["connections"].append(
                (model_data["layers"][i]["name"], model_data["layers"][i + 1]["name"])
            )
        
        return self.convert(model_data)
    
    def convert_to_phase_encoded(self, graph: BineticGraph) -> BineticGraph:
        """
        Convert all weights in graph to phase-encoded format.
        
        Args:
            graph: BineticGraph with float weights.
        
        Returns:
            BineticGraph with phase-encoded weights.
        """
        for node_name, node_data in graph.nodes.items():
            weights = node_data.get("weights")
            if weights is not None:
                if isinstance(weights, list) and len(weights) > 0:
                    if isinstance(weights[0], list):
                        # 2D weights
                        node_data["phase_weights"] = PhaseEncoding.encode_matrix(weights)
                    else:
                        # 1D weights
                        node_data["phase_weights"] = PhaseEncoding.encode_vector(weights)
        
        return graph
    
    def save(self, filepath: str, graph: BineticGraph, version_id: str = "v1") -> None:
        """
        Save converted model to .binetic file.
        
        Args:
            filepath: Path to save file.
            graph: BineticGraph to save.
            version_id: Version identifier.
        """
        fmt = BineticFormat()
        fmt.save(filepath, graph, version_id=version_id)
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
