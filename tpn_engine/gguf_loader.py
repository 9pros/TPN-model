"""
GGUF weight loader for TPN.

Loads weights from GGUF model files and converts them to
phase-encoded format for the TPN engine.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from gguf import GGUFReader
except ImportError:
    GGUFReader = None


@dataclass
class TensorInfo:
    """Information about a GGUF tensor."""
    name: str
    shape: Tuple[int, ...]
    data: Optional[np.ndarray] = None


class GGUFLoader:
    """
    Loads weights from GGUF model files.
    
    Supports loading quantized weights and converting them to
    phase-encoded format for the TPN engine.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize GGUF loader.
        
        Args:
            model_path: Path to GGUF model file.
        """
        self.model_path = model_path
        self._reader = None
        self._tensors = None
        self._metadata = None
    
    def _get_reader(self):
        """Get or create GGUF reader."""
        if self._reader is None:
            if GGUFReader is None:
                raise ImportError("gguf library not installed. Run: pip install gguf")
            self._reader = GGUFReader(self.model_path)
        return self._reader
    
    def get_tensor_names(self) -> List[str]:
        """Get list of all tensor names in the model."""
        reader = self._get_reader()
        return [tensor.name for tensor in reader.tensors]
    
    def load_tensor(self, name: str) -> Optional[TensorInfo]:
        """
        Load a single tensor by name.
        
        Args:
            name: Tensor name.
        
        Returns:
            TensorInfo with shape and data, or None if not found.
        """
        reader = self._get_reader()
        for tensor in reader.tensors:
            if tensor.name == name:
                # Read tensor data
                data = tensor.data
                return TensorInfo(
                    name=tensor.name,
                    shape=tuple(tensor.shape),
                    data=data
                )
        return None
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration from GGUF metadata.
        
        Returns:
            Dictionary with model configuration.
        """
        reader = self._get_reader()
        config = {}
        
        # Extract key metadata
        # GGUF field structure: [key_len, key_bytes, value_type, value_len, value_data]
        for field_name, field in reader.fields.items():
            parts = field.parts
            if field_name == 'general.architecture':
                # String value is in parts[4]
                if len(parts) > 4:
                    val = parts[4]
                    if isinstance(val, np.ndarray):
                        config['architecture'] = val.tobytes().decode('utf-8')
                    elif isinstance(val, bytes):
                        config['architecture'] = val.decode('utf-8')
                    else:
                        config['architecture'] = str(val)
            elif field_name == 'qwen2.embedding_length':
                # UINT32 value is in parts[3]
                config['hidden_size'] = int(np.ravel(parts[3])[0])
            elif field_name == 'qwen2.block_count':
                config['num_layers'] = int(np.ravel(parts[3])[0])
            elif field_name == 'qwen2.feed_forward_length':
                config['intermediate_size'] = int(np.ravel(parts[3])[0])
            elif field_name == 'qwen2.attention.head_count':
                config['num_heads'] = int(np.ravel(parts[3])[0])
            elif field_name == 'qwen2.attention.head_count_kv':
                config['num_kv_heads'] = int(np.ravel(parts[3])[0])
            elif field_name == 'qwen2.context_length':
                config['max_position_embeddings'] = int(np.ravel(parts[3])[0])
        
        # Compute derived values
        if 'hidden_size' in config and 'num_heads' in config:
            config['head_dim'] = config['hidden_size'] // config['num_heads']
        
        return config
    
    def extract_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract all weights from the model.
        
        Returns:
            Dictionary mapping weight names to numpy arrays.
        """
        reader = self._get_reader()
        weights = {}
        
        for tensor in reader.tensors:
            weights[tensor.name] = tensor.data
        
        return weights
    
    def extract_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """
        Extract weights for a specific layer.
        
        Args:
            layer_idx: Layer index.
        
        Returns:
            Dictionary with layer weights.
        """
        reader = self._get_reader()
        prefix = f'blk.{layer_idx}.'
        weights = {}
        
        for tensor in reader.tensors:
            if tensor.name.startswith(prefix):
                # Remove prefix for cleaner names
                name = tensor.name[len(prefix):]
                weights[name] = tensor.data
        
        return weights
    
    def to_phase_encoded(self, weights: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Convert weights to phase-encoded format.
        
        Args:
            weights: Dictionary of weight arrays.
        
        Returns:
            Dictionary of phase-encoded weights.
        """
        from .phase import PhaseEncoding
        
        phase_weights = {}
        for name, data in weights.items():
            if data is not None and len(data.shape) == 2:
                # Convert to list of lists
                data_list = data.tolist()
                # Encode as phase weights
                phase_weights[name] = PhaseEncoding.encode_matrix(data_list)
        
        return phase_weights
    
    def load_into_tpn_model(self, tpn_model) -> None:
        """
        Load weights into a TPN model.
        
        Args:
            tpn_model: TPNModel instance to load weights into.
        """
        from .phase import PhaseEncoding
        
        reader = self._get_reader()
        config = self.get_model_config()
        
        # Load embedding weights
        token_embd = self.load_tensor('token_embd.weight')
        if token_embd is not None and token_embd.data is not None:
            # Convert to phase-encoded format
            data_list = token_embd.data.tolist()
            phase_weights = PhaseEncoding.encode_matrix(data_list)
            tpn_model.embedding = phase_weights
        
        # Load layer weights
        for layer_idx in range(tpn_model.config.num_layers):
            layer_weights = self.extract_layer_weights(layer_idx)
            
            # Convert to phase-encoded format
            for name, data in layer_weights.items():
                if data is not None and len(data.shape) == 2:
                    data_list = data.tolist()
                    phase_weights = PhaseEncoding.encode_matrix(data_list)
                    # Store in model (implementation depends on TPNModel structure)
                    if not hasattr(tpn_model, 'phase_weights'):
                        tpn_model.phase_weights = {}
                    tpn_model.phase_weights[f'blk.{layer_idx}.{name}'] = phase_weights
        
        # Load output weights
        output = self.load_tensor('output.weight')
        if output is not None and output.data is not None:
            data_list = output.data.tolist()
            phase_weights = PhaseEncoding.encode_matrix(data_list)
            tpn_model.output_weights = phase_weights
