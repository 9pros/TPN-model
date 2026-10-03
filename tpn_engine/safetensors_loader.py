"""
Safetensors weight loader for TPN.

Loads weights from safetensors model files (MLX format) and converts them
to phase-encoded format for the TPN engine.
"""

import json
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class SafetensorsLoader:
    """
    Loads weights from safetensors model files.
    
    Supports memory-efficient loading of individual tensors and layers.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize safetensors loader.
        
        Args:
            model_path: Path to safetensors model file.
        """
        self.model_path = model_path
        self._header = None
        self._header_size = None
        self._data_offset = None
    
    def _load_header(self) -> Dict:
        """Load and parse safetensors header."""
        if self._header is not None:
            return self._header
        
        with open(self.model_path, 'rb') as f:
            # First 8 bytes are header size
            self._header_size = struct.unpack('<Q', f.read(8))[0]
            header_bytes = f.read(self._header_size)
            self._header = json.loads(header_bytes.decode('utf-8'))
            self._data_offset = 8 + self._header_size
        
        return self._header
    
    def get_tensor_names(self) -> List[str]:
        """Get list of all tensor names in the model."""
        header = self._load_header()
        return [name for name in header.keys() if name != '__metadata__']
    
    def load_tensor(self, name: str) -> Optional[np.ndarray]:
        """
        Load a single tensor by name.
        
        Args:
            name: Tensor name.
        
        Returns:
            Numpy array with tensor data, or None if not found.
        """
        header = self._load_header()
        
        if name not in header:
            return None
        
        info = header[name]
        dtype = info.get('dtype', 'F32')
        shape = info.get('shape', [])
        data_offsets = info.get('data_offsets', [0, 0])
        
        # Calculate data size
        start, end = data_offsets
        data_size = end - start
        
        # Map dtype to numpy dtype
        dtype_map = {
            'F32': np.float32,
            'F16': np.float16,
            'U32': np.uint32,
            'U8': np.uint8,
            'I32': np.int32,
            'I64': np.int64,
            'F64': np.float64,
        }
        np_dtype = dtype_map.get(dtype, np.float32)
        
        # Read tensor data
        with open(self.model_path, 'rb') as f:
            f.seek(self._data_offset + start)
            data = np.frombuffer(f.read(data_size), dtype=np_dtype)
        
        # Reshape
        if shape:
            data = data.reshape(shape)
        
        return data
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration from companion config.json.
        
        Returns:
            Dictionary with model configuration.
        """
        import os
        config_path = os.path.join(os.path.dirname(self.model_path), 'config.json')
        
        if not os.path.exists(config_path):
            return {}
        
        with open(config_path) as f:
            config = json.load(f)
        
        text_config = config.get('text_config', {})
        
        return {
            'hidden_size': text_config.get('hidden_size', 5120),
            'num_layers': text_config.get('num_hidden_layers', 64),
            'num_heads': text_config.get('num_attention_heads', 24),
            'num_kv_heads': text_config.get('num_key_value_heads', 4),
            'head_dim': text_config.get('head_dim', 256),
            'intermediate_size': text_config.get('intermediate_size', 17408),
            'vocab_size': text_config.get('vocab_size', 248320),
            'max_position_embeddings': text_config.get('max_position_embeddings', 262144),
        }
    
    def extract_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """
        Extract weights for a specific layer.
        
        Args:
            layer_idx: Layer index.
        
        Returns:
            Dictionary with layer weights (short names).
        """
        header = self._load_header()
        prefix = f'language_model.model.layers.{layer_idx}.'
        weights = {}
        
        for name, info in header.items():
            if name.startswith(prefix):
                short_name = name[len(prefix):]
                tensor = self.load_tensor(name)
                if tensor is not None:
                    weights[short_name] = tensor
        
        return weights
    
    def extract_weights(self, max_layers: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract all weights from the model.
        
        Args:
            max_layers: Maximum number of layers to extract (None = all).
        
        Returns:
            Dictionary mapping weight names to numpy arrays.
        """
        header = self._load_header()
        weights = {}
        
        count = 0
        for name, info in header.items():
            if name == '__metadata__':
                continue
            
            # Check layer limit
            if max_layers is not None:
                # Extract layer index from name
                parts = name.split('.')
                if len(parts) > 4 and parts[3] == 'layers':
                    try:
                        layer_idx = int(parts[4])
                        if layer_idx >= max_layers:
                            continue
                    except (ValueError, IndexError):
                        pass
            
            tensor = self.load_tensor(name)
            if tensor is not None:
                weights[name] = tensor
                count += 1
        
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
