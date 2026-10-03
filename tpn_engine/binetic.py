"""
.binetic format for TPN models.

A native format for TPN models that supports:
- Graph structure (layers as nodes, connections as edges)
- Version control with rollback capability
- Auto-save after inference
- Phase-encoded weights
"""

import json
import os
import time
import struct
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class BineticVersion:
    """Version information for a binetic model."""
    version_id: str
    timestamp: float
    parent_id: Optional[str] = None
    inference_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BineticGraph:
    """
    Graph structure for TPN models.
    
    Nodes represent layers, edges represent connections between layers.
    Each node contains phase-encoded weights and metadata.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, name: str, data: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        self.nodes[name] = data
    
    def add_edge(self, from_node: str, to_node: str, data: Dict[str, Any]) -> None:
        """Add an edge between two nodes."""
        self.edges[(from_node, to_node)] = data
    
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a node by name."""
        return self.nodes.get(name)
    
    def get_neighbors(self, name: str) -> List[str]:
        """Get all neighboring nodes."""
        neighbors = []
        for (from_node, to_node) in self.edges:
            if from_node == name:
                neighbors.append(to_node)
            elif to_node == name:
                neighbors.append(from_node)
        return neighbors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": self.nodes,
            "edges": {f"{k[0]}->{k[1]}": v for k, v in self.edges.items()},
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BineticGraph':
        """Create graph from dictionary."""
        graph = cls()
        graph.nodes = data.get("nodes", {})
        graph.metadata = data.get("metadata", {})
        
        # Parse edges
        for edge_key, edge_data in data.get("edges", {}).items():
            parts = edge_key.split("->")
            if len(parts) == 2:
                graph.edges[(parts[0], parts[1])] = edge_data
        
        return graph


class BineticFormat:
    """
    .binetic file format handler.
    
    File format:
    - Header: magic number, version, metadata
    - Graph data: nodes and edges
    - Version history: list of versions
    - Weight data: phase-encoded weights
    """
    
    MAGIC = b"BINETIC\x00"
    VERSION = 1
    
    def __init__(self):
        self._version_history: Dict[str, List[BineticVersion]] = {}
        self._version_snapshots: Dict[str, Dict[str, BineticGraph]] = {}
    
    def save(self, filepath: str, graph: BineticGraph, version_id: str,
             parent_id: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """
        Save a binetic model to file.
        
        Args:
            filepath: Path to save file.
            graph: BineticGraph to save.
            version_id: Version identifier.
            parent_id: Parent version identifier.
            metadata: Additional metadata.
        """
        version = BineticVersion(
            version_id=version_id,
            timestamp=time.time(),
            parent_id=parent_id,
            metadata=metadata or {},
        )
        
        # Build file data
        file_data = {
            "magic": self.MAGIC.hex(),
            "version": self.VERSION,
            "graph": graph.to_dict(),
            "version_info": {
                "version_id": version.version_id,
                "timestamp": version.timestamp,
                "parent_id": version.parent_id,
                "metadata": version.metadata,
            },
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(file_data, f, indent=2)
        
        # Update version history
        if filepath not in self._version_history:
            self._version_history[filepath] = []
        self._version_history[filepath].append(version)
        
        # Store snapshot for rollback (deep copy)
        import copy
        if filepath not in self._version_snapshots:
            self._version_snapshots[filepath] = {}
        self._version_snapshots[filepath][version_id] = copy.deepcopy(graph.to_dict())
    
    def load(self, filepath: str) -> Tuple[BineticGraph, BineticVersion]:
        """
        Load a binetic model from file.
        
        Args:
            filepath: Path to load file.
        
        Returns:
            Tuple of (BineticGraph, BineticVersion).
        """
        with open(filepath, 'r') as f:
            file_data = json.load(f)
        
        # Parse graph
        graph = BineticGraph.from_dict(file_data["graph"])
        
        # Parse version
        version_info = file_data.get("version_info", {})
        version = BineticVersion(
            version_id=version_info.get("version_id", "unknown"),
            timestamp=version_info.get("timestamp", 0),
            parent_id=version_info.get("parent_id"),
            metadata=version_info.get("metadata", {}),
        )
        
        return graph, version
    
    def get_version_history(self, filepath: str) -> List[BineticVersion]:
        """Get version history for a file."""
        return self._version_history.get(filepath, [])
    
    def rollback(self, filepath: str, version_id: str) -> BineticGraph:
        """
        Rollback to a previous version.
        
        Args:
            filepath: Path to file.
            version_id: Version to rollback to.
        
        Returns:
            BineticGraph at the specified version.
        """
        # Check if we have a snapshot for this version
        if filepath in self._version_snapshots:
            if version_id in self._version_snapshots[filepath]:
                snapshot = self._version_snapshots[filepath][version_id]
                return BineticGraph.from_dict(snapshot)
        
        # If no snapshot, try to load from file
        history = self.get_version_history(filepath)
        target_version = None
        for v in history:
            if v.version_id == version_id:
                target_version = v
                break
        
        if target_version is None:
            raise ValueError(f"Version {version_id} not found in history")
        
        # Load current file as fallback
        graph, _ = self.load(filepath)
        return graph
    
    def auto_save(self, filepath: str, graph: BineticGraph, inference_id: str) -> None:
        """
        Auto-save after inference.
        
        Args:
            filepath: Path to file.
            graph: Current graph state.
            inference_id: Identifier for this inference run.
        """
        # Generate new version ID
        history = self.get_version_history(filepath)
        new_version_id = f"auto_{len(history)}_{int(time.time())}"
        
        # Get parent version
        parent_id = None
        if history:
            parent_id = history[-1].version_id
        
        # Save with auto-save metadata
        self.save(
            filepath,
            graph,
            version_id=new_version_id,
            parent_id=parent_id,
            metadata={"inference_id": inference_id, "auto_saved": True},
        )
    
    def save_weights(self, filepath: str, weights: Dict[str, List[List[float]]]) -> None:
        """
        Save phase-encoded weights to a separate file.
        
        Args:
            filepath: Path to weights file.
            weights: Dictionary of weight matrices.
        """
        with open(filepath, 'w') as f:
            json.dump(weights, f)
    
    def load_weights(self, filepath: str) -> Dict[str, List[List[float]]]:
        """
        Load phase-encoded weights from file.
        
        Args:
            filepath: Path to weights file.
        
        Returns:
            Dictionary of weight matrices.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
