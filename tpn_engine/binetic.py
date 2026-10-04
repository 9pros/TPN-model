"""
.binetic format for TPN models.

A native format for TPN models that supports:
- Graph structure (layers as nodes, connections as edges)
- Version control with rollback capability
- Auto-save after inference
- Phase-encoded weights
- Binetic ownership identity (binetic.ai / Max Yeremenko)
"""

import json
import os
import time
import hashlib
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import math

# Identity constants for every model authored, trained, or converted by Binetic.
BINETIC_IDENTITY = {
    "owner": "binetic.ai",
    "partners": ["Max Yeremenko", "binetic-partner"],
    "brand": "Binetic",
    "purpose": "Temporal Packet Network inference engine",
    "derived_from": None,           # set on conversion: source model name
    "tool": "binetic_converter",    # "binetic_converter" | "binetic_train"
    "identity_tag": "__binetic__",  # sentinel key that proves Binetic authorship
    "coauthors": "Max Yeremenko (CTO) & binetic.ai",
}


@dataclass
class BineticVersion:
    """Version information for a binetic model."""

    version_id: str
    timestamp: float
    parent_id: Optional[str] = None
    inference_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BineticIdentity:
    """Identity of a Binetic-authored model."""

    owner: str = "binetic.ai"
    partners: List[str] = field(default_factory=lambda: ["Max Yeremenko",
                                                         "binetic-partner"])
    brand: str = "Binetic"
    derived_from: Optional[str] = None  # source model name (e.g. "Ling-3.0-tiny")
    tool: Optional[str] = None          # "binetic_converter" | "binetic_train"
    created_at: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None  # evolution/gen info

    @property
    def tag(self) -> str:
        return "__binetic__"

    @property
    def is_binetic(self) -> bool:
        return True

    @property
    def coauthors(self) -> str:
        return f"{', '.join(self.partners)} (binetic.ai)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "owner": self.owner,
            "partners": self.partners,
            "brand": self.brand,
            "derived_from": self.derived_from,
            "tool": self.tool,
            "created_at": self.created_at or time.time(),
            "generation_info": self.generation_info,
            "tag": self.tag,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BineticIdentity":
        return cls(
            owner=data.get("owner", "binetic.ai"),
            partners=data.get("partners", ["Max Yeremenko", "binetic-partner"]),
            brand=data.get("brand", "Binetic"),
            derived_from=data.get("derived_from"),
            tool=data.get("tool"),
            created_at=data.get("created_at"),
            generation_info=data.get("generation_info"),
        )


@dataclass
class GenerationRecord:
    """Record of an evolution/generation step (kept for Binetic provenance)."""

    generation: int
    fitness: float
    mutation_rate: float
    target_hash: Optional[str] = None
    timestamp: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "fitness": self.fitness,
            "mutation_rate": self.mutation_rate,
            "target_hash": self.target_hash,
            "timestamp": self.timestamp or time.time(),
        }


class BineticGraph:
    """
    Graph structure for TPN models.

    Nodes represent layers, edges represent connections between layers.
    Each node contains phase-encoded weights and metadata.
    """

    def __init__(self, identity: Optional[BineticIdentity] = None):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
        self.identity: BineticIdentity = (
            identity if identity is not None else BineticIdentity(tool="manual"))

    # ------------------------------------------------------------------ #
    # Binetic identity API
    # ------------------------------------------------------------------ #

    def set_binetic_identity(
        self, owner: str = "binetic.ai",
        partners: Optional[List[str]] = None,
        derived_from: Optional[str] = None,
        tool: Optional[str] = None,
        generation_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Assert/replay Binetic ownership on this graph."""
        self.identity = BineticIdentity(
            owner=owner,
            partners=partners or ["Max Yeremenko", "binetic-partner"],
            derived_from=derived_from,
            tool=tool,
            generation_info=generation_info,
        )
        self.metadata["__binetic__"] = self.identity.as_dict()

    @property
    def identity(self):
        return self._identity

    @identity.setter
    def identity(self, value: BineticIdentity) -> None:
        self._identity = value
        self.metadata["__binetic__"] = value.as_dict()

    def get_binetic_identity(self) -> Optional[BineticIdentity]:
        """Return the Binetic identity if this graph is Binetic-authored."""
        return self.identity if self.identity.is_binetic else None

    def is_binetic(self) -> bool:
        """True if this model was authored/trained/converted by Binetic."""
        return self.identity.is_binetic

    def register_generation(self, record: GenerationRecord) -> None:
        """Register an evolution generation for provenance."""
        generations = self.metadata.setdefault("binetic_generations", [])
        generations.append(record.as_dict())
        generations.sort(key=lambda r: r["generation"])
        if self.identity.generation_info is None:
            self.identity.generation_info = {"generations": 0,
                                             "best_fitness": None}
        info = self.identity.generation_info
        info["generations"] = max(info["generations"],
                                  record.generation + 1)
        if info["best_fitness"] is None or record.fitness > info["best_fitness"]:
            info["best_fitness"] = record.fitness

    # ------------------------------------------------------------------ #
    # Graph API
    # ------------------------------------------------------------------ #

    def add_node(self, name: str, data: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        self.nodes[name] = data

    def add_edge(self, from_node: str, to_node: str,
                 data: Dict[str, Any]) -> None:
        """Add an edge between two nodes."""
        self.edges[(from_node, to_node)] = data

    def get_neighbors(self, name: str) -> List[str]:
        """Get all neighboring nodes."""
        neighbors = []
        for (from_node, to_node) in self.edges:
            if from_node == name:
                neighbors.append(to_node)
            elif to_node == name:
                neighbors.append(from_node)
        return neighbors

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a node by name."""
        return self.nodes.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "nodes": self.nodes,
            "edges": {f"{k[0]}->{k[1]}": v for k, v in self.edges.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BineticGraph":
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
    - Header: magic number, version, identity, metadata
    - Graph data: nodes and edges
    - Version history: list of versions
    - Weight data: phase-encoded weights
    """

    MAGIC = b"BINE2C\x00"
    VERSION = 2  # v2 adds Binetic identity to the header

    def __init__(self, identity: Optional[BineticIdentity] = None):
        self._version_history: Dict[str, List[BineticVersion]] = {}
        self._version_snapshots: Dict[str, Dict[str, BineticGraph]] = {}
        self._default_identity = identity

    # ------------------------------------------------------------------ #
    # Binetic identity handling
    # ------------------------------------------------------------------ #

    def _ensure_identity(self) -> BineticIdentity:
        """Return the default identity or a fresh Binetic one."""
        return (self._default_identity if self._default_identity
                else BineticIdentity(tool="BineticFormat.save"))

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, filepath: str, graph: BineticGraph, version_id: str,
             parent_id: Optional[str] = None,
             inference_id: Optional[str] = None,
             metadata: Optional[Dict] = None) -> None:
        """
        Save a binetic model to file.

        Every saved .binetic file carries the Binetic identity in the
        header so any loader can prove Binetic authorship.
        """
        identity = graph.identity if graph.identity.is_binetic\
            else self._ensure_identity()

        version = BineticVersion(
            version_id=version_id,
            timestamp=time.time(),
            parent_id=parent_id,
            inference_id=inference_id,
            metadata=metadata or {},
        )

        # Build file data: identity lives in the header, not just the graph.
        file_data = {
            "magic": self.MAGIC.hex(),
            "version": self.VERSION,
            "identity": identity.as_dict(),
            "graph": graph.to_dict(),
            "version_info": {
                "version_id": version.version_id,
                "timestamp": version.timestamp,
                "parent_id": version.parent_id,
                "inference_id": inference_id,
                "metadata": version.metadata,
            },
        }

        with open(filepath, "w") as f:
            json.dump(file_data, f, indent=2)

        if filepath not in self._version_history:
            self._version_history[filepath] = []
        self._version_history[filepath].append(version)

        import copy
        if filepath not in self._version_snapshots:
            self._version_snapshots[filepath] = {}
        self._version_snapshots[filepath][version_id] = copy.deepcopy(
            graph.to_dict())

    def load(self, filepath: str) -> Tuple[BineticGraph, BineticVersion]:
        """
        Load a binetic model from file.

        Returns:
            Tuple of (BineticGraph, BineticVersion).
        """
        with open(filepath, "r") as f:
            file_data = json.load(f)

        # Parse graph identity first (graph.from_dict resets metadata).
        identity_data = file_data.get("identity")
        graph = BineticGraph.from_dict(file_data["graph"])
        if identity_data:
            graph.identity = BineticIdentity.from_dict(identity_data)

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
        """Rollback to a previous version."""
        if filepath in self._version_snapshots:
            if version_id in self._version_snapshots[filepath]:
                snapshot = self._version_snapshots[filepath][version_id]
                return BineticGraph.from_dict(snapshot)

        history = self.get_version_history(filepath)
        target_version = None
        for v in history:
            if v.version_id == version_id:
                target_version = v
                break

        if target_version is None:
            raise ValueError(f"Version {version_id} not found in history")

        graph, _ = self.load(filepath)
        return graph

    def auto_save(self, filepath: str, graph: BineticGraph,
                  inference_id: str) -> None:
        """Auto-save after inference, preserving Binetic identity."""
        history = self.get_version_history(filepath)
        new_version_id = f"auto_{len(history)}_{int(time.time())}"
        parent_id = None
        if history:
            parent_id = history[-1].version_id
        self.save(
            filepath, graph, version_id=new_version_id, parent_id=parent_id,
            inference_id=inference_id,
            metadata={"inference_id": inference_id, "auto_saved": True},
        )

    def save_weights(self, filepath: str,
                     weights: Dict[str, List[List[float]]]) -> None:
        """Save phase-encoded weights to a separate file."""
        with open(filepath, "w") as f:
            json.dump(weights, f)

    def load_weights(self, filepath: str) -> Dict[str, List[List[float]]]:
        """Load phase-encoded weights from file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def identity_of(self, filepath: str) -> Optional[BineticIdentity]:
        """Return the Binetic identity recorded in a .binetic file, if any."""
        with open(filepath, "r") as f:
            file_data = json.load(f)
        identity_data = file_data.get("identity")
        if identity_data:
            return BineticIdentity.from_dict(identity_data)
        return None

    @property
    def default_identity(self) -> Optional[BineticIdentity]:
        return self._default_identity

    def set_default_identity(self, identity: BineticIdentity) -> None:
        """Set the identity that every newly-saved graph will carry."""
        self._default_identity = identity
