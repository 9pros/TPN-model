"""
Tests for .binetic format.
"""

import os
import tempfile
import pytest
from tpn_engine.binetic import BineticFormat, BineticGraph, BineticVersion


class TestBineticGraph:
    """Test binetic graph structure."""
    
    def test_graph_creation(self):
        """Test creating a binetic graph."""
        graph = BineticGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = BineticGraph()
        graph.add_node("layer_0", {"weights": [[0.1, 0.2], [0.3, 0.4]]})
        assert "layer_0" in graph.nodes
        assert graph.nodes["layer_0"]["weights"] == [[0.1, 0.2], [0.3, 0.4]]
    
    def test_add_edge(self):
        """Test adding an edge between nodes."""
        graph = BineticGraph()
        graph.add_node("layer_0", {"weights": [[0.1, 0.2]]})
        graph.add_node("layer_1", {"weights": [[0.3, 0.4]]})
        graph.add_edge("layer_0", "layer_1", {"type": "sequential"})
        assert ("layer_0", "layer_1") in graph.edges
        assert graph.edges[("layer_0", "layer_1")]["type"] == "sequential"
    
    def test_get_node(self):
        """Test getting a node by name."""
        graph = BineticGraph()
        graph.add_node("layer_0", {"weights": [[0.1, 0.2]]})
        node = graph.get_node("layer_0")
        assert node is not None
        assert node["weights"] == [[0.1, 0.2]]
    
    def test_get_neighbors(self):
        """Test getting neighboring nodes."""
        graph = BineticGraph()
        graph.add_node("layer_0", {})
        graph.add_node("layer_1", {})
        graph.add_node("layer_2", {})
        graph.add_edge("layer_0", "layer_1", {})
        graph.add_edge("layer_1", "layer_2", {})
        
        neighbors = graph.get_neighbors("layer_1")
        assert "layer_0" in neighbors
        assert "layer_2" in neighbors


class TestBineticVersion:
    """Test binetic version control."""
    
    def test_version_creation(self):
        """Test creating a version."""
        version = BineticVersion(version_id="v1", timestamp=1234567890)
        assert version.version_id == "v1"
        assert version.timestamp == 1234567890
    
    def test_version_with_parent(self):
        """Test creating a version with parent."""
        version = BineticVersion(version_id="v2", timestamp=1234567891, parent_id="v1")
        assert version.parent_id == "v1"


class TestBineticFormat:
    """Test .binetic file format."""
    
    def test_format_creation(self):
        """Test creating a binetic format handler."""
        fmt = BineticFormat()
        assert fmt is not None
    
    def test_save_and_load(self):
        """Test saving and loading a .binetic file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.binetic")
            
            # Create and save
            fmt = BineticFormat()
            graph = BineticGraph()
            graph.add_node("layer_0", {"weights": [[0.1, 0.2], [0.3, 0.4]]})
            graph.add_node("layer_1", {"weights": [[0.5, 0.6], [0.7, 0.8]]})
            graph.add_edge("layer_0", "layer_1", {"type": "sequential"})
            
            fmt.save(filepath, graph, version_id="v1")
            assert os.path.exists(filepath)
            
            # Load
            loaded_graph, loaded_version = fmt.load(filepath)
            assert "layer_0" in loaded_graph.nodes
            assert "layer_1" in loaded_graph.nodes
            assert loaded_version.version_id == "v1"
    
    def test_version_history(self):
        """Test version history tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.binetic")
            
            fmt = BineticFormat()
            graph = BineticGraph()
            graph.add_node("layer_0", {"weights": [[0.1, 0.2]]})
            
            # Save v1
            fmt.save(filepath, graph, version_id="v1")
            
            # Modify and save v2
            graph.nodes["layer_0"]["weights"] = [[0.3, 0.4]]
            fmt.save(filepath, graph, version_id="v2", parent_id="v1")
            
            # Check history
            history = fmt.get_version_history(filepath)
            assert len(history) >= 2
    
    def test_rollback(self):
        """Test rollback to previous version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.binetic")
            
            fmt = BineticFormat()
            graph = BineticGraph()
            graph.add_node("layer_0", {"weights": [[0.1, 0.2]]})
            
            # Save v1
            fmt.save(filepath, graph, version_id="v1")
            
            # Modify and save v2
            graph.nodes["layer_0"]["weights"] = [[0.3, 0.4]]
            fmt.save(filepath, graph, version_id="v2", parent_id="v1")
            
            # Rollback to v1
            rolled_back = fmt.rollback(filepath, "v1")
            assert rolled_back.nodes["layer_0"]["weights"] == [[0.1, 0.2]]
    
    def test_auto_save_after_inference(self):
        """Test auto-save after inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.binetic")
            
            fmt = BineticFormat()
            graph = BineticGraph()
            graph.add_node("layer_0", {"weights": [[0.1, 0.2]]})
            
            # Save initial version
            fmt.save(filepath, graph, version_id="v1")
            
            # Simulate inference (modify weights)
            graph.nodes["layer_0"]["weights"] = [[0.15, 0.25]]
            
            # Auto-save
            fmt.auto_save(filepath, graph, inference_id="inf_001")
            
            # Check that new version was created
            history = fmt.get_version_history(filepath)
            assert len(history) >= 2
    
    def test_graph_with_metadata(self):
        """Test graph with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.binetic")
            
            fmt = BineticFormat()
            graph = BineticGraph()
            graph.add_node("layer_0", {"weights": [[0.1, 0.2]]})
            graph.metadata = {
                "model_name": "test_model",
                "hidden_size": 64,
                "num_layers": 1,
            }
            
            fmt.save(filepath, graph, version_id="v1")
            loaded_graph, _ = fmt.load(filepath)
            
            assert loaded_graph.metadata["model_name"] == "test_model"
            assert loaded_graph.metadata["hidden_size"] == 64
