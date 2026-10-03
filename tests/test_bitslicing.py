"""
Tests for bitsliced parallel operations.
"""

import math
import pytest
from tpn_engine.bitslicing import BitslicedGate, BitslicedOperation, ParallelExecutor


class TestBitslicedGate:
    """Test bitsliced gate operations."""
    
    def test_bitsliced_gate_creation(self):
        """Test creating a bitsliced gate."""
        # Create a bitsliced AND gate with 4 slices
        gate = BitslicedGate(
            gate_type="AND",
            num_slices=4,
            input_a=[1, 0, 1, 0],
            input_b=[1, 1, 0, 0],
        )
        
        assert gate.gate_type == "AND"
        assert gate.num_slices == 4
        assert len(gate.input_a) == 4
        assert len(gate.input_b) == 4
    
    def test_bitsliced_gate_execution(self):
        """Test executing a bitsliced gate."""
        gate = BitslicedGate(
            gate_type="AND",
            num_slices=4,
            input_a=[1, 0, 1, 0],
            input_b=[1, 1, 0, 0],
        )
        
        output = gate.execute()
        
        # AND: [1&1, 0&1, 1&0, 0&0] = [1, 0, 0, 0]
        assert output == [1, 0, 0, 0]
    
    def test_bitsliced_gate_or(self):
        """Test bitsliced OR gate."""
        gate = BitslicedGate(
            gate_type="OR",
            num_slices=4,
            input_a=[1, 0, 1, 0],
            input_b=[1, 1, 0, 0],
        )
        
        output = gate.execute()
        
        # OR: [1|1, 0|1, 1|0, 0|0] = [1, 1, 1, 0]
        assert output == [1, 1, 1, 0]
    
    def test_bitsliced_gate_xor(self):
        """Test bitsliced XOR gate."""
        gate = BitslicedGate(
            gate_type="XOR",
            num_slices=4,
            input_a=[1, 0, 1, 0],
            input_b=[1, 1, 0, 0],
        )
        
        output = gate.execute()
        
        # XOR: [1^1, 0^1, 1^0, 0^0] = [0, 1, 1, 0]
        assert output == [0, 1, 1, 0]
    
    def test_bitsliced_gate_nand(self):
        """Test bitsliced NAND gate."""
        gate = BitslicedGate(
            gate_type="NAND",
            num_slices=4,
            input_a=[1, 0, 1, 0],
            input_b=[1, 1, 0, 0],
        )
        
        output = gate.execute()
        
        # NAND: [!(1&1), !(0&1), !(1&0), !(0&0)] = [0, 1, 1, 1]
        assert output == [0, 1, 1, 1]


class TestBitslicedOperation:
    """Test bitsliced operations."""
    
    def test_operation_creation(self):
        """Test creating a bitsliced operation."""
        op = BitslicedOperation(
            name="add",
            num_slices=4,
            gate_type="AND",
        )
        
        assert op.name == "add"
        assert op.num_slices == 4
        assert op.gate_type == "AND"
    
    def test_operation_with_multiple_gates(self):
        """Test operation with multiple gates."""
        op = BitslicedOperation(
            name="add",
            num_slices=4,
            gate_type="AND",
        )
        
        # Add multiple gates
        op.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
        op.add_gate("OR", [1, 0, 1, 0], [1, 1, 0, 0])
        
        assert len(op.gates) == 2
    
    def test_operation_execution(self):
        """Test executing a bitsliced operation."""
        op = BitslicedOperation(
            name="add",
            num_slices=4,
            gate_type="AND",
        )
        
        op.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
        op.add_gate("OR", [1, 0, 1, 0], [1, 1, 0, 0])
        
        results = op.execute()
        
        assert len(results) == 2
        assert results[0] == [1, 0, 0, 0]  # AND
        assert results[1] == [1, 1, 1, 0]  # OR


class TestParallelExecutor:
    """Test parallel executor."""
    
    def test_executor_creation(self):
        """Test creating a parallel executor."""
        executor = ParallelExecutor(num_workers=4)
        assert executor.num_workers == 4
    
    def test_execute_parallel(self):
        """Test executing operations in parallel."""
        executor = ParallelExecutor(num_workers=4)
        
        # Create multiple operations
        ops = []
        for i in range(4):
            op = BitslicedOperation(
                name=f"op_{i}",
                num_slices=4,
                gate_type="AND",
            )
            op.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
            ops.append(op)
        
        # Execute in parallel
        results = executor.execute_parallel(ops)
        
        assert len(results) == 4
        for result in results:
            assert result == [[1, 0, 0, 0]]
    
    def test_execute_parallel_with_dependencies(self):
        """Test executing operations with dependencies."""
        executor = ParallelExecutor(num_workers=4)
        
        # Create operations with dependencies
        op1 = BitslicedOperation(name="op1", num_slices=4, gate_type="AND")
        op1.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
        
        op2 = BitslicedOperation(name="op2", num_slices=4, gate_type="OR")
        op2.add_gate("OR", [1, 0, 1, 0], [1, 1, 0, 0])
        
        # op2 depends on op1
        dependencies = {"op2": ["op1"]}
        
        results = executor.execute_parallel([op1, op2], dependencies)
        
        assert len(results) == 2
    
    def test_parallelism_evolution(self):
        """Test evolving parallelism."""
        executor = ParallelExecutor(num_workers=4)
        
        # Create operations with different parallelism levels
        ops = []
        for i in range(4):
            op = BitslicedOperation(
                name=f"op_{i}",
                num_slices=4,
                gate_type="AND",
            )
            # Add multiple gates to increase parallelism
            for j in range(i + 1):
                op.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
            ops.append(op)
        
        # Evolve to find best parallelism
        best = executor.evolve_parallelism(ops, generations=10)
        
        assert best is not None
        assert hasattr(best, 'parallelism_score')
    
    def test_parallelism_score(self):
        """Test parallelism score calculation."""
        executor = ParallelExecutor(num_workers=4)
        
        # Create operations with different parallelism
        op1 = BitslicedOperation(name="op1", num_slices=4, gate_type="AND")
        op1.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
        
        op2 = BitslicedOperation(name="op2", num_slices=4, gate_type="AND")
        for i in range(4):
            op2.add_gate("AND", [1, 0, 1, 0], [1, 1, 0, 0])
        
        score1 = executor.compute_parallelism_score(op1)
        score2 = executor.compute_parallelism_score(op2)
        
        # op2 has more gates, so higher parallelism
        assert score2 > score1
