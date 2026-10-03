"""
Bitsliced parallel operations for TPN.

Enables running multiple operations in parallel using the same gate
by bitslicing the inputs. This is similar to SIMD (Single Instruction
Multiple Data) but applied to TPN gates.

Key concepts:
- BitslicedGate: A gate that processes N independent operations simultaneously
- BitslicedOperation: An operation composed of multiple bitsliced gates
- ParallelExecutor: Executes multiple operations in parallel with dependency tracking
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field


@dataclass
class BitslicedGate:
    """
    A gate that processes multiple independent operations simultaneously.
    
    Each "slice" represents an independent operation that shares the same
    gate type but has different inputs.
    """
    gate_type: str
    num_slices: int
    input_a: List[int]
    input_b: List[int]
    
    def execute(self) -> List[int]:
        """Execute the bitsliced gate on all slices."""
        results = []
        for i in range(self.num_slices):
            a = self.input_a[i] if i < len(self.input_a) else 0
            b = self.input_b[i] if i < len(self.input_b) else 0
            results.append(self._execute_single(a, b))
        return results
    
    def _execute_single(self, a: int, b: int) -> int:
        """Execute gate on a single slice."""
        if self.gate_type == "AND":
            return a & b
        elif self.gate_type == "OR":
            return a | b
        elif self.gate_type == "XOR":
            return a ^ b
        elif self.gate_type == "NAND":
            return int(not (a & b))
        elif self.gate_type == "NOR":
            return int(not (a | b))
        elif self.gate_type == "NOT":
            return int(not a)
        elif self.gate_type == "BUF":
            return a
        else:
            return 0


@dataclass
class BitslicedOperation:
    """
    An operation composed of multiple bitsliced gates.
    
    Each gate processes the same number of slices, enabling
    parallel execution of multiple independent operations.
    """
    name: str
    num_slices: int
    gate_type: str
    gates: List[BitslicedGate] = field(default_factory=list)
    
    def add_gate(self, gate_type: str, input_a: List[int], input_b: List[int]) -> None:
        """Add a gate to the operation."""
        gate = BitslicedGate(
            gate_type=gate_type,
            num_slices=self.num_slices,
            input_a=input_a,
            input_b=input_b,
        )
        self.gates.append(gate)
    
    def execute(self) -> List[List[int]]:
        """Execute all gates in the operation."""
        results = []
        for gate in self.gates:
            results.append(gate.execute())
        return results
    
    def get_parallelism(self) -> int:
        """Get the parallelism level (number of gates × slices)."""
        return len(self.gates) * self.num_slices


class ParallelExecutor:
    """
    Executes multiple bitsliced operations in parallel.
    
    Supports:
    - Parallel execution of independent operations
    - Dependency tracking between operations
    - Parallelism evolution to find optimal configuration
    """
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
    
    def execute_parallel(
        self,
        operations: List[BitslicedOperation],
        dependencies: Optional[Dict[str, List[str]]] = None,
    ) -> List[List[List[int]]]:
        """
        Execute operations in parallel, respecting dependencies.
        
        Args:
            operations: List of operations to execute.
            dependencies: Dictionary mapping operation names to lists of
                operation names that must complete first.
        
        Returns:
            List of results for each operation.
        """
        if dependencies is None:
            dependencies = {}
        
        # Build dependency graph
        op_map = {op.name: op for op in operations}
        completed: Set[str] = set()
        results: Dict[str, List[List[int]]] = {}
        
        # Execute in topological order
        pending = set(op.name for op in operations)
        
        while pending:
            # Find operations with all dependencies satisfied
            ready = set()
            for name in pending:
                deps = dependencies.get(name, [])
                if all(d in completed for d in deps):
                    ready.add(name)
            
            if not ready:
                # Circular dependency or missing dependency
                break
            
            # Execute ready operations (up to num_workers at a time)
            batch = list(ready)[:self.num_workers]
            
            for name in batch:
                op = op_map[name]
                results[name] = op.execute()
                completed.add(name)
                pending.remove(name)
        
        # Return results in original order
        return [results.get(op.name, []) for op in operations]
    
    def compute_parallelism_score(self, operation: BitslicedOperation) -> float:
        """
        Compute parallelism score for an operation.
        
        Score = (number of gates × number of slices) / (number of gates + number of slices)
        
        Higher score = more parallelism.
        """
        num_gates = len(operation.gates)
        num_slices = operation.num_slices
        
        if num_gates == 0 or num_slices == 0:
            return 0.0
        
        # Parallelism score: product / sum (harmonic mean-like)
        return (num_gates * num_slices) / (num_gates + num_slices)
    
    def evolve_parallelism(
        self,
        operations: List[BitslicedOperation],
        generations: int = 10,
        mutation_rate: float = 0.1,
    ) -> BitslicedOperation:
        """
        Evolve operations to find the best parallelism configuration.
        
        Uses a simple (1+1) evolution strategy:
        1. Create mutated copies of operations
        2. Compute parallelism scores
        3. Keep the best
        4. Repeat
        
        Args:
            operations: Initial operations.
            generations: Number of evolution generations.
            mutation_rate: Rate of mutation.
        
        Returns:
            Best operation found.
        """
        if not operations:
            return None
        
        # Start with the operation that has the highest parallelism
        best_op = max(operations, key=self.compute_parallelism_score)
        best_score = self.compute_parallelism_score(best_op)
        
        for gen in range(generations):
            # Create mutated copy
            mutated = self._mutate_operation(best_op, mutation_rate)
            
            # Compute score
            score = self.compute_parallelism_score(mutated)
            
            # Keep if better
            if score > best_score:
                best_op = mutated
                best_score = score
        
        # Add parallelism score as attribute
        best_op.parallelism_score = best_score
        return best_op
    
    def _mutate_operation(
        self,
        operation: BitslicedOperation,
        mutation_rate: float,
    ) -> BitslicedOperation:
        """Create a mutated copy of an operation."""
        import copy
        
        # Deep copy
        mutated = BitslicedOperation(
            name=operation.name + "_mut",
            num_slices=operation.num_slices,
            gate_type=operation.gate_type,
        )
        
        # Copy gates with mutation
        for gate in operation.gates:
            # Mutate inputs
            new_a = []
            new_b = []
            for a, b in zip(gate.input_a, gate.input_b):
                if random.random() < mutation_rate:
                    new_a.append(random.randint(0, 1))
                else:
                    new_a.append(a)
                if random.random() < mutation_rate:
                    new_b.append(random.randint(0, 1))
                else:
                    new_b.append(b)
            
            mutated.add_gate(gate.gate_type, new_a, new_b)
        
        return mutated
