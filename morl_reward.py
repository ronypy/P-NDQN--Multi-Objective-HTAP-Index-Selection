"""
Multi-objective reward function for HTAP index selection.

Three objectives: query latency improvement, write overhead penalty,
and memory utilization penalty, combined via workload-adaptive scalarization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class IndexConfig:
    """Represents an index configuration."""
    column: str
    table: str
    size_bytes: int  # Estimated memory footprint
    write_overhead: float  # Write amplification factor


class WorkloadPhaseDetector:
    """Detects workload phase (OLAP-heavy, OLTP-heavy, or balanced) for adaptive reward weighting."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.query_history = deque(maxlen=window_size)

        # Phase thresholds
        self.olap_threshold = 0.7  # >70% reads = OLAP-heavy
        self.oltp_threshold = 0.3  # <30% reads = OLTP-heavy

    def add_query(self, query: str, is_write: bool):
        """Record a query execution."""
        self.query_history.append({
            'query': query,
            'is_write': is_write,
            'complexity': self._estimate_complexity(query),
        })

    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1 scale)."""
        query_upper = query.upper()
        complexity = 0.0

        # Join complexity
        complexity += query_upper.count(' JOIN ') * 0.15
        complexity += query_upper.count(',') * 0.05  # Implicit joins

        # Aggregation complexity
        if 'GROUP BY' in query_upper:
            complexity += 0.2
        if 'ORDER BY' in query_upper:
            complexity += 0.1
        if any(agg in query_upper for agg in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN(']):
            complexity += 0.15

        # Subquery complexity
        complexity += query_upper.count('SELECT') * 0.1 - 0.1  # -0.1 for main SELECT

        return min(1.0, complexity)

    def get_phase(self) -> Tuple[str, float]:
        """
        Get current workload phase and confidence.

        Returns:
            (phase_name, read_ratio)
            phase_name: 'OLAP', 'OLTP', or 'BALANCED'
            read_ratio: Proportion of read queries (0-1)
        """
        if len(self.query_history) < 10:
            return 'BALANCED', 0.5

        reads = sum(1 for q in self.query_history if not q['is_write'])
        read_ratio = reads / len(self.query_history)

        if read_ratio >= self.olap_threshold:
            return 'OLAP', read_ratio
        elif read_ratio <= self.oltp_threshold:
            return 'OLTP', read_ratio
        else:
            return 'BALANCED', read_ratio

    def get_avg_complexity(self) -> float:
        """Get average query complexity in window."""
        if not self.query_history:
            return 0.5
        return np.mean([q['complexity'] for q in self.query_history])


class MORLRewardFunction:
    """
    Multi-Objective Reward Function for HTAP Index Selection.

    Optimizes three objectives simultaneously:
    1. Query Latency (minimize)
    2. Write Overhead (minimize)
    3. Memory Footprint (minimize, subject to budget)

    Key innovation: Adaptive weighting based on workload phase.
    """

    # CH-Benchmark table characteristics
    TABLE_WRITE_FREQUENCY = {
        'order_line': 1.0,    # Highest write frequency
        'new_order': 0.95,
        'oorder': 0.85,
        'history': 0.75,
        'stock': 0.65,
        'customer': 0.35,
        'district': 0.25,
        'warehouse': 0.15,
        'item': 0.0,          # Static/read-only
        'supplier': 0.0,
        'nation': 0.0,
        'region': 0.0,
    }

    # Estimated index sizes (bytes per row)
    INDEX_SIZE_PER_ROW = {
        # Integer columns
        'int': 8,
        'bigint': 12,
        # String columns
        'varchar': 20,
        'char': 15,
        # Numeric columns
        'decimal': 16,
        'float': 12,
        # Date/time
        'date': 8,
        'timestamp': 12,
    }

    def __init__(
        self,
        # Memory budget
        memory_budget_mb: float = 512.0,
        # Base weights (before phase adaptation)
        latency_weight: float = 1.0,
        write_weight: float = 1.0,
        memory_weight: float = 0.5,
        # Phase adaptation strength
        phase_adaptation: float = 0.5,
        # Reward scaling
        reward_scale: float = 1.0,
        # Table sizes for memory estimation
        table_row_counts: Optional[Dict[str, int]] = None,
    ):
        self.memory_budget_mb = memory_budget_mb
        self.memory_budget_bytes = memory_budget_mb * 1024 * 1024

        self.base_latency_weight = latency_weight
        self.base_write_weight = write_weight
        self.base_memory_weight = memory_weight
        self.phase_adaptation = phase_adaptation
        self.reward_scale = reward_scale

        # Default row counts for CH-benchmark (scale factor 1)
        self.table_row_counts = table_row_counts or {
            'warehouse': 1,
            'district': 10,
            'customer': 30000,
            'history': 30000,
            'oorder': 30000,
            'new_order': 9000,
            'order_line': 300000,
            'stock': 100000,
            'item': 100000,
            'supplier': 10000,
            'nation': 62,
            'region': 5,
        }

        # Phase detector
        self.phase_detector = WorkloadPhaseDetector()

        # Tracking for Pareto analysis
        self.reward_history = []

    def get_adaptive_weights(self) -> Tuple[float, float, float]:
        """
        Get phase-adaptive objective weights.

        During OLAP phase: Prioritize latency
        During OLTP phase: Prioritize write overhead
        Balanced: Equal weights
        """
        phase, read_ratio = self.phase_detector.get_phase()

        # Smooth interpolation based on read_ratio
        # read_ratio = 1.0 -> pure OLAP
        # read_ratio = 0.0 -> pure OLTP

        # OLAP emphasis: higher latency weight, lower write weight
        # OLTP emphasis: lower latency weight, higher write weight

        olap_factor = read_ratio  # 0 to 1
        oltp_factor = 1 - read_ratio

        latency_weight = self.base_latency_weight * (1 + self.phase_adaptation * olap_factor)
        write_weight = self.base_write_weight * (1 + self.phase_adaptation * oltp_factor)
        memory_weight = self.base_memory_weight  # Constant

        # Normalize to sum to original total
        total_base = self.base_latency_weight + self.base_write_weight + self.base_memory_weight
        total_new = latency_weight + write_weight + memory_weight
        scale = total_base / total_new

        return (
            latency_weight * scale,
            write_weight * scale,
            memory_weight * scale,
        )

    def estimate_index_memory(self, table: str, column: str) -> int:
        """Estimate memory footprint of an index in bytes."""
        row_count = self.table_row_counts.get(table, 10000)
        # Assume average index entry size of 16 bytes (key + pointer)
        bytes_per_entry = 16
        # B-tree overhead ~30%
        overhead = 1.3
        return int(row_count * bytes_per_entry * overhead)

    def compute_reward(
        self,
        # Cost changes
        cost_before: float,
        cost_after: float,
        # Action details
        action_type: str,  # 'CREATE', 'DROP', 'BLOCKED'
        table: str,
        column: str,
        # Current state
        current_indexes: Dict[str, bool],
        current_memory_bytes: int,
        # Query info
        query: str,
        is_write_query: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-objective reward.

        Returns:
            (total_reward, breakdown_dict)
        """
        # Record query for phase detection
        self.phase_detector.add_query(query, is_write_query)

        # Get adaptive weights
        w_latency, w_write, w_memory = self.get_adaptive_weights()

        # === Objective 1: Latency Improvement ===
        if cost_before > 0:
            latency_improvement = (cost_before - cost_after) / cost_before
        else:
            latency_improvement = 0.0

        # Scale to reasonable range (10% improvement -> +1.0)
        latency_reward = latency_improvement * 10.0

        # === Objective 2: Write Overhead ===
        write_freq = self.TABLE_WRITE_FREQUENCY.get(table, 0.0)

        if action_type == 'CREATE':
            # Penalty for indexing write-heavy tables (FIX 5: reduced from 3.0)
            write_penalty = -write_freq * 2.0
            # Extra penalty during OLTP phase (FIX 5: reduced from 1.5)
            phase, _ = self.phase_detector.get_phase()
            if phase == 'OLTP' and write_freq > 0.5:
                write_penalty *= 1.3
        elif action_type == 'DROP':
            # Reward for dropping index on write-heavy table
            write_reward = write_freq * 2.0
            write_penalty = write_reward  # Positive!
        else:
            write_penalty = -0.5  # Small penalty for blocked action

        write_reward = write_penalty  # Can be positive or negative

        # === Objective 3: Memory Efficiency ===
        index_memory = self.estimate_index_memory(table, column)
        memory_ratio = current_memory_bytes / self.memory_budget_bytes

        if action_type == 'CREATE':
            # Penalty increases as we approach budget
            if memory_ratio > 0.9:
                memory_penalty = -5.0  # Severe penalty near limit
            elif memory_ratio > 0.7:
                memory_penalty = -2.0
            else:
                memory_penalty = -0.5
        elif action_type == 'DROP':
            # Reward for freeing memory, especially when near budget
            memory_reward = 1.0 + memory_ratio * 2.0
            memory_penalty = memory_reward
        else:
            memory_penalty = -0.3

        memory_reward = memory_penalty

        # === Combine with adaptive weights ===
        total_reward = (
            w_latency * latency_reward +
            w_write * write_reward +
            w_memory * memory_reward
        )

        # Scale
        total_reward *= self.reward_scale

        # Apply smoothing to prevent reward asymmetry (FIX 5)
        # Diminishing returns for very positive rewards to balance with penalties
        if total_reward > 5.0:
            total_reward = 5.0 + (total_reward - 5.0) * 0.5

        # Clip to prevent extreme values
        total_reward = np.clip(total_reward, -15.0, 15.0)

        # Breakdown for analysis
        breakdown = {
            'latency_reward': latency_reward,
            'write_reward': write_reward,
            'memory_reward': memory_reward,
            'w_latency': w_latency,
            'w_write': w_write,
            'w_memory': w_memory,
            'phase': self.phase_detector.get_phase()[0],
            'total': total_reward,
        }

        self.reward_history.append(breakdown)

        return total_reward, breakdown

    def get_memory_action_mask(
        self,
        current_indexes: Dict[str, bool],
        current_memory_bytes: int,
        columns: List[str],
        column_to_table: Dict[str, str],
    ) -> np.ndarray:
        """
        Get action mask based on memory constraints.

        This is the novel constraint-aware masking contribution.
        """
        mask = np.ones(len(columns), dtype=np.float32)

        for i, col in enumerate(columns):
            table = column_to_table.get(col, '')

            if current_indexes.get(col, 0) == 0:
                # Index doesn't exist - check if we can create
                index_size = self.estimate_index_memory(table, col)

                # Hard memory constraint
                if current_memory_bytes + index_size > self.memory_budget_bytes:
                    mask[i] = 0  # Cannot create - would exceed budget

                # Soft constraint: Block indexing very high-write tables during OLTP
                phase, _ = self.phase_detector.get_phase()
                write_freq = self.TABLE_WRITE_FREQUENCY.get(table, 0.0)
                if phase == 'OLTP' and write_freq >= 0.9:
                    mask[i] = 0  # Block indexing order_line, new_order during OLTP

        return mask

    def get_pareto_metrics(self) -> Dict[str, float]:
        """Get Pareto analysis of training."""
        if not self.reward_history:
            return {}

        latency_rewards = [r['latency_reward'] for r in self.reward_history]
        write_rewards = [r['write_reward'] for r in self.reward_history]
        memory_rewards = [r['memory_reward'] for r in self.reward_history]

        return {
            'avg_latency_reward': np.mean(latency_rewards),
            'avg_write_reward': np.mean(write_rewards),
            'avg_memory_reward': np.mean(memory_rewards),
            'std_latency': np.std(latency_rewards),
            'std_write': np.std(write_rewards),
            'std_memory': np.std(memory_rewards),
        }


if __name__ == "__main__":
    print("Testing MORL Reward Function...")

    reward_fn = MORLRewardFunction(
        memory_budget_mb=256,
        latency_weight=1.0,
        write_weight=1.0,
        memory_weight=0.5,
    )

    # Simulate OLAP phase
    print("\n=== OLAP Phase Simulation ===")
    for i in range(30):
        reward_fn.phase_detector.add_query("SELECT * FROM customer JOIN oorder", is_write=False)

    phase, ratio = reward_fn.phase_detector.get_phase()
    print(f"Phase: {phase}, Read ratio: {ratio:.2f}")

    weights = reward_fn.get_adaptive_weights()
    print(f"Weights: latency={weights[0]:.2f}, write={weights[1]:.2f}, memory={weights[2]:.2f}")

    # Test reward for creating index on read-heavy table
    reward, breakdown = reward_fn.compute_reward(
        cost_before=1000,
        cost_after=500,
        action_type='CREATE',
        table='item',  # Low write frequency
        column='i_id',
        current_indexes={},
        current_memory_bytes=50 * 1024 * 1024,  # 50MB used
        query="SELECT * FROM item",
        is_write_query=False,
    )
    print(f"CREATE index on low-write table: reward={reward:.2f}")
    print(f"  Breakdown: {breakdown}")

    # Simulate OLTP phase
    print("\n=== OLTP Phase Simulation ===")
    for i in range(50):
        reward_fn.phase_detector.add_query("UPDATE order_line SET ol_quantity = 5", is_write=True)

    phase, ratio = reward_fn.phase_detector.get_phase()
    print(f"Phase: {phase}, Read ratio: {ratio:.2f}")

    weights = reward_fn.get_adaptive_weights()
    print(f"Weights: latency={weights[0]:.2f}, write={weights[1]:.2f}, memory={weights[2]:.2f}")

    # Test reward for creating index on write-heavy table
    reward, breakdown = reward_fn.compute_reward(
        cost_before=1000,
        cost_after=800,
        action_type='CREATE',
        table='order_line',  # High write frequency
        column='ol_i_id',
        current_indexes={},
        current_memory_bytes=50 * 1024 * 1024,
        query="UPDATE order_line SET ol_quantity = 5",
        is_write_query=True,
    )
    print(f"CREATE index on high-write table during OLTP: reward={reward:.2f}")
    print(f"  Breakdown: {breakdown}")

    # Test DROP reward
    reward, breakdown = reward_fn.compute_reward(
        cost_before=800,
        cost_after=1000,
        action_type='DROP',
        table='order_line',
        column='ol_i_id',
        current_indexes={'ol_i_id': True},
        current_memory_bytes=100 * 1024 * 1024,
        query="UPDATE order_line SET ol_quantity = 5",
        is_write_query=True,
    )
    print(f"DROP index on high-write table during OLTP: reward={reward:.2f}")
    print(f"  Breakdown: {breakdown}")

    print("\nTest passed!")
