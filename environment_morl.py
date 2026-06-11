"""
RL environment for HTAP index selection.

Implements multi-objective reward (latency, write overhead, memory),
workload-adaptive weight scalarization, and constraint-enforced action masking.
"""

import numpy as np
import os
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from pg_database import PG_Database
from morl_reward import MORLRewardFunction, WorkloadPhaseDetector


class MORLIndexEnvironment:
    """Multi-objective RL environment for HTAP index selection."""

    def __init__(
        self,
        workload_path: str,
        hypo: bool = True,
        # Episode configuration
        max_steps_per_episode: int = 50,
        # Constraint configuration
        max_indexes: int = 15,
        memory_budget_mb: float = 512.0,
        # Reward configuration
        latency_weight: float = 1.0,
        write_weight: float = 1.0,
        memory_weight: float = 0.5,
        phase_adaptation: float = 0.5,
        reward_scale: float = 1.0,
        # Workload configuration
        query_window_size: int = 20,
        # Ablation
        use_action_mask: bool = True,
    ):
        # Database connection
        self.db = PG_Database(hypo=hypo)
        self.use_action_mask = use_action_mask
        self.hypo = hypo

        # Schema info
        self.table_columns = self.db.tables
        self.tables = list(self.table_columns.keys())
        self.columns = []
        for table in self.tables:
            self.columns.extend(self.table_columns[table])

        self.column_to_table = {}
        for table, cols in self.table_columns.items():
            for col in cols:
                self.column_to_table[col] = table

        # Workload
        self.workload = self._load_workload(workload_path)
        self.query_window_size = query_window_size

        # Episode configuration
        self.max_steps_per_episode = max_steps_per_episode
        self.max_indexes = max_indexes
        self.memory_budget_mb = memory_budget_mb

        # Action space: One action per column (toggle index)
        self.n_columns = len(self.columns)
        self.n_actions = self.n_columns

        # State space:
        # - Index existence (n_columns binary)
        # - Column usage frequency (n_columns float)
        # - Workload phase features (3 floats: read_ratio, avg_complexity, write_ratio)
        # - Memory utilization (1 float)
        self.n_features = self.n_columns * 2 + 4

        # Multi-objective reward function
        self.reward_fn = MORLRewardFunction(
            memory_budget_mb=memory_budget_mb,
            latency_weight=latency_weight,
            write_weight=write_weight,
            memory_weight=memory_weight,
            phase_adaptation=phase_adaptation,
            reward_scale=reward_scale,
        )

        # Episode state
        self.current_step = 0
        self.workload_idx = 0
        self.episode_reward = 0
        self.current_memory_bytes = 0

        # Tracking
        self.action_history = []
        self.reward_breakdown_history = []

        # Action diversity tracking within episode (FIX 2)
        self.episode_action_counts = {}  # {action_idx: count}
        self.indexed_tables = set()  # Track unique tables with indexes

        # Diversity reward parameters (FIX 2)
        self.diversity_penalty_coef = 0.3  # Penalty for repeated actions
        self.table_diversity_bonus_coef = 0.1  # Bonus for unique tables

        print("=" * 60)
        print("MORL-INDEX ENVIRONMENT")
        print("=" * 60)
        print(f"Workload: {len(self.workload)} queries")
        print(f"Columns: {self.n_columns}")
        print(f"Actions: {self.n_actions}")
        print(f"Features: {self.n_features}")
        print(f"Max steps/episode: {max_steps_per_episode}")
        print(f"Max indexes: {max_indexes}")
        print(f"Memory budget: {memory_budget_mb} MB")
        print(f"Weights: latency={latency_weight}, write={write_weight}, memory={memory_weight}")
        print("=" * 60)

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        # Clear all indexes
        self.db.reset_indexes()

        # Reset state
        self.current_step = 0
        self.episode_reward = 0
        self.current_memory_bytes = 0
        self.action_history = []
        self.reward_breakdown_history = []

        # Reset diversity tracking (FIX 2)
        self.episode_action_counts = {}
        self.indexed_tables = set()

        # Random starting point in workload
        self.workload_idx = random.randint(0, len(self.workload) - 1)

        # Get state and action mask
        state = self._get_state()
        action_mask = self._get_action_mask()

        info = {
            'action_mask': action_mask,
            'phase': self.reward_fn.phase_detector.get_phase()[0],
            'memory_used_mb': self.current_memory_bytes / (1024 * 1024),
        }

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action.

        Returns:
            (state, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Get current state
        indexes_before = self.db.get_indexes()
        column = self.columns[action]
        table = self.column_to_table[column]

        # Get current query and check if write
        query = self.workload[self.workload_idx]
        is_write = self._is_write_query(query)

        # Measure cost before
        cost_before = self._compute_window_cost()

        # Execute action
        action_mask = self._get_action_mask()
        if action_mask[action] == 0:
            # Invalid action (should not happen with proper masking)
            action_type = 'BLOCKED'
        elif indexes_before[column] == 1:
            # Index exists -> DROP
            self.db.drop_index(table, column)
            action_type = 'DROP'
            # Update memory
            index_memory = self.reward_fn.estimate_index_memory(table, column)
            self.current_memory_bytes = max(0, self.current_memory_bytes - index_memory)
        else:
            # Index doesn't exist -> CREATE
            self.db.create_index(table, column)
            action_type = 'CREATE'
            # Update memory
            index_memory = self.reward_fn.estimate_index_memory(table, column)
            self.current_memory_bytes += index_memory

        # Measure cost after
        cost_after = self._compute_window_cost()

        # Compute multi-objective reward
        reward, breakdown = self.reward_fn.compute_reward(
            cost_before=cost_before,
            cost_after=cost_after,
            action_type=action_type,
            table=table,
            column=column,
            current_indexes=self.db.get_indexes(),
            current_memory_bytes=self.current_memory_bytes,
            query=query,
            is_write_query=is_write,
        )

        # Track action counts for diversity (FIX 2)
        self.episode_action_counts[action] = self.episode_action_counts.get(action, 0) + 1
        action_count = self.episode_action_counts[action]

        # Compute diversity penalty for repeated actions (FIX 2)
        diversity_penalty = 0.0
        if action_count > 1:
            diversity_penalty = -self.diversity_penalty_coef * (action_count - 1)

        # Track unique tables indexed for bonus (FIX 2)
        if action_type == 'CREATE':
            self.indexed_tables.add(table)

        # Compute table diversity bonus (FIX 2)
        table_diversity_bonus = 0.0
        if len(self.indexed_tables) > 1:
            table_diversity_bonus = self.table_diversity_bonus_coef * (len(self.indexed_tables) - 1)

        # Add diversity components to reward (FIX 2)
        reward += diversity_penalty + table_diversity_bonus

        self.episode_reward += reward
        self.action_history.append((action, action_type, column, table))
        self.reward_breakdown_history.append(breakdown)

        # Advance workload
        self._advance_workload()

        # Get new state
        state = self._get_state()

        # Termination
        terminated = False
        truncated = self.current_step >= self.max_steps_per_episode

        # Info
        indexes = self.db.get_indexes()
        index_count = sum(indexes.values())
        phase, read_ratio = self.reward_fn.phase_detector.get_phase()

        info = {
            'action_mask': self._get_action_mask(),
            'action_type': action_type,
            'column': column,
            'table': table,
            'cost_before': cost_before,
            'cost_after': cost_after,
            'index_count': index_count,
            'memory_used_mb': self.current_memory_bytes / (1024 * 1024),
            'phase': phase,
            'read_ratio': read_ratio,
            'reward_breakdown': breakdown,
            'step': self.current_step,
            # Diversity metrics (FIX 2)
            'diversity_penalty': diversity_penalty,
            'table_diversity_bonus': table_diversity_bonus,
            'unique_tables_indexed': len(self.indexed_tables),
            'action_repeat_count': action_count,
        }

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        Get state observation.

        State includes:
        - Index existence vector
        - Column usage frequency
        - Workload phase features
        - Memory utilization
        """
        # Index existence
        indexes = self.db.get_indexes()
        index_state = np.array([indexes[col] for col in self.columns], dtype=np.float32)

        # Column usage frequency
        usage = self._compute_column_usage()

        # Phase features
        phase, read_ratio = self.reward_fn.phase_detector.get_phase()
        avg_complexity = self.reward_fn.phase_detector.get_avg_complexity()
        write_ratio = 1.0 - read_ratio

        # Memory utilization
        memory_ratio = self.current_memory_bytes / (self.memory_budget_mb * 1024 * 1024)

        # Combine
        phase_features = np.array([read_ratio, write_ratio, avg_complexity, memory_ratio], dtype=np.float32)
        state = np.concatenate([index_state, usage, phase_features])

        return state

    def _get_action_mask(self) -> np.ndarray:
        """
        Get constraint-aware action mask.

        Masks invalid actions based on:
        1. Memory budget
        2. Index count limit
        3. Write-heavy table restrictions during OLTP
        """
        if not self.use_action_mask:
            return np.ones(self.n_actions, dtype=np.float32)

        indexes = self.db.get_indexes()
        index_count = sum(indexes.values())
        phase, _ = self.reward_fn.phase_detector.get_phase()

        mask = np.ones(self.n_actions, dtype=np.float32)

        for i, col in enumerate(self.columns):
            table = self.column_to_table.get(col, '')

            if indexes[col] == 0:
                # Index doesn't exist - check CREATE constraints

                # 1. Index count limit
                if index_count >= self.max_indexes:
                    mask[i] = 0
                    continue

                # 2. Memory budget
                index_memory = self.reward_fn.estimate_index_memory(table, col)
                if self.current_memory_bytes + index_memory > self.memory_budget_mb * 1024 * 1024:
                    mask[i] = 0
                    continue

                # 3. Block very high-write tables during OLTP phase
                write_freq = self.reward_fn.TABLE_WRITE_FREQUENCY.get(table, 0.0)
                if phase == 'OLTP' and write_freq >= 0.9:
                    mask[i] = 0
                    continue

            # DROP actions are always allowed for existing indexes

        # Ensure at least one action is valid
        if mask.sum() == 0:
            # Allow all DROP actions
            for i, col in enumerate(self.columns):
                if indexes[col] == 1:
                    mask[i] = 1

        return mask

    def _compute_column_usage(self) -> np.ndarray:
        """Compute column usage frequency in query window."""
        usage = np.zeros(self.n_columns, dtype=np.float32)

        for i in range(self.query_window_size):
            idx = (self.workload_idx + i) % len(self.workload)
            query = self.workload[idx].upper()

            for j, col in enumerate(self.columns):
                if col.upper() in query:
                    usage[j] += 1

        # Normalize
        if usage.sum() > 0:
            usage = usage / usage.max()

        return usage

    def _compute_window_cost(self) -> float:
        """Compute total cost for queries in window."""
        total_cost = 0.0
        for i in range(self.query_window_size):
            idx = (self.workload_idx + i) % len(self.workload)
            query = self.workload[idx]
            try:
                cost = self.db.get_query_cost(query)
                total_cost += cost
            except Exception:
                total_cost += 1000
        return total_cost

    def _advance_workload(self):
        """Advance to next query."""
        self.workload_idx = (self.workload_idx + 1) % len(self.workload)

    def _is_write_query(self, query: str) -> bool:
        """Check if query is a write operation."""
        q_upper = query.strip().upper()
        return any(q_upper.startswith(kw) for kw in ['INSERT', 'UPDATE', 'DELETE'])

    def _load_workload(self, path: str) -> List[str]:
        """Load workload from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Workload not found: {path}")

        with open(path, 'r') as f:
            data = f.read()

        queries = []
        for q in data.split('\n'):
            q = q.strip()
            if q and not q.startswith('--'):
                queries.append(q)

        return queries

    def get_final_metrics(self) -> Dict:
        """Get metrics at end of episode."""
        indexes = self.db.get_indexes()
        active_indexes = [col for col, v in indexes.items() if v]

        return {
            'episode_reward': self.episode_reward,
            'index_count': len(active_indexes),
            'active_indexes': active_indexes,
            'memory_used_mb': self.current_memory_bytes / (1024 * 1024),
            'memory_budget_mb': self.memory_budget_mb,
            'memory_utilization': self.current_memory_bytes / (self.memory_budget_mb * 1024 * 1024),
            'pareto_metrics': self.reward_fn.get_pareto_metrics(),
        }

    def close(self):
        """Close environment."""
        self.db.reset_indexes()
        self.db.close_connection()

    @property
    def observation_space_shape(self) -> Tuple[int]:
        return (self.n_features,)

    @property
    def action_space_n(self) -> int:
        return self.n_actions


if __name__ == "__main__":
    print("Testing MORL-Index Environment...")

    try:
        env = MORLIndexEnvironment(
            workload_path='data/workload/chbench_htap_train.sql',
            hypo=True,
            max_steps_per_episode=30,
            max_indexes=10,
            memory_budget_mb=256,
            latency_weight=1.0,
            write_weight=1.0,
            memory_weight=0.5,
        )

        # Run episode
        state, info = env.reset()
        print(f"Initial state shape: {state.shape}")
        print(f"Initial phase: {info['phase']}")

        total_reward = 0
        for step in range(30):
            mask = info['action_mask']
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) == 0:
                print("No valid actions!")
                break

            action = np.random.choice(valid_actions)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 5 == 0:
                print(f"Step {step}: action={action}, type={info['action_type']}, "
                      f"reward={reward:.2f}, indexes={info['index_count']}, "
                      f"phase={info['phase']}, mem={info['memory_used_mb']:.1f}MB")

            if truncated:
                break

        metrics = env.get_final_metrics()
        print(f"\n=== Episode Complete ===")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final indexes: {metrics['index_count']}")
        print(f"Memory used: {metrics['memory_used_mb']:.1f} / {metrics['memory_budget_mb']:.1f} MB")
        print(f"Pareto metrics: {metrics['pareto_metrics']}")

        env.close()
        print("\nTest passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
