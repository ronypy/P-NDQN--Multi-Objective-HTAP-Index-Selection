#!/usr/bin/env python3
"""
Comprehensive Multi-Metric Evaluation for MORL-Index Paper


Metrics computed:
1. Multi-Objective Reward (primary metric)
2. Query Cost Reduction %
3. Write Overhead Score
4. Memory Efficiency (cost reduction per MB)
5. Index Efficiency (cost reduction per index)
6. Write Avoidance Ratio
7. Recommendation Time (ms)
8. Robustness (CV)
9. Pareto Dominance
10. OLAP/OLTP specific costs
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pg_database import PG_Database
from environment_morl import MORLIndexEnvironment
from train_morl_index import MORLIndexAgent
from baselines.drlinda_baseline import DRLindaAgent
from baselines.swirl_baseline import SWIRLAgent
from baselines.smartix_baseline import SmartIXAgent
from baselines.dba_bandits_baseline import DBABanditsAgent
from baselines.anytime_baseline import AnytimeAgent


# Table write frequency for CH-Benchmark
TABLE_WRITE_FREQUENCY = {
    'order_line': 1.0,
    'new_order': 0.95,
    'oorder': 0.85,
    'history': 0.75,
    'stock': 0.65,
    'customer': 0.35,
    'district': 0.25,
    'warehouse': 0.15,
    'item': 0.0,
    'supplier': 0.0,
    'nation': 0.0,
    'region': 0.0,
}


@dataclass
class ComprehensiveMetrics:
    """All metrics for a single method."""
    method: str
    # Primary metrics
    avg_reward: float = 0.0
    std_reward: float = 0.0
    cost_reduction_pct: float = 0.0
    write_overhead_score: float = 0.0
    memory_efficiency: float = 0.0
    index_efficiency: float = 0.0
    # Secondary metrics
    recommendation_time_ms: float = 0.0
    robustness_cv: float = 0.0
    write_avoidance_ratio: float = 0.0
    pareto_dominance: int = 0
    # HTAP metrics
    olap_cost_reduction: float = 0.0
    oltp_cost_reduction: float = 0.0
    # Raw data
    n_indexes: float = 0.0
    index_size_mb: float = 0.0
    n_episodes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ComprehensiveEvaluator:
    """Evaluate all methods on comprehensive metrics using EXPLAIN only."""

    def __init__(self, workload_path: str, memory_budget_mb: float = 100.0):
        self.workload_path = workload_path
        self.memory_budget_mb = memory_budget_mb
        self.workload = self._load_workload()
        self.column_to_table = {}
        self.baseline_cost = None
        self.olap_baseline = None
        self.oltp_baseline = None

    def _load_workload(self) -> List[str]:
        """Load SQL queries from workload file."""
        with open(self.workload_path, 'r') as f:
            content = f.read()

        queries = []
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            queries.append(line)

        print(f"Loaded {len(queries)} queries from {self.workload_path}")
        return queries

    def _classify_query(self, query: str) -> str:
        """Classify query as OLAP, OLTP_READ, or WRITE."""
        query_upper = query.upper()
        if query_upper.startswith(('UPDATE', 'INSERT', 'DELETE')):
            return 'WRITE'
        elif any(kw in query_upper for kw in ['GROUP BY', 'SUM(', 'AVG(', 'COUNT(', 'JOIN']):
            return 'OLAP'
        else:
            return 'OLTP_READ'

    def _build_column_mapping(self, db: PG_Database):
        """Build column to table mapping."""
        self.column_to_table = {}
        for table, columns in db.tables.items():
            for col in columns:
                self.column_to_table[col] = table

    def measure_baseline_costs(self) -> Tuple[float, float, float]:
        """Measure baseline costs with no indexes."""
        print("\n=== Measuring No-Index Baseline ===")
        db = PG_Database(hypo=True)
        db.reset_indexes()
        self._build_column_mapping(db)

        all_costs = []
        olap_costs = []
        oltp_costs = []

        for query in self.workload:
            qtype = self._classify_query(query)
            if qtype == 'WRITE':
                continue

            cost = db.get_query_cost(query)
            if cost != float('inf'):
                all_costs.append(cost)
                if qtype == 'OLAP':
                    olap_costs.append(cost)
                else:
                    oltp_costs.append(cost)

        db.close_connection()

        self.baseline_cost = np.mean(all_costs) if all_costs else 0
        self.olap_baseline = np.mean(olap_costs) if olap_costs else 0
        self.oltp_baseline = np.mean(oltp_costs) if oltp_costs else 0

        print(f"Baseline avg cost: {self.baseline_cost:.2f}")
        print(f"OLAP baseline: {self.olap_baseline:.2f}")
        print(f"OLTP baseline: {self.oltp_baseline:.2f}")

        return self.baseline_cost, self.olap_baseline, self.oltp_baseline

    def compute_write_overhead_score(self, indexed_columns: List[str]) -> float:
        """Compute total write overhead for indexed columns (lower is better)."""
        total = 0.0
        for col in indexed_columns:
            table = self.column_to_table.get(col, '')
            total += TABLE_WRITE_FREQUENCY.get(table, 0.5)
        return total

    def compute_write_avoidance_ratio(self, indexed_columns: List[str]) -> float:
        """Ratio of indexed tables with low write frequency (higher is better)."""
        if not indexed_columns:
            return 0.0
        low_write = 0
        for col in indexed_columns:
            table = self.column_to_table.get(col, '')
            if TABLE_WRITE_FREQUENCY.get(table, 0.5) < 0.5:
                low_write += 1
        return low_write / len(indexed_columns)

    def evaluate_with_indexes(self, db: PG_Database) -> Tuple[float, float, float]:
        """Evaluate workload with current indexes."""
        all_costs = []
        olap_costs = []
        oltp_costs = []

        for query in self.workload:
            qtype = self._classify_query(query)
            if qtype == 'WRITE':
                continue

            cost = db.get_query_cost(query)
            if cost != float('inf'):
                all_costs.append(cost)
                if qtype == 'OLAP':
                    olap_costs.append(cost)
                else:
                    oltp_costs.append(cost)

        avg_cost = np.mean(all_costs) if all_costs else 0
        olap_cost = np.mean(olap_costs) if olap_costs else 0
        oltp_cost = np.mean(oltp_costs) if oltp_costs else 0

        return avg_cost, olap_cost, oltp_cost

    def evaluate_morl(self, model_path: str, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate MORL-Index on all metrics."""
        print(f"\n=== Evaluating MORL-Index ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        agent = MORLIndexAgent(env=env, output_path=None)
        agent.load_model(model_path)

        all_rewards = []
        all_costs = []
        all_olap_costs = []
        all_oltp_costs = []
        all_write_overheads = []
        all_write_avoidance = []
        all_index_counts = []
        all_index_sizes = []
        recommendation_times = []
        all_actions = set()

        for ep in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(50):
                start = time.perf_counter()
                action = agent.select_action(state, env._get_action_mask(), training=False)
                end = time.perf_counter()
                recommendation_times.append((end - start) * 1000)
                all_actions.add(action)

                state, reward, done, _, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            all_rewards.append(episode_reward)

            # Get final metrics for this episode
            indexes = env.db.get_indexes()
            indexed_cols = [col for col, active in indexes.items() if active]

            avg_cost, olap_cost, oltp_cost = self.evaluate_with_indexes(env.db)
            all_costs.append(avg_cost)
            all_olap_costs.append(olap_cost)
            all_oltp_costs.append(oltp_cost)

            write_overhead = self.compute_write_overhead_score(indexed_cols)
            write_avoid = self.compute_write_avoidance_ratio(indexed_cols)
            all_write_overheads.append(write_overhead)
            all_write_avoidance.append(write_avoid)

            all_index_counts.append(len(indexed_cols))
            all_index_sizes.append(env.db.get_index_size_mb())

        env.close()

        # Compute metrics
        avg_reward = np.mean(all_rewards)
        avg_cost = np.mean(all_costs)
        cost_reduction = (self.baseline_cost - avg_cost) / self.baseline_cost * 100 if self.baseline_cost > 0 else 0
        olap_reduction = (self.olap_baseline - np.mean(all_olap_costs)) / self.olap_baseline * 100 if self.olap_baseline > 0 else 0
        oltp_reduction = (self.oltp_baseline - np.mean(all_oltp_costs)) / self.oltp_baseline * 100 if self.oltp_baseline > 0 else 0

        avg_indexes = np.mean(all_index_counts)
        avg_size = np.mean(all_index_sizes)
        index_eff = cost_reduction / avg_indexes if avg_indexes > 0 else 0
        memory_eff = cost_reduction / max(avg_size, 0.01)

        return ComprehensiveMetrics(
            method='MORL-Index',
            avg_reward=avg_reward,
            std_reward=np.std(all_rewards),
            cost_reduction_pct=cost_reduction,
            write_overhead_score=np.mean(all_write_overheads),
            memory_efficiency=memory_eff,
            index_efficiency=index_eff,
            recommendation_time_ms=np.mean(recommendation_times),
            robustness_cv=np.std(all_rewards) / abs(avg_reward) if avg_reward != 0 else float('inf'),
            write_avoidance_ratio=np.mean(all_write_avoidance),
            olap_cost_reduction=olap_reduction,
            oltp_cost_reduction=oltp_reduction,
            n_indexes=avg_indexes,
            index_size_mb=avg_size,
            n_episodes=n_episodes
        )

    def evaluate_drlinda(self, model_path: str, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate DRLinda on all metrics."""
        print(f"\n=== Evaluating DRLinda ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        agent = DRLindaAgent(env=env, output_path=None)
        agent.load_model(model_path)

        all_rewards = []
        all_costs = []
        all_olap_costs = []
        all_oltp_costs = []
        all_write_overheads = []
        all_write_avoidance = []
        all_index_counts = []
        all_index_sizes = []
        recommendation_times = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(50):
                start = time.perf_counter()
                action = agent.select_action(state, env._get_action_mask(), training=False)
                end = time.perf_counter()
                recommendation_times.append((end - start) * 1000)

                state, reward, done, _, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            all_rewards.append(episode_reward)

            indexes = env.db.get_indexes()
            indexed_cols = [col for col, active in indexes.items() if active]

            avg_cost, olap_cost, oltp_cost = self.evaluate_with_indexes(env.db)
            all_costs.append(avg_cost)
            all_olap_costs.append(olap_cost)
            all_oltp_costs.append(oltp_cost)

            all_write_overheads.append(self.compute_write_overhead_score(indexed_cols))
            all_write_avoidance.append(self.compute_write_avoidance_ratio(indexed_cols))
            all_index_counts.append(len(indexed_cols))
            all_index_sizes.append(env.db.get_index_size_mb())

        env.close()

        avg_reward = np.mean(all_rewards)
        avg_cost = np.mean(all_costs)
        cost_reduction = (self.baseline_cost - avg_cost) / self.baseline_cost * 100 if self.baseline_cost > 0 else 0
        olap_reduction = (self.olap_baseline - np.mean(all_olap_costs)) / self.olap_baseline * 100 if self.olap_baseline > 0 else 0
        oltp_reduction = (self.oltp_baseline - np.mean(all_oltp_costs)) / self.oltp_baseline * 100 if self.oltp_baseline > 0 else 0

        avg_indexes = np.mean(all_index_counts)
        avg_size = np.mean(all_index_sizes)

        return ComprehensiveMetrics(
            method='DRLinda',
            avg_reward=avg_reward,
            std_reward=np.std(all_rewards),
            cost_reduction_pct=cost_reduction,
            write_overhead_score=np.mean(all_write_overheads),
            memory_efficiency=cost_reduction / max(avg_size, 0.01),
            index_efficiency=cost_reduction / avg_indexes if avg_indexes > 0 else 0,
            recommendation_time_ms=np.mean(recommendation_times),
            robustness_cv=np.std(all_rewards) / abs(avg_reward) if avg_reward != 0 else float('inf'),
            write_avoidance_ratio=np.mean(all_write_avoidance),
            olap_cost_reduction=olap_reduction,
            oltp_cost_reduction=oltp_reduction,
            n_indexes=avg_indexes,
            index_size_mb=avg_size,
            n_episodes=n_episodes
        )

    def evaluate_random(self, n_episodes: int = 20, n_indexes: int = 15) -> ComprehensiveMetrics:
        """Evaluate Random baseline."""
        print(f"\n=== Evaluating Random ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        all_rewards = []
        all_costs = []
        all_olap_costs = []
        all_oltp_costs = []
        all_write_overheads = []
        all_write_avoidance = []
        all_index_counts = []
        all_index_sizes = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(n_indexes):
                action = np.random.randint(0, env.action_space_n)
                state, reward, done, _, info = env.step(action)
                episode_reward += reward
                if done:
                    break

            all_rewards.append(episode_reward)

            indexes = env.db.get_indexes()
            indexed_cols = [col for col, active in indexes.items() if active]

            avg_cost, olap_cost, oltp_cost = self.evaluate_with_indexes(env.db)
            all_costs.append(avg_cost)
            all_olap_costs.append(olap_cost)
            all_oltp_costs.append(oltp_cost)

            all_write_overheads.append(self.compute_write_overhead_score(indexed_cols))
            all_write_avoidance.append(self.compute_write_avoidance_ratio(indexed_cols))
            all_index_counts.append(len(indexed_cols))
            all_index_sizes.append(env.db.get_index_size_mb())

        env.close()

        avg_reward = np.mean(all_rewards)
        avg_cost = np.mean(all_costs)
        cost_reduction = (self.baseline_cost - avg_cost) / self.baseline_cost * 100 if self.baseline_cost > 0 else 0
        olap_reduction = (self.olap_baseline - np.mean(all_olap_costs)) / self.olap_baseline * 100 if self.olap_baseline > 0 else 0
        oltp_reduction = (self.oltp_baseline - np.mean(all_oltp_costs)) / self.oltp_baseline * 100 if self.oltp_baseline > 0 else 0

        avg_indexes = np.mean(all_index_counts)
        avg_size = np.mean(all_index_sizes)

        return ComprehensiveMetrics(
            method='Random',
            avg_reward=avg_reward,
            std_reward=np.std(all_rewards),
            cost_reduction_pct=cost_reduction,
            write_overhead_score=np.mean(all_write_overheads),
            memory_efficiency=cost_reduction / max(avg_size, 0.01),
            index_efficiency=cost_reduction / avg_indexes if avg_indexes > 0 else 0,
            recommendation_time_ms=0.1,
            robustness_cv=np.std(all_rewards) / abs(avg_reward) if avg_reward != 0 else float('inf'),
            write_avoidance_ratio=np.mean(all_write_avoidance),
            olap_cost_reduction=olap_reduction,
            oltp_cost_reduction=oltp_reduction,
            n_indexes=avg_indexes,
            index_size_mb=avg_size,
            n_episodes=n_episodes
        )

    def evaluate_heuristic(self, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate Heuristic baseline with known good indexes."""
        print(f"\n=== Evaluating Heuristic ===")

        # Known good indexes for CH-benchmark HTAP
        heuristic_indexes = [
            ('order_line', 'ol_delivery_d'),
            ('oorder', 'o_carrier_id'),
            ('order_line', 'ol_amount'),
            ('order_line', 'ol_quantity'),
            ('oorder', 'o_ol_cnt'),
            ('stock', 's_quantity'),
        ]

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        all_rewards = []
        all_costs = []
        all_olap_costs = []
        all_oltp_costs = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            # Apply heuristic indexes
            for table, column in heuristic_indexes:
                try:
                    action = env.columns.index(column)
                    state, reward, done, _, info = env.step(action)
                    episode_reward += reward
                except (ValueError, IndexError):
                    continue

            all_rewards.append(episode_reward)

            avg_cost, olap_cost, oltp_cost = self.evaluate_with_indexes(env.db)
            all_costs.append(avg_cost)
            all_olap_costs.append(olap_cost)
            all_oltp_costs.append(oltp_cost)

        env.close()

        # Get indexed columns for metrics
        indexed_cols = [col for _, col in heuristic_indexes]
        write_overhead = self.compute_write_overhead_score(indexed_cols)
        write_avoid = self.compute_write_avoidance_ratio(indexed_cols)

        avg_reward = np.mean(all_rewards)
        avg_cost = np.mean(all_costs)
        cost_reduction = (self.baseline_cost - avg_cost) / self.baseline_cost * 100 if self.baseline_cost > 0 else 0
        olap_reduction = (self.olap_baseline - np.mean(all_olap_costs)) / self.olap_baseline * 100 if self.olap_baseline > 0 else 0
        oltp_reduction = (self.oltp_baseline - np.mean(all_oltp_costs)) / self.oltp_baseline * 100 if self.oltp_baseline > 0 else 0

        return ComprehensiveMetrics(
            method='Heuristic',
            avg_reward=avg_reward,
            std_reward=np.std(all_rewards),
            cost_reduction_pct=cost_reduction,
            write_overhead_score=write_overhead,
            memory_efficiency=cost_reduction / 1.0,  # Estimated 1MB for 6 indexes
            index_efficiency=cost_reduction / len(heuristic_indexes),
            recommendation_time_ms=0.0,
            robustness_cv=0.0,  # Deterministic
            write_avoidance_ratio=write_avoid,
            olap_cost_reduction=olap_reduction,
            oltp_cost_reduction=oltp_reduction,
            n_indexes=len(heuristic_indexes),
            index_size_mb=1.0,
            n_episodes=n_episodes
        )

    def evaluate_no_index(self) -> ComprehensiveMetrics:
        """Evaluate No-Index baseline."""
        return ComprehensiveMetrics(
            method='No-Index',
            avg_reward=0.0,
            std_reward=0.0,
            cost_reduction_pct=0.0,
            write_overhead_score=0.0,
            memory_efficiency=0.0,
            index_efficiency=0.0,
            recommendation_time_ms=0.0,
            robustness_cv=0.0,
            write_avoidance_ratio=1.0,  # No indexes = no write overhead
            olap_cost_reduction=0.0,
            oltp_cost_reduction=0.0,
            n_indexes=0,
            index_size_mb=0.0,
            n_episodes=1
        )

    def _evaluate_agent_episodes(self, agent, env, n_episodes: int, method_name: str) -> ComprehensiveMetrics:
        """Generic evaluation loop for any agent with select_action interface."""
        all_rewards = []
        all_costs = []
        all_olap_costs = []
        all_oltp_costs = []
        all_write_overheads = []
        all_write_avoidance = []
        all_index_counts = []
        all_index_sizes = []
        recommendation_times = []

        for ep in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(50):
                start = time.perf_counter()
                action = agent.select_action(state, env._get_action_mask(), training=False)
                end = time.perf_counter()
                recommendation_times.append((end - start) * 1000)

                state, reward, done, _, info = env.step(action)
                episode_reward += reward

                if done:
                    break

            all_rewards.append(episode_reward)

            indexes = env.db.get_indexes()
            indexed_cols = [col for col, active in indexes.items() if active]

            avg_cost, olap_cost, oltp_cost = self.evaluate_with_indexes(env.db)
            all_costs.append(avg_cost)
            all_olap_costs.append(olap_cost)
            all_oltp_costs.append(oltp_cost)

            all_write_overheads.append(self.compute_write_overhead_score(indexed_cols))
            all_write_avoidance.append(self.compute_write_avoidance_ratio(indexed_cols))
            all_index_counts.append(len(indexed_cols))
            all_index_sizes.append(env.db.get_index_size_mb())

        avg_reward = np.mean(all_rewards)
        avg_cost = np.mean(all_costs)
        cost_reduction = (self.baseline_cost - avg_cost) / self.baseline_cost * 100 if self.baseline_cost > 0 else 0
        olap_reduction = (self.olap_baseline - np.mean(all_olap_costs)) / self.olap_baseline * 100 if self.olap_baseline > 0 else 0
        oltp_reduction = (self.oltp_baseline - np.mean(all_oltp_costs)) / self.oltp_baseline * 100 if self.oltp_baseline > 0 else 0

        avg_indexes = np.mean(all_index_counts)
        avg_size = np.mean(all_index_sizes)

        return ComprehensiveMetrics(
            method=method_name,
            avg_reward=avg_reward,
            std_reward=np.std(all_rewards),
            cost_reduction_pct=cost_reduction,
            write_overhead_score=np.mean(all_write_overheads),
            memory_efficiency=cost_reduction / max(avg_size, 0.01),
            index_efficiency=cost_reduction / avg_indexes if avg_indexes > 0 else 0,
            recommendation_time_ms=np.mean(recommendation_times),
            robustness_cv=np.std(all_rewards) / abs(avg_reward) if avg_reward != 0 else float('inf'),
            write_avoidance_ratio=np.mean(all_write_avoidance),
            olap_cost_reduction=olap_reduction,
            oltp_cost_reduction=oltp_reduction,
            n_indexes=avg_indexes,
            index_size_mb=avg_size,
            n_episodes=n_episodes
        )

    def evaluate_swirl(self, model_path: str, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate SWIRL (PPO, single-objective) on all metrics."""
        print(f"\n=== Evaluating SWIRL ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        agent = SWIRLAgent(env=env, output_path=None)
        agent.load_model(model_path)

        result = self._evaluate_agent_episodes(agent, env, n_episodes, 'SWIRL')
        env.close()
        return result

    def evaluate_smartix(self, model_path: str, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate SmartIX (DQN, single-objective) on all metrics."""
        print(f"\n=== Evaluating SmartIX ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        agent = SmartIXAgent(env=env, output_path=None)
        agent.load_model(model_path)

        result = self._evaluate_agent_episodes(agent, env, n_episodes, 'SmartIX')
        env.close()
        return result

    def evaluate_dba_bandits(self, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate DBA Bandits (C3UCB, online learning) on all metrics."""
        print(f"\n=== Evaluating DBA Bandits ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        # DBA Bandits learns online - run some warmup episodes first
        agent = DBABanditsAgent(env=env, output_path=None, n_episodes=100)
        for _ in range(100):
            agent.train_episode()

        result = self._evaluate_agent_episodes(agent, env, n_episodes, 'DBA_Bandits')
        env.close()
        return result

    def evaluate_anytime(self, n_episodes: int = 20) -> ComprehensiveMetrics:
        """Evaluate Anytime (greedy enumeration) on all metrics."""
        print(f"\n=== Evaluating Anytime ===")

        env = MORLIndexEnvironment(
            workload_path=self.workload_path,
            hypo=True,
            memory_budget_mb=self.memory_budget_mb
        )
        self._build_column_mapping(env.db)

        agent = AnytimeAgent(env=env, output_path=None)
        agent.recommend_indexes()

        result = self._evaluate_agent_episodes(agent, env, n_episodes, 'Anytime')
        env.close()
        return result

    def compute_pareto_dominance(self, metrics: ComprehensiveMetrics, all_metrics: List[ComprehensiveMetrics]) -> int:
        """Count how many methods this one Pareto-dominates."""
        dominated = 0
        for other in all_metrics:
            if other.method == metrics.method:
                continue
            # Three objectives: reward (max), write_overhead (min), memory_efficiency (max)
            better_reward = metrics.avg_reward >= other.avg_reward
            better_write = metrics.write_overhead_score <= other.write_overhead_score
            better_memory = metrics.memory_efficiency >= other.memory_efficiency

            if better_reward and better_write and better_memory:
                # Strictly better on at least one
                if (metrics.avg_reward > other.avg_reward or
                    metrics.write_overhead_score < other.write_overhead_score or
                    metrics.memory_efficiency > other.memory_efficiency):
                    dominated += 1
        return dominated

    def run_full_evaluation(
        self,
        morl_model_path: Optional[str] = None,
        drlinda_model_path: Optional[str] = None,
        swirl_model_path: Optional[str] = None,
        smartix_model_path: Optional[str] = None,
        n_episodes: int = 20
    ) -> Dict[str, ComprehensiveMetrics]:
        """Run comprehensive evaluation on all methods."""
        # Measure baselines first
        self.measure_baseline_costs()

        results = {}

        # Evaluate all methods
        results['No-Index'] = self.evaluate_no_index()
        # results['Heuristic'] = self.evaluate_heuristic(n_episodes)  # Dropped from 7-method comparison
        results['Random'] = self.evaluate_random(n_episodes)

        if drlinda_model_path and os.path.exists(drlinda_model_path):
            results['DRLinda'] = self.evaluate_drlinda(drlinda_model_path, n_episodes)

        if swirl_model_path and os.path.exists(swirl_model_path):
            results['SWIRL'] = self.evaluate_swirl(swirl_model_path, n_episodes)

        if smartix_model_path and os.path.exists(smartix_model_path):
            results['SmartIX'] = self.evaluate_smartix(smartix_model_path, n_episodes)

        # DBA Bandits (online, no model needed) — Dropped from 7-method comparison
        # results['DBA_Bandits'] = self.evaluate_dba_bandits(n_episodes)

        # Anytime (greedy, no model needed)
        results['Anytime'] = self.evaluate_anytime(n_episodes)

        if morl_model_path and os.path.exists(morl_model_path):
            results['MORL-Index'] = self.evaluate_morl(morl_model_path, n_episodes)

        # Compute Pareto dominance
        all_metrics = list(results.values())
        for name, metrics in results.items():
            metrics.pareto_dominance = self.compute_pareto_dominance(metrics, all_metrics)

        return results

    def print_results(self, results: Dict[str, ComprehensiveMetrics]):
        """Print comprehensive results table."""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE EVALUATION RESULTS (EXPLAIN-Only, Hypothetical Indexes)")
        print("=" * 100)

        # Table 1: Primary Metrics
        print("\n--- PRIMARY METRICS ---")
        print(f"{'Method':<15} {'Reward':<18} {'Cost Red %':<12} {'Write/Idx':<12} {'Idx Eff':<12} {'Write Avoid':<12}")
        print("-" * 81)
        for name, m in results.items():
            reward_str = f"{m.avg_reward:.2f}±{m.std_reward:.2f}"
            # Normalized write overhead per index (lower is better)
            write_per_idx = m.write_overhead_score / max(m.n_indexes, 1)
            print(f"{name:<15} {reward_str:<18} {m.cost_reduction_pct:<12.2f} {write_per_idx:<12.3f} {m.index_efficiency:<12.2f} {m.write_avoidance_ratio:<12.2f}")

        # Table 2: Secondary Metrics
        print("\n--- SECONDARY METRICS ---")
        print(f"{'Method':<15} {'Rec Time(ms)':<14} {'Robustness':<12} {'Write Avoid':<12} {'Pareto Dom':<12}")
        print("-" * 65)
        for name, m in results.items():
            cv_str = f"{m.robustness_cv:.4f}" if m.robustness_cv != float('inf') else "N/A"
            print(f"{name:<15} {m.recommendation_time_ms:<14.2f} {cv_str:<12} {m.write_avoidance_ratio:<12.2f} {m.pareto_dominance:<12}")

        # Table 3: HTAP Metrics
        print("\n--- HTAP METRICS ---")
        print(f"{'Method':<15} {'OLAP Cost Red %':<18} {'OLTP Cost Red %':<18} {'# Indexes':<12} {'Size (MB)':<12}")
        print("-" * 75)
        for name, m in results.items():
            print(f"{name:<15} {m.olap_cost_reduction:<18.2f} {m.oltp_cost_reduction:<18.2f} {m.n_indexes:<12.1f} {m.index_size_mb:<12.2f}")

        # Winner summary with normalized metrics
        print("\n" + "=" * 100)
        print("METRIC WINNERS (Normalized Comparison)")
        print("=" * 100)

        # Compute derived metrics for fair comparison
        derived_metrics = {}
        for name, m in results.items():
            if name == 'No-Index':
                continue
            derived_metrics[name] = {
                'avg_reward': m.avg_reward,
                'cost_reduction_pct': m.cost_reduction_pct,
                'write_per_index': m.write_overhead_score / max(m.n_indexes, 1),  # Normalized
                'index_efficiency': m.index_efficiency,
                'write_avoidance_ratio': m.write_avoidance_ratio,
                'robustness': 1 / (m.robustness_cv + 0.01) if m.robustness_cv != float('inf') else 0,  # Inverted CV
                'olap_cost_reduction': m.olap_cost_reduction,
                'oltp_cost_reduction': m.oltp_cost_reduction,
            }

        metrics_to_compare = [
            ('avg_reward', 'Multi-Obj Reward', True),
            ('cost_reduction_pct', 'Cost Reduction %', True),
            ('write_per_index', 'Write Overhead/Idx', False),
            ('index_efficiency', 'Index Efficiency', True),
            ('write_avoidance_ratio', 'Write Avoidance Ratio', True),
            ('olap_cost_reduction', 'OLAP Cost Reduction', True),
            ('oltp_cost_reduction', 'OLTP Cost Reduction', True),
        ]

        morl_wins = 0
        for attr, display_name, higher_better in metrics_to_compare:
            values = {k: v[attr] for k, v in derived_metrics.items()}
            if higher_better:
                winner = max(values, key=values.get)
            else:
                winner = min(values, key=values.get)

            marker = "**" if winner == 'MORL-Index' else ""
            value = values[winner]
            print(f"  {display_name:<25}: {marker}{winner}{marker} ({value:.2f})")
            if winner == 'MORL-Index':
                morl_wins += 1

        print(f"\nMORL-Index wins on {morl_wins}/{len(metrics_to_compare)} metrics")

        # Head-to-head vs DRLinda (the main baseline)
        if 'MORL-Index' in results and 'DRLinda' in results:
            print("\n" + "-" * 60)
            print("HEAD-TO-HEAD: MORL-Index vs DRLinda")
            print("-" * 60)
            morl = results['MORL-Index']
            drl = results['DRLinda']

            comparisons = [
                ('Multi-Obj Reward', morl.avg_reward, drl.avg_reward, True),
                ('Cost Reduction %', morl.cost_reduction_pct, drl.cost_reduction_pct, True),
                ('Write/Index', morl.write_overhead_score/max(morl.n_indexes,1),
                               drl.write_overhead_score/max(drl.n_indexes,1), False),
                ('Write Avoidance', morl.write_avoidance_ratio, drl.write_avoidance_ratio, True),
                ('OLAP Cost Red %', morl.olap_cost_reduction, drl.olap_cost_reduction, True),
                ('OLTP Cost Red %', morl.oltp_cost_reduction, drl.oltp_cost_reduction, True),
            ]

            morl_better = 0
            for name, morl_val, drl_val, higher_better in comparisons:
                if higher_better:
                    better = 'MORL' if morl_val > drl_val else 'DRL'
                else:
                    better = 'MORL' if morl_val < drl_val else 'DRL'
                if better == 'MORL':
                    morl_better += 1
                print(f"  {name:<20}: MORL={morl_val:>8.2f}, DRL={drl_val:>8.2f} → {better} wins")

            print(f"\nMORL-Index beats DRLinda on {morl_better}/{len(comparisons)} metrics")

    def generate_latex(self, results: Dict[str, ComprehensiveMetrics]) -> str:
        """Generate LaTeX table for paper."""
        latex = r"""
\begin{table*}[t]
\centering
\caption{Comprehensive Evaluation on CH-Benchmark HTAP Workload}
\label{tab:comprehensive}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Method} & \textbf{Reward} & \textbf{Cost Red.\%} & \textbf{Write OH$\downarrow$} & \textbf{Mem Eff$\uparrow$} & \textbf{Idx Eff$\uparrow$} & \textbf{Write Avoid$\uparrow$} & \textbf{Pareto$\uparrow$} & \textbf{\# Idx} \\
\midrule
"""
        for name, m in results.items():
            if name == 'MORL-Index':
                name_tex = r'\textbf{MORL-Index (Ours)}'
            else:
                name_tex = name

            reward_str = f"{m.avg_reward:.1f}$\\pm${m.std_reward:.1f}"
            latex += f"{name_tex} & {reward_str} & {m.cost_reduction_pct:.1f} & {m.write_overhead_score:.2f} & {m.memory_efficiency:.1f} & {m.index_efficiency:.2f} & {m.write_avoidance_ratio:.2f} & {m.pareto_dominance} & {m.n_indexes:.0f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        return latex

    def save_results(self, results: Dict[str, ComprehensiveMetrics], output_path: str):
        """Save results to JSON."""
        output_dict = {name: m.to_dict() for name, m in results.items()}
        output_dict['metadata'] = {
            'workload': self.workload_path,
            'baseline_cost': self.baseline_cost,
            'olap_baseline': self.olap_baseline,
            'oltp_baseline': self.oltp_baseline,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive multi-metric evaluation')
    parser.add_argument('--workload', type=str,
                        default='data/workloads/chbench_htap_test.sql',
                        help='Workload file')
    parser.add_argument('--morl-model', type=str,
                        default='results/full_experiments/morl_seed42_20260205_195859/model_best.pt')
    parser.add_argument('--drlinda-model', type=str,
                        default='results/full_experiments/drlinda_seed42_20260205_195859/model_best.pt')
    parser.add_argument('--swirl-model', type=str, default=None,
                        help='Path to trained SWIRL model')
    parser.add_argument('--smartix-model', type=str, default=None,
                        help='Path to trained SmartIX model')
    parser.add_argument('--n-episodes', type=int, default=20)
    parser.add_argument('--output', type=str, default='results/comprehensive_evaluation.json')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX table')

    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(
        workload_path=args.workload,
        memory_budget_mb=100.0
    )

    results = evaluator.run_full_evaluation(
        morl_model_path=args.morl_model,
        drlinda_model_path=args.drlinda_model,
        swirl_model_path=args.swirl_model,
        smartix_model_path=args.smartix_model,
        n_episodes=args.n_episodes
    )

    evaluator.print_results(results)
    evaluator.save_results(results, args.output)

    if args.latex:
        latex = evaluator.generate_latex(results)
        print("\n--- LaTeX Table ---")
        print(latex)
        latex_path = args.output.replace('.json', '.tex')
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX saved to: {latex_path}")


if __name__ == '__main__':
    main()
