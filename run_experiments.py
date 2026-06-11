"""
MORL-Index Paper Experiments Runner
====================================

Runs all baselines and MORL-Index with multiple seeds for paper comparison.

Usage:
    python run_experiments.py --episodes 10000 --seeds 3
    python run_experiments.py --methods morl,drlinda --episodes 5000 --seeds 1
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment_morl import MORLIndexEnvironment
from train_morl_index import MORLIndexAgent
from baselines.random_baseline import RandomIndexSelector
from baselines.drlinda_baseline import DRLindaAgent


def run_no_index_baseline(env: MORLIndexEnvironment, n_episodes: int = 20, seed: int = 42) -> Dict:
    """Evaluate cost with no indexes active."""
    np.random.seed(seed)

    all_rewards = []
    all_costs = []

    for ep in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_costs = []

        for step in range(50):
            cost = env._compute_window_cost()
            episode_costs.append(cost)
            env._advance_workload()

        all_costs.append(np.mean(episode_costs))
        all_rewards.append(0)

    return {
        'method': 'No-Index',
        'avg_reward': 0.0,
        'std_reward': 0.0,
        'avg_indexes': 0.0,
        'std_indexes': 0.0,
        'avg_cost': np.mean(all_costs),
        'std_cost': np.std(all_costs),
        'n_episodes': n_episodes,
        'seed': seed,
    }


def run_heuristic_baseline(env: MORLIndexEnvironment, n_episodes: int = 20, seed: int = 42) -> Dict:
    """
    Heuristic Baseline: Use predefined optimal indexes for CH-benchmark.
    """
    np.random.seed(seed)

    # Predefined indexes based on CH-benchmark query analysis
    heuristic_indexes = [
        ('customer', 'c_id'),
        ('customer', 'c_w_id'),
        ('item', 'i_id'),
        ('item', 'i_price'),
        ('district', 'd_id'),
        ('district', 'd_w_id'),
        ('warehouse', 'w_id'),
        ('stock', 's_i_id'),
        ('stock', 's_w_id'),
        ('supplier', 'su_suppkey'),
    ]

    all_rewards = []
    all_costs = []
    all_indexes = []

    for ep in range(n_episodes):
        state, info = env.reset()
        episode_costs = []

        # Create heuristic indexes
        indexes_created = 0
        for table, column in heuristic_indexes:
            if column in env.columns:
                col_idx = env.columns.index(column)
                current_indexes = env.db.get_indexes()
                if current_indexes.get(column, 0) == 0:
                    try:
                        env.db.create_index(table, column)
                        indexes_created += 1
                    except Exception:
                        pass

        # Measure cost with heuristic indexes
        for step in range(50):
            cost = env._compute_window_cost()
            episode_costs.append(cost)
            env._advance_workload()

        all_costs.append(np.mean(episode_costs))
        all_indexes.append(indexes_created)
        all_rewards.append(0)  # No RL reward for heuristic

    return {
        'method': 'Heuristic',
        'avg_reward': 0.0,
        'std_reward': 0.0,
        'avg_indexes': np.mean(all_indexes),
        'std_indexes': np.std(all_indexes),
        'avg_cost': np.mean(all_costs),
        'std_cost': np.std(all_costs),
        'n_episodes': n_episodes,
        'seed': seed,
    }


def run_random_baseline(env: MORLIndexEnvironment, n_episodes: int = 20, seed: int = 42) -> Dict:
    """Run Random baseline."""
    selector = RandomIndexSelector(env, seed=seed)
    return selector.evaluate(n_episodes)


def run_drlinda_baseline(
    workload_path: str,
    n_episodes: int = 10000,
    seed: int = 42,
    output_dir: str = None,
) -> Dict:
    """Train and evaluate DRLinda baseline."""
    env = MORLIndexEnvironment(
        workload_path=workload_path,
        hypo=True,
        max_steps_per_episode=50,
        max_indexes=15,
        memory_budget_mb=512,
    )

    if output_dir is None:
        output_dir = f"output/drlinda_seed{seed}_{int(time.time())}/"

    agent = DRLindaAgent(
        env=env,
        output_path=output_dir,
        n_episodes=n_episodes,
        seed=seed,
    )

    train_stats = agent.train()
    test_results = agent.test(n_episodes=20)

    env.close()

    return {
        'method': 'DRLinda',
        'train_stats': train_stats,
        'test_results': test_results,
        'seed': seed,
        'output_path': output_dir,
    }


def run_morl_index(
    workload_path: str,
    n_episodes: int = 10000,
    seed: int = 42,
    output_dir: str = None,
) -> Dict:
    """Train and evaluate MORL-Index (our method)."""
    import random
    import torch

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MORLIndexEnvironment(
        workload_path=workload_path,
        hypo=True,
        max_steps_per_episode=50,
        max_indexes=15,
        memory_budget_mb=512,
    )

    if output_dir is None:
        output_dir = f"output/morl_seed{seed}_{int(time.time())}/"

    agent = MORLIndexAgent(
        env=env,
        output_path=output_dir,
        n_episodes=n_episodes,
        eps_end=0.05,
        eps_decay_episodes=1000,
        entropy_coef=0.02,
        sigma_decay=0.9995,
    )

    train_stats = agent.train()
    test_results = agent.test(n_episodes=20)

    env.close()

    return {
        'method': 'MORL-Index',
        'train_stats': train_stats,
        'test_results': test_results,
        'seed': seed,
        'output_path': output_dir,
    }


def aggregate_results(results_list: List[Dict]) -> Dict:
    """Aggregate results across multiple seeds."""
    if not results_list:
        return {}

    method = results_list[0].get('method', 'Unknown')

    # Collect metrics
    rewards = []
    indexes = []
    costs = []

    for r in results_list:
        if 'test_results' in r:
            rewards.append(r['test_results'].get('avg_reward', 0))
            indexes.append(r['test_results'].get('avg_indexes', 0))
        else:
            rewards.append(r.get('avg_reward', 0))
            indexes.append(r.get('avg_indexes', 0))
        costs.append(r.get('avg_cost', 0))

    return {
        'method': method,
        'n_seeds': len(results_list),
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'indexes_mean': np.mean(indexes),
        'indexes_std': np.std(indexes),
        'cost_mean': np.mean(costs) if any(costs) else 0,
        'cost_std': np.std(costs) if any(costs) else 0,
        'seeds': [r.get('seed') for r in results_list],
    }


def run_all_experiments(
    workload_path: str,
    n_episodes: int = 10000,
    seeds: List[int] = [42, 123, 456],
    methods: List[str] = None,
    output_base: str = "results/",
) -> Dict:
    """
    Run all experiments for paper comparison.

    Args:
        workload_path: Path to workload SQL file
        n_episodes: Number of training episodes for learnable methods
        seeds: List of random seeds
        methods: List of methods to run (default: all)
        output_base: Base directory for results
    """
    if methods is None:
        methods = ['no_index', 'heuristic', 'random', 'drlinda', 'morl']

    os.makedirs(output_base, exist_ok=True)

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("MORL-INDEX PAPER EXPERIMENTS")
    print("=" * 70)
    print(f"Workload: {workload_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print("=" * 70)

    # Create base environment for non-learnable baselines
    env = MORLIndexEnvironment(
        workload_path=workload_path,
        hypo=True,
        max_steps_per_episode=50,
        max_indexes=15,
        memory_budget_mb=512,
    )

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running: {method.upper()}")
        print(f"{'='*50}")

        method_results = []

        for seed in seeds:
            print(f"\n  Seed {seed}...")

            if method == 'no_index':
                result = run_no_index_baseline(env, n_episodes=20, seed=seed)

            elif method == 'heuristic':
                result = run_heuristic_baseline(env, n_episodes=20, seed=seed)

            elif method == 'random':
                result = run_random_baseline(env, n_episodes=20, seed=seed)

            elif method == 'drlinda':
                output_dir = f"{output_base}drlinda_seed{seed}_{timestamp}/"
                result = run_drlinda_baseline(
                    workload_path, n_episodes, seed, output_dir
                )

            elif method == 'morl':
                output_dir = f"{output_base}morl_seed{seed}_{timestamp}/"
                result = run_morl_index(
                    workload_path, n_episodes, seed, output_dir
                )

            else:
                print(f"  Unknown method: {method}")
                continue

            method_results.append(result)
            print(f"  Seed {seed} complete.")

        # Aggregate results for this method
        all_results[method] = {
            'individual': method_results,
            'aggregated': aggregate_results(method_results),
        }

    env.close()

    # Save all results
    results_file = f"{output_base}experiment_results_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'Method':<15} {'Reward':>12} {'Indexes':>12} {'Cost':>12}")
    print("-" * 55)

    for method, data in all_results.items():
        agg = data['aggregated']
        reward_str = f"{agg['reward_mean']:.2f} +/- {agg['reward_std']:.2f}"
        idx_str = f"{agg['indexes_mean']:.1f} +/- {agg['indexes_std']:.1f}"
        cost_str = f"{agg['cost_mean']:.0f}" if agg['cost_mean'] else "N/A"
        print(f"{method:<15} {reward_str:>12} {idx_str:>12} {cost_str:>12}")

    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='MORL-Index Paper Experiments')
    parser.add_argument('--workload', type=str,
                        default='data/workloads/chbench_htap_balanced.sql',
                        help='Workload file path')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Training episodes for learnable methods')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods (default: all)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    args = parser.parse_args()

    # Parse methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
    else:
        methods = None

    # Generate seed list
    seeds = [42 + i * 111 for i in range(args.seeds)]

    run_all_experiments(
        workload_path=args.workload,
        n_episodes=args.episodes,
        seeds=seeds,
        methods=methods,
        output_base=args.output,
    )


if __name__ == '__main__':
    main()
