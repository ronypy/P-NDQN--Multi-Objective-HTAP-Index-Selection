#!/usr/bin/env python3
"""
Evaluate Ablation Models for P-NDQN Paper

Loads each ablation model and evaluates on the test workload.
Computes MO Reward for each configuration and outputs the ablation table.
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment_morl import MORLIndexEnvironment
from train_morl_index import MORLIndexAgent


# Ablation configurations — models trained at 100 MB budget (seed 42).
# These are the models that produced the paper's Table 3 results.
# The "memory penalty" variant is excluded from the paper table (its effect
# overlaps functionally with action masking) but is kept here for completeness.
ABLATION_CONFIGS = [
    {
        'name': 'P-NDQN (full)',
        'model_dir': 'results/models/morl_seed42/',
        'agent_kwargs': {'use_per': True, 'use_noisy': True, 'use_dueling': True},
        'env_kwargs': {'use_action_mask': True},
    },
    {
        'name': '- Write penalty',
        'model_dir': 'results/ablation/no_write_seed42/',
        'agent_kwargs': {'use_per': True, 'use_noisy': True, 'use_dueling': True},
        'env_kwargs': {'use_action_mask': True},
    },
    {
        'name': '- Phase adaptation',
        'model_dir': 'results/ablation/no_phase_seed42/',
        'agent_kwargs': {'use_per': True, 'use_noisy': True, 'use_dueling': True},
        'env_kwargs': {'use_action_mask': True},
    },
    {
        'name': '- PER (uniform replay)',
        'model_dir': 'results/ablation/no_per_seed42/',
        'agent_kwargs': {'use_per': False, 'use_noisy': True, 'use_dueling': True},
        'env_kwargs': {'use_action_mask': True},
    },
    {
        'name': '- NoisyNet (eps-greedy)',
        'model_dir': 'results/ablation/no_noisy_seed42/',
        'agent_kwargs': {'use_per': True, 'use_noisy': False, 'use_dueling': True},
        'env_kwargs': {'use_action_mask': True},
    },
    {
        'name': '- Dueling (standard head)',
        'model_dir': 'results/ablation/no_dueling_seed42/',
        'agent_kwargs': {'use_per': True, 'use_noisy': True, 'use_dueling': False},
        'env_kwargs': {'use_action_mask': True},
    },
    {
        'name': '- Action masking',
        'model_dir': 'results/ablation/no_mask_seed42/',
        'agent_kwargs': {'use_per': True, 'use_noisy': True, 'use_dueling': True},
        'env_kwargs': {'use_action_mask': False},
    },
]

TEST_WORKLOAD = 'data/workloads/chbench_htap_test.sql'
N_EPISODES = 15
SEED = 42


def evaluate_ablation_model(config: dict) -> dict:
    """Evaluate a single ablation model."""
    name = config['name']
    model_dir = config['model_dir']
    model_path = os.path.join(model_dir, 'model_best.pt')

    if not os.path.exists(model_path):
        print(f"  WARNING: {model_path} not found, skipping")
        return None

    print(f"\n--- Evaluating: {name} ---")
    print(f"  Model: {model_path}")

    # Create environment — same settings as comprehensive_evaluation.py
    # All models evaluated with the SAME full reward function (default weights)
    # so MO Reward is comparable across rows. Only architecture differs.
    env_kwargs = config.get('env_kwargs', {})

    env = MORLIndexEnvironment(
        workload_path=TEST_WORKLOAD,
        hypo=True,
        memory_budget_mb=100.0,  # Must match comprehensive_evaluation.py
        **env_kwargs,
    )

    # Create agent with appropriate architecture flags
    agent_kwargs = config.get('agent_kwargs', {})
    agent = MORLIndexAgent(
        env=env,
        output_path=None,
        **agent_kwargs,
    )
    agent.load_model(model_path)

    # Evaluate
    all_rewards = []
    all_index_counts = []
    all_cost_reductions = []

    for ep in range(N_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(50):
            action = agent.select_action(state, env._get_action_mask(), training=False)
            state, reward, done, _, info = env.step(action)
            episode_reward += reward

            if done:
                break

        metrics = env.get_final_metrics()
        all_rewards.append(episode_reward)
        all_index_counts.append(metrics['index_count'])

    env.close()

    avg_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    avg_indexes = float(np.mean(all_index_counts))

    print(f"  MO Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Avg indexes: {avg_indexes:.1f}")

    return {
        'name': name,
        'model_dir': model_dir,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_indexes': avg_indexes,
    }


def main():
    print("=" * 60)
    print("P-NDQN Ablation Study Evaluation")
    print(f"Test workload: {TEST_WORKLOAD}")
    print(f"Episodes: {N_EPISODES}, Seed: {SEED}")
    print("=" * 60)

    results = []
    for config in ABLATION_CONFIGS:
        result = evaluate_ablation_model(config)
        if result is not None:
            results.append(result)

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<30} {'MO Reward':>12} {'Δ vs Full':>12} {'# Idx':>8}")
    print("-" * 70)

    full_reward = None
    for r in results:
        if r['name'] == 'P-NDQN (full)':
            full_reward = r['avg_reward']
            break

    for r in results:
        delta = r['avg_reward'] - full_reward if full_reward is not None else 0
        delta_str = f"{delta:+.2f}" if r['name'] != 'P-NDQN (full)' else '---'
        print(f"{r['name']:<30} {r['avg_reward']:>+12.2f} {delta_str:>12} {r['avg_indexes']:>8.1f}")

    # Add DRLinda-equivalent row (already known from comprehensive evaluation)
    print(f"{'Single-obj (DRLinda-equiv.)':<30} {-41.39:>+12.2f} {-41.39 - full_reward if full_reward else 0:>+12.2f} {'3.9':>8}")
    print("=" * 70)

    # Save results
    output_path = 'results/ablation_results.json'
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'configs': results,
            'full_reward': full_reward,
            'drlinda_reward': -41.39,
            'seed': SEED,
            'n_episodes': N_EPISODES,
            'workload': TEST_WORKLOAD,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate LaTeX table
    print("\n--- LaTeX Table (copy to paper) ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study: impact of removing each component from P-NDQN}")
    print(r"\label{tab:ablation}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{MO Reward} & \textbf{$\Delta$ vs.\ Full} \\")
    print(r"\midrule")

    for r in results:
        delta = r['avg_reward'] - full_reward if full_reward is not None else 0
        name_tex = r['name']
        if name_tex == 'P-NDQN (full)':
            delta_str = '---'
        else:
            delta_str = f"${delta:+.2f}$"
            name_tex = r'\quad ' + name_tex

        reward_str = f"${r['avg_reward']:+.2f}$" if r['avg_reward'] < 0 else f"$+{r['avg_reward']:.2f}$"
        if r['name'] == 'P-NDQN (full)':
            reward_str = f"$+{r['avg_reward']:.2f}$"

        print(f"{name_tex} & {reward_str} & {delta_str} \\\\")

    drl_delta = -41.39 - full_reward if full_reward else 0
    print(f"Single-obj (DRLinda-equiv.) & $-41.39$ & ${drl_delta:+.2f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == '__main__':
    main()
