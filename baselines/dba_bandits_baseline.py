"""
DBA Bandits baseline: C3UCB contextual combinatorial bandit.

Re-implementation of DBA Bandits (Perera et al., VLDB 2021). Each indexable
column is a bandit arm; selects a super arm using UCB scores with a linear
payoff model. Online learning, no separate training phase.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from environment_morl import MORLIndexEnvironment


class C3UCBBandit:
    """
    C3UCB Contextual Combinatorial Upper Confidence Bound bandit.

    Maintains a linear model for predicting index benefit from context.
    Uses UCB exploration bonus for selecting index arms.
    """

    def __init__(
        self,
        n_arms: int,
        context_size: int,
        alpha: float = 1.0,
        alpha_decay: float = 1.05,
        lambda_reg: float = 1.0,
    ):
        self.n_arms = n_arms
        self.context_size = context_size
        self.alpha = alpha
        self.alpha_init = alpha
        self.alpha_decay = alpha_decay

        # Linear model parameters
        self.V = lambda_reg * np.eye(context_size)
        self.b = np.zeros(context_size)
        self.V_inv = np.linalg.inv(self.V)
        self.theta = self.V_inv @ self.b

        # Per-arm statistics
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)

    def compute_ucb(self, context_vectors: np.ndarray) -> np.ndarray:
        """Compute UCB scores for all arms given their context vectors."""
        n = context_vectors.shape[0]
        ucb_scores = np.zeros(n)

        for i in range(n):
            ctx = context_vectors[i]
            avg_reward = self.theta @ ctx
            confidence = self.alpha * np.sqrt(ctx @ self.V_inv @ ctx)
            ucb_scores[i] = avg_reward + confidence

        return ucb_scores

    def select_super_arm(self, ucb_scores: np.ndarray, action_mask: np.ndarray,
                         max_indexes: int = 15) -> List[int]:
        """Greedy super arm selection respecting constraints."""
        selected = []
        available = np.where(action_mask == 1)[0]

        # Sort by UCB score descending
        sorted_arms = sorted(available, key=lambda a: ucb_scores[a], reverse=True)

        for arm in sorted_arms:
            if len(selected) >= max_indexes:
                break
            if ucb_scores[arm] > 0:
                selected.append(arm)

        return selected

    def update(self, context: np.ndarray, reward: float):
        """Update linear model with observed reward."""
        self.V += np.outer(context, context)
        self.b += reward * context

        # Recompute inverse and weights
        self.V_inv = np.linalg.inv(self.V)
        self.theta = self.V_inv @ self.b

        # Decay exploration
        self.alpha /= self.alpha_decay

    def reset_exploration(self):
        """Reset alpha for new episode."""
        self.alpha = self.alpha_init


class DBABanditsAgent:
    """
    DBA Bandits agent: C3UCB for index selection.

    Single-objective: Optimizes query cost reduction only.
    Uses contextual bandit with linear payoff model.
    No separate training phase - learns online.
    """

    def __init__(
        self,
        env: MORLIndexEnvironment,
        output_path: str = None,
        alpha: float = 1.0,
        alpha_decay: float = 1.05,
        lambda_reg: float = 1.0,
        n_episodes: int = 10000,
        seed: int = 42,
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.seed = seed

        np.random.seed(seed)

        self.n_actions = env.n_actions
        # Context: [query_frequency, column_index_indicator, step_ratio]
        self.context_size = 3

        self.bandit = C3UCBBandit(
            n_arms=self.n_actions,
            context_size=self.context_size,
            alpha=alpha,
            alpha_decay=alpha_decay,
            lambda_reg=lambda_reg,
        )

        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"output/dba_bandits_{timestamp}/"
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.episode_rewards = []
        self.episode_index_counts = []
        self.episode = 0

    def _build_context(self, state: np.ndarray, action: int) -> np.ndarray:
        """Build context vector for an arm (action)."""
        n_cols = self.n_actions // 2  # Half are CREATE, half are DROP
        col_idx = action % n_cols

        # Feature 1: Query frequency for this column (from state)
        if col_idx < len(state) // 2:
            query_freq = state[n_cols + col_idx] if (n_cols + col_idx) < len(state) else 0.0
        else:
            query_freq = 0.0

        # Feature 2: Whether this column is currently indexed
        is_indexed = state[col_idx] if col_idx < len(state) else 0.0

        # Feature 3: Normalized action index
        action_norm = action / max(self.n_actions - 1, 1)

        return np.array([query_freq, is_indexed, action_norm])

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """Select action using UCB scores."""
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return 0

        if not training:
            # Greedy: use theta directly without exploration
            old_alpha = self.bandit.alpha
            self.bandit.alpha = 0.0

        # Build contexts for all valid actions
        contexts = np.array([self._build_context(state, a) for a in valid_actions])
        ucb_scores = self.bandit.compute_ucb(contexts)

        if not training:
            self.bandit.alpha = old_alpha

        best_idx = np.argmax(ucb_scores)
        return valid_actions[best_idx]

    def compute_single_objective_reward(self, cost_before: float, cost_after: float) -> float:
        """Single-objective reward: cost improvement."""
        if cost_before > 0:
            return (cost_before - cost_after) / cost_before * 10.0
        return 0.0

    def train_episode(self) -> Dict:
        """Run one episode with online learning."""
        state, info = self.env.reset()
        action_mask = info['action_mask']
        episode_reward = 0
        episode_steps = 0

        # Reset exploration for new episode
        self.bandit.reset_exploration()

        while True:
            action = self.select_action(state, action_mask, training=True)

            next_state, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_mask = info['action_mask']

            cost_before = info.get('cost_before', 1000)
            cost_after = info.get('cost_after', 1000)
            reward = self.compute_single_objective_reward(cost_before, cost_after)

            # Update bandit model
            context = self._build_context(state, action)
            self.bandit.update(context, reward)

            episode_reward += reward
            episode_steps += 1

            if done:
                break

            state = next_state
            action_mask = next_mask

        metrics = self.env.get_final_metrics()

        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'index_count': metrics['index_count'],
        }

    def train(self):
        """Main training loop (online learning)."""
        print(f"\n{'='*60}")
        print("DBA Bandits BASELINE TRAINING (C3UCB Online)")
        print(f"{'='*60}")

        log_file = open(self.output_path + 'log.txt', 'w')
        log_file.write("episode\treward\tsteps\tindexes\n")

        best_reward = float('-inf')
        reward_window = deque(maxlen=50)

        for ep in range(self.n_episodes):
            self.episode = ep
            result = self.train_episode()

            self.episode_rewards.append(result['reward'])
            self.episode_index_counts.append(result['index_count'])
            reward_window.append(result['reward'])
            avg_reward = np.mean(reward_window)

            log_file.write(f"{ep}\t{result['reward']:.2f}\t{result['steps']}\t{result['index_count']}\n")
            log_file.flush()

            if ep % 100 == 0:
                print(f"Episode {ep:5d} | Reward: {result['reward']:7.2f} | "
                      f"Avg(50): {avg_reward:7.2f} | Indexes: {result['index_count']:2d}")

            if avg_reward > best_reward and ep >= 50:
                best_reward = avg_reward
                self.save_model('model_best.pt')

        self.save_model('model_final.pt')
        log_file.close()

        stats = {
            'method': 'DBA_Bandits',
            'n_episodes': self.n_episodes,
            'best_avg_reward': best_reward,
            'final_avg_reward': float(np.mean(self.episode_rewards[-50:])),
            'avg_index_count': float(np.mean(self.episode_index_counts[-50:])),
            'seed': self.seed,
        }
        with open(self.output_path + 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print(f"DBA Bandits TRAINING COMPLETE | Best: {best_reward:.2f}")
        print(f"{'='*60}")
        return stats

    def test(self, n_episodes: int = 20) -> Dict:
        """Test the bandit agent."""
        all_rewards, all_indexes = [], []

        for ep in range(n_episodes):
            state, info = self.env.reset()
            action_mask = info['action_mask']
            episode_reward = 0

            self.bandit.reset_exploration()

            while True:
                action = self.select_action(state, action_mask, training=False)
                next_state, _, terminated, truncated, info = self.env.step(action)

                cost_before = info.get('cost_before', 1000)
                cost_after = info.get('cost_after', 1000)
                episode_reward += self.compute_single_objective_reward(cost_before, cost_after)

                if terminated or truncated:
                    break
                state = next_state
                action_mask = info['action_mask']

            metrics = self.env.get_final_metrics()
            all_rewards.append(episode_reward)
            all_indexes.append(metrics['index_count'])

        return {
            'method': 'DBA_Bandits',
            'avg_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'avg_indexes': float(np.mean(all_indexes)),
            'n_episodes': n_episodes,
            'seed': self.seed,
        }

    def save_model(self, filename: str):
        """Save bandit model parameters."""
        np.savez(
            self.output_path + filename.replace('.pt', '.npz'),
            V=self.bandit.V,
            b=self.bandit.b,
            arm_counts=self.bandit.arm_counts,
            arm_rewards=self.bandit.arm_rewards,
            alpha=self.bandit.alpha,
        )

    def load_model(self, filepath: str):
        """Load bandit model parameters."""
        if filepath.endswith('.pt'):
            filepath = filepath.replace('.pt', '.npz')
        data = np.load(filepath)
        self.bandit.V = data['V']
        self.bandit.b = data['b']
        self.bandit.arm_counts = data['arm_counts']
        self.bandit.arm_rewards = data['arm_rewards']
        self.bandit.alpha = float(data['alpha'])
        self.bandit.V_inv = np.linalg.inv(self.bandit.V)
        self.bandit.theta = self.bandit.V_inv @ self.bandit.b


def train_dba_bandits(
    workload_path: str = 'data/workload/chbench_htap_balanced.sql',
    n_episodes: int = 10000,
    seed: int = 42,
) -> Dict:
    env = MORLIndexEnvironment(
        workload_path=workload_path,
        hypo=True,
        max_steps_per_episode=50,
        max_indexes=15,
        memory_budget_mb=512,
    )
    agent = DBABanditsAgent(env=env, n_episodes=n_episodes, seed=seed)
    stats = agent.train()
    test_results = agent.test(n_episodes=20)
    env.close()
    return {**stats, **test_results}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workload', type=str, default='data/workload/chbench_htap_balanced.sql')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        results = train_dba_bandits(
            workload_path=args.workload,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(f"\n=== DBA Bandits Results ===")
        print(f"Avg Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Avg Indexes: {results['avg_indexes']:.1f}")
    else:
        env = MORLIndexEnvironment(
            workload_path=args.workload, hypo=True,
            max_steps_per_episode=50, max_indexes=15, memory_budget_mb=512,
        )
        agent = DBABanditsAgent(env=env, seed=args.seed)
        agent.load_model(args.model)
        results = agent.test(n_episodes=20)
        env.close()
        print(f"Test Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
