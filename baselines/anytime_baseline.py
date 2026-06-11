"""
Anytime baseline: greedy index enumeration.

Re-implementation of the Anytime algorithm (Chaudhuri & Narasayya, 2020).
One-shot greedy approach: naive enumeration for small k, then greedy
marginal-benefit addition up to max_num indexes.
"""

import os
import json
import time
import itertools
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from environment_morl import MORLIndexEnvironment


class AnytimeAgent:
    """
    Anytime agent: Greedy enumeration for index selection.

    One-shot algorithm that greedily builds an index configuration.
    Evaluated in the MORL environment for fair metric comparison.
    """

    def __init__(
        self,
        env: MORLIndexEnvironment,
        output_path: str = None,
        max_indexes: int = 15,
        naive_num: int = 3,
        max_time: float = 60.0,
        n_episodes: int = 10000,
        seed: int = 42,
    ):
        self.env = env
        self.max_indexes = max_indexes
        self.naive_num = naive_num
        self.max_time = max_time
        self.n_episodes = n_episodes
        self.seed = seed

        np.random.seed(seed)

        self.n_actions = env.n_actions

        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"output/anytime_{timestamp}/"
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.episode_rewards = []
        self.episode_index_counts = []
        self.episode = 0

        # Pre-compute recommended indexes
        self.recommended_actions = None

    def _get_candidate_actions(self, state: np.ndarray) -> List[int]:
        """Get candidate CREATE actions based on workload frequency in state."""
        n_cols = self.n_actions // 2
        candidates = []

        for col_idx in range(n_cols):
            # Check if column appears in workload (has non-zero frequency)
            freq_idx = n_cols + col_idx
            if freq_idx < len(state):
                freq = state[freq_idx]
                if freq > 0:
                    candidates.append(col_idx)  # CREATE action for this column

        if not candidates:
            candidates = list(range(min(n_cols, 20)))

        return candidates

    def _evaluate_index_set(self, action_sequence: List[int]) -> float:
        """Evaluate a set of indexes by running them in the environment."""
        state, info = self.env.reset()
        total_cost_reduction = 0

        for action in action_sequence:
            action_mask = info['action_mask']
            if action_mask[action] == 0:
                continue

            next_state, _, terminated, truncated, info = self.env.step(action)

            cost_before = info.get('cost_before', 1000)
            cost_after = info.get('cost_after', 1000)
            if cost_before > 0:
                total_cost_reduction += (cost_before - cost_after)

            if terminated or truncated:
                break

            state = next_state

        return total_cost_reduction

    def recommend_indexes(self) -> List[int]:
        """Run Anytime algorithm to recommend index actions.

        Phase 1: Identify candidate columns from workload
        Phase 2: Naive enumeration for small sets
        Phase 3: Greedy addition for remaining
        """
        start_time = time.time()

        # Get initial state to find candidates
        state, info = self.env.reset()
        candidates = self._get_candidate_actions(state)

        if not candidates:
            return []

        # Phase 1: Naive enumeration for small k
        naive_k = min(self.naive_num, len(candidates))
        best_actions = []
        best_cost = float('inf')

        for k in range(1, naive_k + 1):
            if time.time() - start_time >= self.max_time:
                break

            for combo in itertools.combinations(candidates, k):
                if time.time() - start_time >= self.max_time:
                    break

                actions = list(combo)
                cost = -self._evaluate_index_set(actions)  # Negative because we want to minimize

                if cost < best_cost:
                    best_cost = cost
                    best_actions = actions

        # Phase 2: Greedy addition
        current_actions = set(best_actions)
        remaining = [c for c in candidates if c not in current_actions]

        while len(current_actions) < self.max_indexes and remaining:
            if time.time() - start_time >= self.max_time:
                break

            best_addition = None
            best_new_cost = best_cost

            for action in remaining:
                trial = list(current_actions) + [action]
                cost = -self._evaluate_index_set(trial)

                if cost < best_new_cost:
                    best_new_cost = cost
                    best_addition = action

            if best_addition is not None:
                current_actions.add(best_addition)
                remaining.remove(best_addition)
                best_cost = best_new_cost
            else:
                break

        self.recommended_actions = list(current_actions)
        return self.recommended_actions

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """Select action from pre-computed recommendations."""
        if self.recommended_actions is None:
            self.recommended_actions = self.recommend_indexes()

        # Find the next recommended action that hasn't been applied yet
        n_cols = self.n_actions // 2

        for action in self.recommended_actions:
            if action < n_cols and action < len(state):
                if state[action] == 0 and action_mask[action] == 1:
                    return action

        # If all recommended already applied, select NOOP or any valid action
        valid = np.where(action_mask == 1)[0]
        if len(valid) > 0:
            # Prefer a "no change" action - pick drop of non-indexed column (no-op effect)
            for a in valid:
                if a >= n_cols:
                    col = a - n_cols
                    if col < len(state) and state[col] == 0:
                        return a
            return valid[0]
        return 0

    def compute_single_objective_reward(self, cost_before: float, cost_after: float) -> float:
        """Single-objective reward: cost improvement."""
        if cost_before > 0:
            return (cost_before - cost_after) / cost_before * 10.0
        return 0.0

    def train_episode(self) -> Dict:
        """Run one episode applying recommended indexes."""
        state, info = self.env.reset()
        action_mask = info['action_mask']
        episode_reward = 0
        episode_steps = 0

        while True:
            action = self.select_action(state, action_mask, training=True)
            next_state, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            cost_before = info.get('cost_before', 1000)
            cost_after = info.get('cost_after', 1000)
            reward = self.compute_single_objective_reward(cost_before, cost_after)
            episode_reward += reward
            episode_steps += 1

            if done:
                break

            state = next_state
            action_mask = info['action_mask']

        metrics = self.env.get_final_metrics()

        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'index_count': metrics['index_count'],
        }

    def train(self):
        """Training for Anytime = run recommendation then evaluate."""
        print(f"\n{'='*60}")
        print("Anytime BASELINE (Greedy Enumeration)")
        print(f"{'='*60}")

        # First, compute recommendations
        print("Computing index recommendations...")
        self.recommend_indexes()
        print(f"Recommended {len(self.recommended_actions)} index actions: {self.recommended_actions}")

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

        log_file.close()

        stats = {
            'method': 'Anytime',
            'n_episodes': self.n_episodes,
            'best_avg_reward': best_reward,
            'final_avg_reward': float(np.mean(self.episode_rewards[-50:])),
            'avg_index_count': float(np.mean(self.episode_index_counts[-50:])),
            'recommended_actions': self.recommended_actions,
            'seed': self.seed,
        }
        with open(self.output_path + 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Anytime COMPLETE | Best: {best_reward:.2f}")
        print(f"{'='*60}")
        return stats

    def test(self, n_episodes: int = 20) -> Dict:
        """Test the Anytime agent."""
        if self.recommended_actions is None:
            self.recommend_indexes()

        all_rewards, all_indexes = [], []

        for ep in range(n_episodes):
            state, info = self.env.reset()
            action_mask = info['action_mask']
            episode_reward = 0

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
            'method': 'Anytime',
            'avg_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'avg_indexes': float(np.mean(all_indexes)),
            'n_episodes': n_episodes,
            'seed': self.seed,
        }

    def save_model(self, filename: str):
        """Save recommended actions."""
        data = {
            'recommended_actions': self.recommended_actions,
            'seed': self.seed,
        }
        with open(self.output_path + filename.replace('.pt', '.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def load_model(self, filepath: str):
        """Load recommended actions."""
        if filepath.endswith('.pt'):
            filepath = filepath.replace('.pt', '.json')
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.recommended_actions = data['recommended_actions']


def train_anytime(
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
    agent = AnytimeAgent(env=env, n_episodes=n_episodes, seed=seed)
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
        results = train_anytime(
            workload_path=args.workload,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(f"\n=== Anytime Results ===")
        print(f"Avg Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Avg Indexes: {results['avg_indexes']:.1f}")
    else:
        env = MORLIndexEnvironment(
            workload_path=args.workload, hypo=True,
            max_steps_per_episode=50, max_indexes=15, memory_budget_mb=512,
        )
        agent = AnytimeAgent(env=env, seed=args.seed)
        agent.load_model(args.model)
        results = agent.test(n_episodes=20)
        env.close()
        print(f"Test Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
