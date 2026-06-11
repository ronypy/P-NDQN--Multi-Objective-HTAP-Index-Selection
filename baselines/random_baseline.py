"""Random index selection baseline."""

import numpy as np
import random
from typing import Dict, List, Tuple
import sys
sys.path.append('..')

from environment_morl import MORLIndexEnvironment


class RandomIndexSelector:
    """
    Random baseline that randomly toggles indexes.

    This represents the worst-case scenario for comparison.
    """

    def __init__(self, env: MORLIndexEnvironment, seed: int = 42):
        self.env = env
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def select_action(self, action_mask: np.ndarray) -> int:
        """Randomly select from valid actions."""
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return 0
        return np.random.choice(valid_actions)

    def evaluate(self, n_episodes: int = 20) -> Dict:
        """Evaluate random baseline over multiple episodes."""
        all_rewards = []
        all_indexes = []
        all_costs = []

        for ep in range(n_episodes):
            state, info = self.env.reset()
            action_mask = info['action_mask']
            episode_reward = 0
            episode_costs = []

            while True:
                action = self.select_action(action_mask)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                if 'cost_after' in info:
                    episode_costs.append(info['cost_after'])

                if terminated or truncated:
                    break

                state = next_state
                action_mask = info['action_mask']

            metrics = self.env.get_final_metrics()
            all_rewards.append(episode_reward)
            all_indexes.append(metrics['index_count'])
            if episode_costs:
                all_costs.append(np.mean(episode_costs))

        return {
            'method': 'Random',
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_indexes': np.mean(all_indexes),
            'std_indexes': np.std(all_indexes),
            'avg_cost': np.mean(all_costs) if all_costs else 0,
            'n_episodes': n_episodes,
            'seed': self.seed,
        }


def run_random_baseline(
    workload_path: str = 'data/workload/chbench_htap_balanced.sql',
    n_episodes: int = 20,
    seed: int = 42,
) -> Dict:
    """Run random baseline evaluation."""
    env = MORLIndexEnvironment(
        workload_path=workload_path,
        hypo=True,
        max_steps_per_episode=50,
        max_indexes=15,
        memory_budget_mb=512,
    )

    selector = RandomIndexSelector(env, seed=seed)
    results = selector.evaluate(n_episodes)

    env.close()
    return results


if __name__ == '__main__':
    print("Running Random Baseline...")
    results = run_random_baseline(n_episodes=20)
    print(f"\n=== Random Baseline Results ===")
    print(f"Avg Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Avg Indexes: {results['avg_indexes']:.1f} +/- {results['std_indexes']:.1f}")
    print(f"Avg Cost: {results['avg_cost']:.2f}")
