"""
SWIRL baseline: PPO-based single-objective index selection.

Re-implementation of SWIRL (Kossmann et al., VLDB 2022) using the same
environment and evaluation protocol as P-NDQN. Optimizes latency only with PPO.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from environment_morl import MORLIndexEnvironment


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with action masking."""

    def __init__(self, n_features: int, n_actions: int, hidden_size: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x, action_mask=None):
        features = self.shared(x)
        logits = self.actor(features)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, x, action_mask=None, action=None):
        logits, value = self.forward(x, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


class SWIRLAgent:
    """
    SWIRL-like agent: PPO for single-objective index selection.

    Single-objective: Only optimizes for query latency reduction.
    Uses PPO with action masking, GAE, and clipped objective.
    """

    def __init__(
        self,
        env: MORLIndexEnvironment,
        output_path: str = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_episodes: int = 10000,
        n_epochs: int = 4,
        batch_size: int = 64,
        hidden_size: int = 256,
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_features = env.n_features
        self.n_actions = env.n_actions
        self.network = ActorCritic(self.n_features, self.n_actions, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"output/swirl_seed{seed}_{timestamp}/"
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.episode_rewards = []
        self.episode_index_counts = []
        self.episode = 0

    def compute_single_objective_reward(self, cost_before: float, cost_after: float) -> float:
        """Single-objective reward: latency improvement only."""
        if cost_before > 0:
            improvement = (cost_before - cost_after) / cost_before
        else:
            improvement = 0.0
        return improvement * 10.0

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """Select action using PPO policy."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if training:
                action, _, _, _ = self.network.get_action_and_value(state_t, mask_t)
            else:
                logits, _ = self.network.forward(state_t, mask_t)
                action = logits.argmax(dim=-1)

        return action.cpu().numpy()[0]

    def collect_episode(self):
        """Collect one episode with single-objective reward."""
        states, actions, rewards, values, log_probs, dones, action_masks = [], [], [], [], [], [], []

        state, info = self.env.reset()
        action_mask = info['action_mask']
        episode_reward = 0

        while True:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(state_t, mask_t)

            action_np = action.cpu().numpy()[0]

            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.cpu().numpy()[0])
            values.append(value.cpu().numpy()[0])
            action_masks.append(action_mask)

            next_state, _, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # SINGLE-OBJECTIVE reward (latency only)
            cost_before = info.get('cost_before', 1000)
            cost_after = info.get('cost_after', 1000)
            reward = self.compute_single_objective_reward(cost_before, cost_after)

            rewards.append(reward)
            dones.append(done)
            episode_reward += reward

            if done:
                break

            state = next_state
            action_mask = info['action_mask']

        final_value = 0.0
        advantages, returns = self._compute_gae(rewards, values, dones, final_value)

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'returns': returns,
            'advantages': advantages,
            'action_masks': np.array(action_masks),
            'episode_reward': episode_reward,
            'episode_length': len(rewards),
            'final_index_count': info.get('index_count', 0),
        }

    def _compute_gae(self, rewards, values, dones, final_value):
        n = len(rewards)
        advantages = np.zeros(n)
        gae = 0
        for t in reversed(range(n)):
            next_value = final_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, batch):
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(batch['action_masks'], dtype=torch.float32, device=self.device)

        n_samples = len(states)
        indices = np.arange(n_samples)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                bi = indices[start:end]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    states[bi], action_masks[bi], actions[bi]
                )

                ratio = torch.exp(new_log_probs - old_log_probs[bi])
                surr1 = ratio * advantages[bi]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[bi]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[bi])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("SWIRL BASELINE TRAINING (Single-Objective PPO)")
        print(f"{'='*60}")

        log_file = open(self.output_path + 'log.txt', 'w')
        log_file.write("episode\treward\tsteps\tindexes\n")

        best_reward = float('-inf')
        reward_window = deque(maxlen=50)

        for ep in range(self.n_episodes):
            self.episode = ep
            batch = self.collect_episode()
            self.update(batch)

            reward = batch['episode_reward']
            self.episode_rewards.append(reward)
            self.episode_index_counts.append(batch['final_index_count'])
            reward_window.append(reward)
            avg_reward = np.mean(reward_window)

            log_file.write(f"{ep}\t{reward:.2f}\t{batch['episode_length']}\t{batch['final_index_count']}\n")
            log_file.flush()

            if ep % 100 == 0:
                print(f"Episode {ep:5d} | Reward: {reward:7.2f} | Avg(50): {avg_reward:7.2f} | "
                      f"Indexes: {batch['final_index_count']:2d}")

            if avg_reward > best_reward and ep >= 50:
                best_reward = avg_reward
                self.save_model('model_best.pt')

        self.save_model('model_final.pt')
        log_file.close()

        stats = {
            'method': 'SWIRL',
            'n_episodes': self.n_episodes,
            'best_avg_reward': best_reward,
            'final_avg_reward': float(np.mean(self.episode_rewards[-50:])),
            'avg_index_count': float(np.mean(self.episode_index_counts[-50:])),
            'seed': self.seed,
        }
        with open(self.output_path + 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SWIRL TRAINING COMPLETE | Best: {best_reward:.2f}")
        print(f"{'='*60}")
        return stats

    def test(self, n_episodes: int = 20) -> Dict:
        self.network.eval()
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
            'method': 'SWIRL',
            'avg_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'avg_indexes': float(np.mean(all_indexes)),
            'n_episodes': n_episodes,
            'seed': self.seed,
        }

    def save_model(self, filename: str):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
        }, self.output_path + filename)

    def load_model(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(ckpt['network'])


def train_swirl(
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
    agent = SWIRLAgent(env=env, n_episodes=n_episodes, seed=seed)
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
        results = train_swirl(
            workload_path=args.workload,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(f"\n=== SWIRL Results ===")
        print(f"Avg Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Avg Indexes: {results['avg_indexes']:.1f}")
    else:
        env = MORLIndexEnvironment(
            workload_path=args.workload, hypo=True,
            max_steps_per_episode=50, max_indexes=15, memory_budget_mb=512,
        )
        agent = SWIRLAgent(env=env, seed=args.seed)
        agent.load_model(args.model)
        results = agent.test(n_episodes=20)
        env.close()
        print(f"Test Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
