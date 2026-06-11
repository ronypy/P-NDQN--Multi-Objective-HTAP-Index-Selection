"""
SmartIX baseline: DQN-based single-objective index selection.

Re-implementation of SmartIX (Licks et al., Applied Intelligence 2020) using
the same environment and evaluation protocol as P-NDQN. Uses inverse-cost
reward (1/cost × 100000), 64-unit hidden layers, and γ=0.9.
"""

import os
import json
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from environment_morl import MORLIndexEnvironment


class SmartIXQNet(nn.Module):
    """SmartIX Q-Network: smaller architecture (64 hidden)."""

    def __init__(self, n_features: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_values = self.network(x)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e9)
        return q_values


class SmartIXReplayBuffer:
    """Replay buffer for SmartIX."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class SmartIXAgent:
    """
    SmartIX agent: DQN for single-objective index selection.

    Single-objective: Uses inverse cost reward (1/cost * 100000).
    Smaller network (64 hidden), lower gamma (0.9), larger batch (1024).
    """

    def __init__(
        self,
        env: MORLIndexEnvironment,
        output_path: str = None,
        gamma: float = 0.9,
        learning_rate: float = 1e-4,
        n_episodes: int = 10000,
        memory_size: int = 10000,
        batch_size: int = 1024,
        target_update_freq: int = 128,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_rate: float = 0.01,
        hidden_size: int = 64,
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_rate = eps_decay_rate
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_features = env.n_features
        self.n_actions = env.n_actions

        self.qnet = SmartIXQNet(self.n_features, self.n_actions, hidden_size).to(self.device)
        self.qnet_target = SmartIXQNet(self.n_features, self.n_actions, hidden_size).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        self.memory = SmartIXReplayBuffer(memory_size)

        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"output/smartix_seed{seed}_{timestamp}/"
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.epsilon = eps_start
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_index_counts = []

    def compute_single_objective_reward(self, cost_before: float, cost_after: float) -> float:
        """SmartIX reward: inverse cost formulation."""
        if cost_after > 0:
            return (1.0 / cost_after) * 100000.0
        return 0.0

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """Select action with epsilon-greedy."""
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return 0

        eps = self.epsilon if training else 0.0

        if training and random.random() < eps:
            return np.random.choice(valid_actions)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.qnet(state_t, mask_t)
                return q_values.argmax(dim=-1).item()

    def update(self) -> Optional[float]:
        """Update Q-network."""
        if len(self.memory) < self.batch_size:
            return None

        samples = self.memory.sample(self.batch_size)

        states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long, device=self.device)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32, device=self.device)

        current_q = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.qnet_target(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = (current_q - target_q).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_episode(self) -> Dict:
        """Train one episode."""
        state, info = self.env.reset()
        action_mask = info['action_mask']
        episode_reward = 0
        episode_steps = 0
        losses = []

        while True:
            action = self.select_action(state, action_mask, training=True)
            next_state, _, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_mask = info['action_mask']

            cost_before = info.get('cost_before', 1000)
            cost_after = info.get('cost_after', 1000)
            reward = self.compute_single_objective_reward(cost_before, cost_after)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            self.memory.add((state, action, reward, next_state, float(done)))

            loss = self.update()
            if loss is not None:
                losses.append(loss)

            if self.total_steps % self.target_update_freq == 0:
                self.qnet_target.load_state_dict(self.qnet.state_dict())

            if done:
                break

            state = next_state
            action_mask = next_mask

        # Epsilon decay (SmartIX style: multiplicative)
        if len(self.memory) > self.batch_size:
            self.epsilon -= self.epsilon * self.eps_decay_rate
            if self.epsilon < self.eps_end:
                self.epsilon = self.eps_end

        metrics = self.env.get_final_metrics()

        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'loss': np.mean(losses) if losses else 0,
            'index_count': metrics['index_count'],
            'epsilon': self.epsilon,
        }

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("SmartIX BASELINE TRAINING (Single-Objective DQN)")
        print(f"{'='*60}")

        log_file = open(self.output_path + 'log.txt', 'w')
        log_file.write("episode\treward\tsteps\tloss\tindexes\tepsilon\n")

        best_reward = float('-inf')
        reward_window = deque(maxlen=50)

        for ep in range(self.n_episodes):
            self.episode = ep
            result = self.train_episode()

            self.episode_rewards.append(result['reward'])
            self.episode_index_counts.append(result['index_count'])
            reward_window.append(result['reward'])
            avg_reward = np.mean(reward_window)

            log_file.write(f"{ep}\t{result['reward']:.2f}\t{result['steps']}\t"
                           f"{result['loss']:.4f}\t{result['index_count']}\t{result['epsilon']:.3f}\n")
            log_file.flush()

            if ep % 100 == 0:
                print(f"Episode {ep:5d} | Reward: {result['reward']:7.2f} | "
                      f"Avg(50): {avg_reward:7.2f} | Indexes: {result['index_count']:2d} | "
                      f"Eps: {result['epsilon']:.3f}")

            if avg_reward > best_reward and ep >= 50:
                best_reward = avg_reward
                self.save_model('model_best.pt')

        self.save_model('model_final.pt')
        log_file.close()

        stats = {
            'method': 'SmartIX',
            'n_episodes': self.n_episodes,
            'total_steps': self.total_steps,
            'best_avg_reward': best_reward,
            'final_avg_reward': float(np.mean(self.episode_rewards[-50:])),
            'avg_index_count': float(np.mean(self.episode_index_counts[-50:])),
            'seed': self.seed,
        }
        with open(self.output_path + 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SmartIX TRAINING COMPLETE | Best: {best_reward:.2f}")
        print(f"{'='*60}")
        return stats

    def test(self, n_episodes: int = 20) -> Dict:
        """Test trained model."""
        self.qnet.eval()
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
            'method': 'SmartIX',
            'avg_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'avg_indexes': float(np.mean(all_indexes)),
            'n_episodes': n_episodes,
            'seed': self.seed,
        }

    def save_model(self, filename: str):
        torch.save({
            'qnet': self.qnet.state_dict(),
            'qnet_target': self.qnet_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
        }, self.output_path + filename)

    def load_model(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.qnet.load_state_dict(ckpt['qnet'])
        self.qnet_target.load_state_dict(ckpt['qnet_target'])


def train_smartix(
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
    agent = SmartIXAgent(env=env, n_episodes=n_episodes, seed=seed)
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
        results = train_smartix(
            workload_path=args.workload,
            n_episodes=args.episodes,
            seed=args.seed,
        )
        print(f"\n=== SmartIX Results ===")
        print(f"Avg Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"Avg Indexes: {results['avg_indexes']:.1f}")
    else:
        env = MORLIndexEnvironment(
            workload_path=args.workload, hypo=True,
            max_steps_per_episode=50, max_indexes=15, memory_budget_mb=512,
        )
        agent = SmartIXAgent(env=env, seed=args.seed)
        agent.load_model(args.model)
        results = agent.test(n_episodes=20)
        env.close()
        print(f"Test Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
