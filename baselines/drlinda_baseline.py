"""
DRLinda baseline: single-objective DQN for index selection.

Re-implementation of DRLinda (Sadri et al., SIGMOD 2020) using the same
environment and evaluation protocol as P-NDQN. Optimizes latency only
with basic DQN (no dueling, PER, or NoisyNet).
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


class SimpleDQN(nn.Module):
    """Basic DQN without dueling or noisy layers."""

    def __init__(self, n_features: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_values = self.network(x)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e9)
        return q_values


class SimpleReplayBuffer:
    """Simple replay buffer without prioritization."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class DRLindaAgent:
    """
    DRLinda-like agent: Basic DQN for index selection.

    Single-objective: Only optimizes for query latency.
    No advanced techniques (no dueling, PER, NoisyNet).
    """

    def __init__(
        self,
        env: MORLIndexEnvironment,
        output_path: str = None,
        # Hyperparameters
        gamma: float = 0.99,
        learning_rate: float = 1e-4,  # Reduced from 1e-3 for stability
        n_episodes: int = 10000,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        # Epsilon-greedy
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_episodes: int = 500,
        # Network
        hidden_size: int = 128,
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.seed = seed

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.n_features = env.n_features
        self.n_actions = env.n_actions

        self.qnet = SimpleDQN(self.n_features, self.n_actions, hidden_size).to(self.device)
        self.qnet_target = SimpleDQN(self.n_features, self.n_actions, hidden_size).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        # Replay buffer (simple, not prioritized)
        self.memory = SimpleReplayBuffer(memory_size)

        # Output
        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"output/drlinda_{timestamp}/"
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        # Tracking
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_index_counts = []

    def get_epsilon(self) -> float:
        """Get current epsilon for exploration."""
        decay = min(1.0, self.episode / self.eps_decay_episodes)
        return self.eps_start + (self.eps_end - self.eps_start) * decay

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """Select action with epsilon-greedy."""
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return 0

        eps = self.get_epsilon() if training else 0.0

        if training and random.random() < eps:
            return np.random.choice(valid_actions)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.qnet(state_t, mask_t)
                return q_values.argmax(dim=-1).item()

    def compute_single_objective_reward(self, cost_before: float, cost_after: float) -> float:
        """
        Single-objective reward: Only latency improvement.

        This is the key difference from MORL-Index.
        DRLinda only cares about query cost reduction.
        """
        if cost_before > 0:
            improvement = (cost_before - cost_after) / cost_before
        else:
            improvement = 0.0
        return improvement * 10.0  # Scale similar to MORL-Index

    def update(self) -> Optional[float]:
        """Update Q-network with simple DQN."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch (uniform, not prioritized)
        samples = self.memory.sample(self.batch_size)

        # Unpack
        states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long, device=self.device)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32, device=self.device)

        # Current Q-values
        current_q = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (simple DQN, not Double DQN)
        with torch.no_grad():
            next_q = self.qnet_target(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Huber loss for stability (robust to outliers)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent loss explosion
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def hard_update_target(self):
        """Hard update target network."""
        self.qnet_target.load_state_dict(self.qnet.state_dict())

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

            # Compute SINGLE-OBJECTIVE reward (latency only)
            cost_before = info.get('cost_before', 1000)
            cost_after = info.get('cost_after', 1000)
            reward = self.compute_single_objective_reward(cost_before, cost_after)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Store transition
            self.memory.add((state, action, reward, next_state, float(done)))

            # Update
            loss = self.update()
            if loss is not None:
                losses.append(loss)

            # Target update
            if self.total_steps % self.target_update_freq == 0:
                self.hard_update_target()

            if done:
                break

            state = next_state
            action_mask = next_mask

        metrics = self.env.get_final_metrics()

        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'loss': np.mean(losses) if losses else 0,
            'index_count': metrics['index_count'],
            'epsilon': self.get_epsilon(),
        }

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("DRLinda BASELINE TRAINING (Single-Objective DQN)")
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

            # Log
            log_line = f"{ep}\t{result['reward']:.2f}\t{result['steps']}\t"
            log_line += f"{result['loss']:.4f}\t{result['index_count']}\t{result['epsilon']:.3f}"
            log_file.write(log_line + '\n')
            log_file.flush()

            if ep % 100 == 0:
                print(f"Episode {ep:5d} | Reward: {result['reward']:7.2f} | "
                      f"Avg(50): {avg_reward:7.2f} | Indexes: {result['index_count']:2d} | "
                      f"Eps: {result['epsilon']:.3f}")

            # Save best model
            if avg_reward > best_reward and ep >= 50:
                best_reward = avg_reward
                self.save_model('model_best.pt')

        # Final save
        self.save_model('model_final.pt')
        log_file.close()

        # Save stats
        stats = {
            'method': 'DRLinda',
            'n_episodes': self.n_episodes,
            'total_steps': self.total_steps,
            'best_avg_reward': best_reward,
            'final_avg_reward': np.mean(self.episode_rewards[-50:]),
            'avg_index_count': np.mean(self.episode_index_counts[-50:]),
            'seed': self.seed,
        }
        with open(self.output_path + 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print("DRLinda TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best avg reward: {best_reward:.2f}")
        print(f"Final avg reward: {stats['final_avg_reward']:.2f}")

        return stats

    def test(self, n_episodes: int = 20) -> Dict:
        """Test trained model."""
        self.qnet.eval()

        all_rewards = []
        all_indexes = []
        all_costs = []

        for ep in range(n_episodes):
            state, info = self.env.reset()
            action_mask = info['action_mask']
            episode_reward = 0
            episode_costs = []

            while True:
                action = self.select_action(state, action_mask, training=False)
                next_state, _, terminated, truncated, info = self.env.step(action)

                cost_before = info.get('cost_before', 1000)
                cost_after = info.get('cost_after', 1000)
                reward = self.compute_single_objective_reward(cost_before, cost_after)
                episode_reward += reward
                episode_costs.append(cost_after)

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
            'method': 'DRLinda',
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_indexes': np.mean(all_indexes),
            'std_indexes': np.std(all_indexes),
            'avg_cost': np.mean(all_costs) if all_costs else 0,
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


def train_drlinda(
    workload_path: str = 'data/workload/chbench_htap_balanced.sql',
    n_episodes: int = 10000,
    seed: int = 42,
) -> Dict:
    """Train DRLinda baseline."""
    env = MORLIndexEnvironment(
        workload_path=workload_path,
        hypo=True,
        max_steps_per_episode=50,
        max_indexes=15,
        memory_budget_mb=512,
    )

    agent = DRLindaAgent(
        env=env,
        n_episodes=n_episodes,
        seed=seed,
    )

    stats = agent.train()
    test_results = agent.test(n_episodes=20)

    env.close()

    return {**stats, **test_results}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Training DRLinda Baseline...")
    results = train_drlinda(n_episodes=args.episodes, seed=args.seed)
    print(f"\n=== DRLinda Results ===")
    print(f"Avg Reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"Avg Indexes: {results['avg_indexes']:.1f}")
