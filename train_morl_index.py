"""
P-NDQN training script.

Algorithm: Rainbow DQN (Dueling + NoisyNet + PER + N-step) with multi-objective
reward and workload-adaptive scalarization.

Usage:
    python train_morl_index.py --mode train --episodes 1000
    python train_morl_index.py --mode test --model results/models/morl_seed42/model_best.pt
"""

import os
import sys
import json
import time
import math
import random
import argparse
import collections
import numpy as np
from typing import Optional, List, Dict, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from environment_morl import MORLIndexEnvironment


class NoisyLinear(nn.Module):
    """Factorized NoisyLinear layer for exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_sigma(self, scale: float):
        """Scale the sigma parameters for exploration annealing."""
        self.weight_sigma.data *= scale
        self.bias_sigma.data *= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNoisyNet(nn.Module):
    """
    Dueling DQN with NoisyNet layers and action masking support.

    Architecture:
    - Shared feature extractor
    - Separate value and advantage streams (toggleable via use_dueling)
    - NoisyNet for exploration (toggleable via use_noisy)
    - Action masking for constraint satisfaction
    """

    def __init__(self, n_features: int, n_actions: int, hidden_size: int = 256,
                 use_noisy: bool = True, use_dueling: bool = True):
        super().__init__()
        self.use_noisy = use_noisy
        self.use_dueling = use_dueling

        LinearClass = NoisyLinear if use_noisy else nn.Linear

        # Shared layers
        self.shared = nn.Sequential(
            LinearClass(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            LinearClass(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        if use_dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                LinearClass(hidden_size, hidden_size // 2),
                nn.ReLU(),
                LinearClass(hidden_size // 2, 1),
            )
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                LinearClass(hidden_size, hidden_size // 2),
                nn.ReLU(),
                LinearClass(hidden_size // 2, n_actions),
            )
        else:
            # Standard Q-head (no dueling)
            self.q_head = nn.Sequential(
                LinearClass(hidden_size, hidden_size // 2),
                nn.ReLU(),
                LinearClass(hidden_size // 2, n_actions),
            )

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional action masking."""
        features = self.shared(x)

        if self.use_dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Dueling: Q = V + (A - mean(A))
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            q_values = self.q_head(features)

        # Apply action mask
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e9)

        return q_values

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        if not self.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def scale_noise_sigma(self, scale: float):
        """Scale sigma in all NoisyLinear layers for exploration annealing."""
        if not self.use_noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.scale_sigma(scale)


class PrioritizedReplay:
    """Prioritized Experience Replay with proper episode handling."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = 1e-6

        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def __len__(self):
        return len(self.data)

    def add(self, transition, priority: float = None):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition

        self.priorities[self.pos] = priority if priority else self.max_priority
        self.max_priority = max(self.max_priority, self.priorities[self.pos])
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, frame_idx: int):
        n = len(self.data)
        priorities = self.priorities[:n]

        probs = (priorities + self.eps) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.data[i] for i in indices]

        # Anneal beta
        beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = abs(prio) + self.eps
            self.max_priority = max(self.max_priority, self.priorities[idx])


class UniformReplay:
    """Simple uniform experience replay buffer (no prioritization)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def add(self, transition, priority: float = None):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, frame_idx: int):
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        samples = [self.data[i] for i in indices]
        weights = torch.ones(batch_size, dtype=torch.float32)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        pass  # No-op for uniform replay


class MORLIndexAgent:
    """Rainbow DQN agent for multi-objective HTAP index selection."""

    def __init__(
        self,
        env: MORLIndexEnvironment,
        output_path: str = None,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        n_episodes: int = 30000,
        memory_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 1000,
        n_step: int = 3,
        hidden_size: int = 256,
        entropy_coef: float = 0.02,
        sigma_decay: float = 0.9995,
        min_sigma_scale: float = 0.1,
        # Ablation flags
        use_per: bool = True,
        use_noisy: bool = True,
        use_dueling: bool = True,
    ):
        self.env = env
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.n_step = n_step

        # Ablation flags
        self.use_per = use_per
        self.use_noisy = use_noisy
        self.use_dueling = use_dueling

        # Entropy regularization (FIX 1)
        self.entropy_coef = entropy_coef

        # NoisyNet sigma annealing (FIX 3)
        self.sigma_decay = sigma_decay
        self.min_sigma_scale = min_sigma_scale
        self.current_sigma_scale = 1.0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Networks
        self.n_features = env.n_features
        self.n_actions = env.n_actions

        self.qnet = DuelingNoisyNet(self.n_features, self.n_actions, hidden_size,
                                     use_noisy=use_noisy, use_dueling=use_dueling).to(self.device)
        self.qnet_target = DuelingNoisyNet(self.n_features, self.n_actions, hidden_size,
                                            use_noisy=use_noisy, use_dueling=use_dueling).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

        # Replay buffer
        if use_per:
            self.memory = PrioritizedReplay(memory_size)
        else:
            self.memory = UniformReplay(memory_size)

        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)

        # Output
        if output_path is None:
            timestamp = int(time.time())
            self.output_path = f"output/morl_index_{timestamp}/"
        else:
            self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        # Tracking
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_index_counts = []

        # Action distribution tracking for mode collapse detection (FIX 4)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.episode_action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self.action_entropy_history = []
        self.unique_actions_history = []
        self.entropy_warning_threshold = 0.3  # Warn if entropy drops below this

        # Save hyperparameters
        self._save_hyperparameters(locals())

        print("=" * 60)
        print("MORL-INDEX AGENT (Rainbow DQN)")
        print("=" * 60)
        print(f"Features: {self.n_features}, Actions: {self.n_actions}")
        print(f"Episodes: {n_episodes}, Batch: {batch_size}")
        print(f"Gamma: {gamma}, LR: {learning_rate}")
        print(f"Ablation: PER={use_per}, NoisyNet={use_noisy}, Dueling={use_dueling}")
        print(f"N-step: {n_step}, Target update: {target_update_freq}")
        print(f"Eps: {eps_start} -> {eps_end} over {eps_decay_episodes} episodes")
        print(f"Entropy coef: {entropy_coef}, Sigma decay: {sigma_decay}")
        print(f"Output: {self.output_path}")
        print("=" * 60)

    def _save_hyperparameters(self, params: Dict):
        """Save hyperparameters to file."""
        save_params = {k: v for k, v in params.items() if k not in ['self', 'env', 'output_path']}
        save_params['n_features'] = self.n_features
        save_params['n_actions'] = self.n_actions
        with open(self.output_path + 'hyperparameters.json', 'w') as f:
            json.dump(save_params, f, indent=2, default=str)

    def get_epsilon(self) -> float:
        """Get current epsilon."""
        decay = min(1.0, self.episode / self.eps_decay_episodes)
        return self.eps_start + (self.eps_end - self.eps_start) * decay

    def compute_action_entropy(self, counts: np.ndarray) -> float:
        """Compute normalized entropy of action distribution (0 to 1)."""
        if counts.sum() == 0:
            return 0.0
        probs = counts / counts.sum()
        # Avoid log(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True
    ) -> int:
        """Select action with epsilon-greedy + NoisyNet + action masking."""
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            # Fallback - shouldn't happen
            return 0

        eps = self.get_epsilon() if training else 0.0

        if training and random.random() < eps:
            # Random exploration from valid actions
            return np.random.choice(valid_actions)
        else:
            # Greedy with NoisyNet exploration
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.qnet.train()  # Enable noise
            self.qnet.reset_noise()

            with torch.no_grad():
                q_values = self.qnet(state_t, mask_t)
                return q_values.argmax(dim=-1).item()

    def _compute_n_step_return(self) -> Tuple:
        """Compute N-step return from buffer."""
        reward = 0.0
        for i, (_, _, r, _, d, _) in enumerate(reversed(list(self.n_step_buffer))):
            reward = r + self.gamma * reward * (1 - d)

        first = self.n_step_buffer[0]
        last = self.n_step_buffer[-1]

        return (
            first[0],  # state
            first[1],  # action
            reward,    # n-step return
            last[3],   # next_state
            last[4],   # done
            first[5],  # action_mask
        )

    def update(self) -> Optional[float]:
        """Update Q-network."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        samples, indices, weights = self.memory.sample(self.batch_size, self.total_steps)
        weights = weights.to(self.device)

        # Unpack
        states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long, device=self.device)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32, device=self.device)

        # Double DQN target (compute FIRST, before current_q to avoid in-place modification issues)
        with torch.no_grad():
            self.qnet.reset_noise()
            next_actions = self.qnet(next_states).argmax(dim=1)
            next_q = self.qnet_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        # Current Q-values (compute AFTER target to avoid gradient issues with reset_noise)
        self.qnet.reset_noise()
        all_q_values = self.qnet(states)  # [batch, n_actions] - for entropy computation
        current_q = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD errors for PER
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Compute entropy bonus from Q-values (soft policy) - FIX 1
        temperature = 1.0
        q_probs = F.softmax(all_q_values / temperature, dim=-1)
        log_probs = F.log_softmax(all_q_values / temperature, dim=-1)
        entropy = -torch.sum(q_probs * log_probs, dim=-1).mean()

        # Weighted Huber loss with entropy bonus
        td_loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        # Entropy bonus encourages exploration (subtracted because we minimize loss)
        loss = td_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors)

        return loss.item(), entropy.item()

    def soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.qnet_target.parameters(), self.qnet.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_episode(self) -> Dict:
        """Train one episode."""
        state, info = self.env.reset()
        action_mask = info['action_mask']

        # CRITICAL: Clear N-step buffer at episode start
        self.n_step_buffer.clear()

        # Reset episode action counts (FIX 4)
        self.episode_action_counts = np.zeros(self.n_actions, dtype=np.int64)

        episode_reward = 0
        episode_steps = 0
        losses = []
        entropies = []  # Track entropy values (FIX 1)

        while True:
            # Select action
            action = self.select_action(state, action_mask, training=True)

            # Track action for diversity metrics (FIX 4)
            self.action_counts[action] += 1
            self.episode_action_counts[action] += 1

            # Execute
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            next_mask = info['action_mask']

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Add to N-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, float(done), action_mask))

            # Store N-step transition
            if len(self.n_step_buffer) == self.n_step:
                transition = self._compute_n_step_return()
                self.memory.add(transition)

            # Update
            update_result = self.update()
            if update_result is not None:
                loss, ent = update_result
                losses.append(loss)
                entropies.append(ent)

            # Target update
            if self.total_steps % self.target_update_freq == 0:
                self.soft_update_target()

            # Apply NoisyNet sigma annealing (FIX 3)
            if self.current_sigma_scale > self.min_sigma_scale and self.total_steps % 100 == 0:
                decay_factor = self.sigma_decay ** 100  # Batch update
                self.current_sigma_scale *= decay_factor
                self.qnet.scale_noise_sigma(decay_factor)
                self.qnet_target.scale_noise_sigma(decay_factor)

            if done:
                # Flush remaining N-step buffer
                while len(self.n_step_buffer) > 0:
                    transition = self._compute_n_step_return()
                    self.memory.add(transition)
                    self.n_step_buffer.popleft()
                break

            state = next_state
            action_mask = next_mask

        # Compute action diversity metrics for this episode (FIX 4)
        episode_entropy = self.compute_action_entropy(self.episode_action_counts)
        unique_actions = int(np.sum(self.episode_action_counts > 0))

        self.action_entropy_history.append(episode_entropy)
        self.unique_actions_history.append(unique_actions)

        # Mode collapse warning
        if episode_entropy < self.entropy_warning_threshold and self.episode > 50:
            print(f"  WARNING: Low action entropy ({episode_entropy:.3f}) - possible mode collapse!")

        # Get final metrics
        metrics = self.env.get_final_metrics()

        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'loss': np.mean(losses) if losses else 0,
            'entropy': np.mean(entropies) if entropies else 0,
            'action_entropy': episode_entropy,
            'unique_actions': unique_actions,
            'index_count': metrics['index_count'],
            'memory_used_mb': metrics['memory_used_mb'],
            'epsilon': self.get_epsilon(),
            'sigma_scale': self.current_sigma_scale,
            'pareto': metrics['pareto_metrics'],
        }

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)

        log_file = open(self.output_path + 'log.txt', 'w')
        log_file.write("episode\treward\tsteps\tloss\tindexes\tmemory_mb\tepsilon\tact_entropy\tunique_acts\n")

        best_reward = float('-inf')
        reward_window = deque(maxlen=50)

        for ep in range(self.n_episodes):
            self.episode = ep
            result = self.train_episode()

            self.episode_rewards.append(result['reward'])
            self.episode_index_counts.append(result['index_count'])
            reward_window.append(result['reward'])
            avg_reward = np.mean(reward_window)

            # Log (with new action diversity metrics)
            log_line = f"{ep}\t{result['reward']:.2f}\t{result['steps']}\t"
            log_line += f"{result['loss']:.4f}\t{result['index_count']}\t"
            log_line += f"{result['memory_used_mb']:.1f}\t{result['epsilon']:.3f}\t"
            log_line += f"{result['action_entropy']:.3f}\t{result['unique_actions']}"
            log_file.write(log_line + '\n')
            log_file.flush()

            # Print progress (with action entropy)
            if ep % 10 == 0:
                print(f"Episode {ep:4d} | Reward: {result['reward']:7.2f} | "
                      f"Avg(50): {avg_reward:7.2f} | Indexes: {result['index_count']:2d} | "
                      f"ActEnt: {result['action_entropy']:.3f} | Unique: {result['unique_actions']} | "
                      f"Eps: {result['epsilon']:.3f}")

            # Save best model
            if avg_reward > best_reward and ep >= 50:
                best_reward = avg_reward
                self.save_model('model_best.pt')
                print(f"  -> New best model! Avg reward: {avg_reward:.2f}")

            # Checkpoint
            if ep % 200 == 0 and ep > 0:
                self.save_model(f'model_ep{ep}.pt')

        # Final save
        self.save_model('model_final.pt')
        log_file.close()

        # Action distribution analysis (FIX 4)
        action_probs = self.action_counts / self.action_counts.sum() if self.action_counts.sum() > 0 else self.action_counts
        top_5_actions = np.argsort(action_probs)[-5:][::-1]

        # Save training stats
        stats = {
            'n_episodes': self.n_episodes,
            'total_steps': self.total_steps,
            'best_avg_reward': best_reward,
            'final_avg_reward': np.mean(self.episode_rewards[-50:]),
            'avg_index_count': np.mean(self.episode_index_counts[-50:]),
            # Action diversity metrics
            'final_action_entropy': float(self.compute_action_entropy(self.action_counts)),
            'avg_episode_entropy': float(np.mean(self.action_entropy_history[-50:])) if self.action_entropy_history else 0,
            'total_unique_actions_used': int(np.sum(self.action_counts > 0)),
            'top_5_actions': [(int(a), float(action_probs[a]), self.env.columns[a]) for a in top_5_actions],
        }
        with open(self.output_path + 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        # Plot
        self._plot_training()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best avg reward: {best_reward:.2f}")
        print(f"Final avg reward: {stats['final_avg_reward']:.2f}")
        print(f"Avg index count: {stats['avg_index_count']:.1f}")
        print(f"Output: {self.output_path}")

        return stats

    def _plot_training(self):
        """Plot training curves."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))  # Changed to 3 subplots
            window = 50

            # Rewards
            ax1 = axes[0]
            ax1.plot(self.episode_rewards, alpha=0.3, color='blue')
            if len(self.episode_rewards) > window:
                smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.episode_rewards)), smoothed, color='red', label=f'MA-{window}')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.legend()
            ax1.grid(True)

            # Index counts
            ax2 = axes[1]
            ax2.plot(self.episode_index_counts, alpha=0.5, color='green')
            if len(self.episode_index_counts) > window:
                smoothed = np.convolve(self.episode_index_counts, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(self.episode_index_counts)), smoothed, color='red')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Index Count')
            ax2.set_title('Final Index Count per Episode')
            ax2.grid(True)

            # Action entropy (FIX 4 - mode collapse detection)
            ax3 = axes[2]
            if self.action_entropy_history:
                ax3.plot(self.action_entropy_history, alpha=0.5, color='purple')
                if len(self.action_entropy_history) > window:
                    smoothed = np.convolve(self.action_entropy_history, np.ones(window)/window, mode='valid')
                    ax3.plot(range(window-1, len(self.action_entropy_history)), smoothed, color='red', label=f'MA-{window}')
                ax3.axhline(y=self.entropy_warning_threshold, color='orange', linestyle='--', label='Warning threshold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Action Entropy (normalized)')
            ax3.set_title('Action Diversity (Mode Collapse Detection)')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True)

            plt.tight_layout()
            plt.savefig(self.output_path + 'training_curves.png', dpi=150)
            plt.close()

        except ImportError:
            print("matplotlib not available, skipping plots")

    def test(self, n_episodes: int = 20):
        """Test trained model."""
        print("\n" + "=" * 60)
        print("TESTING")
        print("=" * 60)

        self.qnet.eval()

        all_rewards = []
        all_indexes = []
        all_memory = []
        all_costs = []

        for ep in range(n_episodes):
            state, info = self.env.reset()
            action_mask = info['action_mask']
            episode_reward = 0
            episode_costs = []

            while True:
                action = self.select_action(state, action_mask, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                # Track query cost
                if 'cost_after' in info:
                    episode_costs.append(info['cost_after'])

                if terminated or truncated:
                    break

                state = next_state
                action_mask = info['action_mask']

            metrics = self.env.get_final_metrics()
            all_rewards.append(episode_reward)
            all_indexes.append(metrics['index_count'])
            all_memory.append(metrics['memory_used_mb'])
            if episode_costs:
                all_costs.append(np.mean(episode_costs))

            print(f"Episode {ep}: reward={episode_reward:.2f}, "
                  f"indexes={metrics['index_count']}, "
                  f"memory={metrics['memory_used_mb']:.1f}MB")
            if metrics['active_indexes']:
                print(f"  Indexes: {metrics['active_indexes'][:5]}...")

        avg_cost = np.mean(all_costs) if all_costs else 0

        print("\n=== Test Summary ===")
        print(f"Avg reward: {np.mean(all_rewards):.2f} (+/- {np.std(all_rewards):.2f})")
        print(f"Avg indexes: {np.mean(all_indexes):.1f}")
        print(f"Avg memory: {np.mean(all_memory):.1f} MB")
        print(f"Avg cost: {avg_cost:.2f}")

        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'avg_indexes': np.mean(all_indexes),
            'avg_memory_mb': np.mean(all_memory),
            'avg_cost': avg_cost,
        }

    def save_model(self, filename: str):
        """Save model."""
        torch.save({
            'qnet': self.qnet.state_dict(),
            'qnet_target': self.qnet_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'total_steps': self.total_steps,
        }, self.output_path + filename)

    def load_model(self, filepath: str):
        """Load model."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.qnet.load_state_dict(ckpt['qnet'])
        self.qnet_target.load_state_dict(ckpt['qnet_target'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.episode = ckpt.get('episode', 0)
        self.total_steps = ckpt.get('total_steps', 0)


def main():
    parser = argparse.ArgumentParser(description='MORL-Index Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default=None, help='Model path for testing')
    parser.add_argument('--workload', type=str, default='data/workloads/chbench_htap_balanced.sql')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--max-indexes', type=int, default=15)
    parser.add_argument('--memory-budget', type=float, default=512, help='Memory budget in MB')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--no-hypo', action='store_true')
    # Multi-objective weights
    parser.add_argument('--latency-weight', type=float, default=1.0)
    parser.add_argument('--write-weight', type=float, default=1.0)
    parser.add_argument('--memory-weight', type=float, default=0.5)
    parser.add_argument('--phase-adaptation', type=float, default=0.5)
    # Exploration and entropy (FIX 1, 3)
    parser.add_argument('--entropy-coef', type=float, default=0.02, help='Entropy coefficient for exploration')
    parser.add_argument('--sigma-decay', type=float, default=0.9995, help='NoisyNet sigma decay rate')
    parser.add_argument('--eps-end', type=float, default=0.05, help='Final epsilon value')
    parser.add_argument('--eps-decay', type=int, default=1000, help='Episodes for epsilon decay')
    # Ablation flags
    parser.add_argument('--no-per', action='store_true', help='Ablation: disable PER, use uniform replay')
    parser.add_argument('--no-noisy', action='store_true', help='Ablation: disable NoisyNet, use nn.Linear')
    parser.add_argument('--no-dueling', action='store_true', help='Ablation: disable Dueling, use standard head')
    parser.add_argument('--no-action-mask', action='store_true', help='Ablation: disable action masking')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Create environment
    env = MORLIndexEnvironment(
        workload_path=args.workload,
        hypo=not args.no_hypo,
        max_steps_per_episode=args.max_steps,
        max_indexes=args.max_indexes,
        memory_budget_mb=args.memory_budget,
        latency_weight=args.latency_weight,
        write_weight=args.write_weight,
        memory_weight=args.memory_weight,
        phase_adaptation=args.phase_adaptation,
        use_action_mask=not args.no_action_mask,
    )

    # Create agent (with diversity and entropy fixes)
    agent = MORLIndexAgent(
        env=env,
        output_path=args.output_dir,
        n_episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=0.99,
        n_step=3,
        hidden_size=256,
        target_update_freq=100,
        tau=0.005,
        eps_start=1.0,
        eps_end=args.eps_end,  # FIX 3: Higher floor (0.05)
        eps_decay_episodes=args.eps_decay,  # FIX 3: Slower decay (1000)
        entropy_coef=args.entropy_coef,  # FIX 1: Entropy bonus
        sigma_decay=args.sigma_decay,  # FIX 3: Sigma annealing
        use_per=not args.no_per,
        use_noisy=not args.no_noisy,
        use_dueling=not args.no_dueling,
    )

    if args.mode == 'train':
        agent.train()
        print("\nRunning post-training test...")
        agent.test(n_episodes=20)
    else:
        if not args.model:
            print("Error: --model required for test mode")
            return
        agent.load_model(args.model)
        agent.test(n_episodes=20)

    env.close()


if __name__ == '__main__':
    main()
