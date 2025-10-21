from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .policy_gradient import PolicyNetwork, ValueNetwork, _mask_logits


@dataclass
class Transition:
    observation: torch.Tensor
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    legal_actions: Sequence[int]


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    num_heads: int = 4
    depth: int = 2


class PPOAgent:
    def __init__(self, obs_dim: int, config: PPOConfig, device: torch.device) -> None:
        self.device = device
        self.config = config

        self.policy = PolicyNetwork(
            obs_dim=obs_dim, hidden_dim=config.hidden_dim, n_head=config.num_heads, depth=config.depth
        ).to(device)
        self.value_function = ValueNetwork(
            obs_dim=obs_dim, hidden_dim=config.hidden_dim, n_head=config.num_heads, depth=config.depth
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_function.parameters()),
            lr=config.learning_rate,
        )

    def act(self, observation: torch.Tensor, legal_actions: Sequence[int]) -> tuple[int, float, float]:
        obs = observation.to(self.device)
        logits = self.policy(obs.unsqueeze(0)).squeeze(0)
        masked_logits = _mask_logits(logits, list(legal_actions))
        distribution = Categorical(logits=masked_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        value = self.value_function(obs.unsqueeze(0)).squeeze(0)
        return int(action.item()), float(log_prob.detach().cpu().item()), float(value.detach().cpu().item())

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor, legal_actions: List[Sequence[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(observations)
        log_probs = []
        entropies = []
        for idx, mask_indices in enumerate(legal_actions):
            masked_logits = _mask_logits(logits[idx], list(mask_indices))
            distribution = Categorical(logits=masked_logits)
            log_probs.append(distribution.log_prob(actions[idx]))
            entropies.append(distribution.entropy())
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        values = self.value_function(observations)
        return log_probs_tensor, entropies_tensor, values

    def update(self, transitions: Iterable[Transition]) -> dict[str, float]:
        batch = list(transitions)
        if not batch:
            return {}

        observations = torch.stack([t.observation for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in batch], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
        values = torch.tensor([t.value for t in batch], dtype=torch.float32, device=self.device)
        legal_actions = [t.legal_actions for t in batch]

        advantages, returns = self._compute_gae(rewards, dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = len(batch)

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_legal = [legal_actions[i.item()] for i in mb_indices]

                new_log_probs, entropies, values_pred = self.evaluate_actions(mb_obs, mb_actions, mb_legal)

                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surrogate1 = ratios * mb_advantages
                surrogate2 = torch.clamp(ratios, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(values_pred, mb_returns)
                entropy_loss = entropies.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.detach().cpu().item()) * mb_obs.size(0)
                total_value_loss += float(value_loss.detach().cpu().item()) * mb_obs.size(0)
                total_entropy += float(entropy_loss.detach().cpu().item()) * mb_obs.size(0)

        updates = self.config.update_epochs * (total_steps // self.config.minibatch_size + (total_steps % self.config.minibatch_size != 0))
        if updates == 0:
            updates = 1
        denom = total_steps * self.config.update_epochs
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

    def _compute_gae(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[dones] = 0.0

        deltas = rewards + self.config.gamma * next_values - values
        deltas[dones] = rewards[dones] - values[dones]

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 0.0 if dones[step] else 1.0
            gae = deltas[step] + self.config.gamma * self.config.gae_lambda * gae * mask
            advantages[step] = gae
        returns = advantages + values
        return advantages, returns
