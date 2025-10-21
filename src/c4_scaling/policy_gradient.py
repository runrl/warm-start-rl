from __future__ import annotations

from typing import Callable, List

import torch
from torch import nn
from torch.distributions import Categorical

from .tictactoe_env import NUM_ACTIONS, TicTacToeEnv


def _mask_logits(logits: torch.Tensor, legal_actions: List[int]) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    indices = torch.tensor(legal_actions, dtype=torch.long, device=logits.device)
    masked.index_copy_(0, indices, logits.index_select(0, indices))
    return masked


VOCAB_SIZE = 4  # empty, agent token, opponent token, start token
SEQ_LEN = NUM_ACTIONS + 1


def _observation_to_tokens(observation: torch.Tensor) -> torch.Tensor:
    single = observation.dim() == 1
    if single:
        observation = observation.unsqueeze(0)
    board_agent = observation[:, :NUM_ACTIONS]
    board_opponent = observation[:, NUM_ACTIONS : 2 * NUM_ACTIONS]

    tokens_board = torch.zeros(observation.size(0), NUM_ACTIONS, dtype=torch.long, device=observation.device)
    agent_mask = board_agent > 0.5
    opponent_mask = board_opponent > 0.5
    tokens_board = torch.where(agent_mask, torch.ones_like(tokens_board), tokens_board)
    tokens_board = torch.where(opponent_mask, torch.full_like(tokens_board, 2), tokens_board)

    start_token = torch.full((observation.size(0), 1), 3, dtype=torch.long, device=observation.device)
    tokens = torch.cat([tokens_board, start_token], dim=1)
    return tokens.squeeze(0) if single else tokens


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        ln_x = self.ln1(x)
        y, _ = self.attn(ln_x, ln_x, ln_x, attn_mask=attn_mask)
        x = x + self.dropout(y)
        x = x + self.mlp(self.ln2(x))
        return x


class _BoardTransformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_head, dropout=dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d_model)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed[:, : tokens.size(1), :]
        mask = self.causal_mask[: tokens.size(1), : tokens.size(1)]
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_head: int = 4, depth: int = 2) -> None:
        super().__init__()
        d_model = hidden_dim
        self.transformer = _BoardTransformer(d_model=d_model, n_head=n_head, depth=depth)
        self.logit_head = nn.Linear(d_model, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        tokens = _observation_to_tokens(x)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        hidden = self.transformer(tokens)
        global_state = hidden[:, -1, :]
        action_logits = self.logit_head(global_state)
        return action_logits.squeeze(0) if single else action_logits


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_head: int = 4, depth: int = 2) -> None:
        super().__init__()
        d_model = hidden_dim
        self.transformer = _BoardTransformer(d_model=d_model, n_head=n_head, depth=depth)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        tokens = _observation_to_tokens(x)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        hidden = self.transformer(tokens)
        cls_token = hidden[:, -1, :]
        values = self.value_head(cls_token).squeeze(-1)
        return values.squeeze(0) if single else values


def evaluate_policy(
    env_factory: Callable[[], TicTacToeEnv],
    policy: PolicyNetwork,
    episodes: int,
    device: torch.device,
    *,
    stochastic: bool = False,
    batch_size: int | None = None,
) -> tuple[float, float]:
    was_training = policy.training
    policy.eval()
    wins = 0
    draws = 0
    try:
        if episodes <= 0:
            return 0.0, 0.0
        if batch_size is None:
            batch_size = episodes
        batch_size = max(1, min(int(batch_size), int(episodes)))
        remaining = int(episodes)
        with torch.inference_mode():
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                envs = [env_factory() for _ in range(current_batch)]
                obs_buffer = [env.reset() for env in envs]
                done_flags = [False] * current_batch
                final_rewards = [0.0] * current_batch
                active = current_batch

                while active > 0:
                    observations = torch.stack(obs_buffer, dim=0).to(device)
                    logits = policy(observations)
                    for idx, env in enumerate(envs):
                        if done_flags[idx]:
                            continue
                        legal = env.legal_actions()
                        masked_logits = _mask_logits(logits[idx], legal)
                        if stochastic:
                            distribution = Categorical(logits=masked_logits)
                            action = int(distribution.sample().item())
                        else:
                            action = int(torch.argmax(masked_logits).item())
                        result = env.step(action)
                        obs_buffer[idx] = result.observation
                        if result.done:
                            done_flags[idx] = True
                            active -= 1
                            final_rewards[idx] = result.reward
                            obs_buffer[idx] = torch.zeros_like(obs_buffer[idx])

                for reward in final_rewards:
                    if reward > 0:
                        wins += 1
                    elif reward == 0:
                        draws += 1
                remaining -= current_batch
    finally:
        policy.train(was_training)
    denom = max(episodes, 1)
    return wins / denom, draws / denom
