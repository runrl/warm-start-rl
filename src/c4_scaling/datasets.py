from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .tictactoe_env import (
    OpponentStrategy,
    TicTacToeEnv,
    _minimax,
    minimax_agent_action,
    minimax_opponent_action,
)


def _resolve_opponent(opponent: str | None) -> OpponentStrategy | None:
    if opponent is None or opponent == "random":
        return None
    if opponent == "minimax":
        return minimax_opponent_action
    raise ValueError(f"Unsupported opponent strategy for dataset generation: {opponent}")


@dataclass
class TicTacToeExample:
    observation: torch.Tensor
    action: int
    value: float


class TicTacToeImitationDataset(Dataset):
    def __init__(self, observations: torch.Tensor, actions: torch.Tensor, values: torch.Tensor) -> None:
        if not (observations.size(0) == actions.size(0) == values.size(0)):
            raise ValueError("Observations, actions, and values must have matching lengths")
        self.observations = observations
        self.actions = actions.long()
        self.values = values.float()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.observations.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx], self.values[idx]


def generate_minimax_dataset(
    num_games: int,
    opponent: str = "random",
    seed: Optional[int] = None,
) -> List[TicTacToeExample]:
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    opponent_strategy = _resolve_opponent(opponent)
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    examples: List[TicTacToeExample] = []
    device = torch.device("cpu")

    for _ in range(num_games):
        env = TicTacToeEnv(device=device, opponent_strategy=opponent_strategy)
        observation = env.reset()
        done = False

        while not done:
            legal_actions = env.legal_actions()
            board_tensor = env.to_board_tensor()
            action = minimax_agent_action(board_tensor, legal_actions)
            value = float(_minimax(board_tensor, is_agent_turn=True))
            examples.append(TicTacToeExample(observation.clone(), action, value))

            result = env.step(action)
            observation = result.observation
            done = result.done

    if examples:
        if generator is not None:
            perm = torch.randperm(len(examples), generator=generator)
        else:
            perm = torch.randperm(len(examples))
        examples = [examples[idx] for idx in perm.tolist()]

    return examples


def save_imitation_dataset(examples: Sequence[TicTacToeExample], path: Path) -> None:
    if not examples:
        raise ValueError("Cannot save empty dataset")
    observations = torch.stack([ex.observation for ex in examples])
    actions = torch.tensor([ex.action for ex in examples], dtype=torch.long)
    values = torch.tensor([ex.value for ex in examples], dtype=torch.float32)
    torch.save({"observations": observations, "actions": actions, "values": values}, path)


def load_imitation_dataset(path: Path) -> TicTacToeImitationDataset:
    data = torch.load(path)
    observations = data["observations"].float()
    actions = data["actions"].long()
    if "values" in data:
        values = data["values"].float()
    else:
        env = TicTacToeEnv()
        values_list: List[float] = []
        for obs in observations:
            board = torch.zeros((3, 3), dtype=torch.int8)
            flat = board.view(-1)
            flat[obs[:9] > 0.5] = 1
            flat[obs[9:18] > 0.5] = -1
            values_list.append(float(_minimax(board, is_agent_turn=True)))
        values = torch.tensor(values_list, dtype=torch.float32)
    return TicTacToeImitationDataset(observations, actions, values)


def subset_dataset(
    dataset: TicTacToeImitationDataset,
    size: int,
    generator: Optional[torch.Generator] = None,
) -> TicTacToeImitationDataset:
    size = max(0, min(size, len(dataset)))
    if size == 0:
        observations = dataset.observations[:0].clone()
        actions = dataset.actions[:0].clone()
        values = dataset.values[:0].clone()
        return TicTacToeImitationDataset(observations, actions, values)

    if generator is not None:
        indices = torch.randperm(len(dataset), generator=generator)[:size]
    else:
        indices = torch.randperm(len(dataset))[:size]
    observations = dataset.observations[indices].clone()
    actions = dataset.actions[indices].clone()
    values = dataset.values[indices].clone()
    return TicTacToeImitationDataset(observations, actions, values)
