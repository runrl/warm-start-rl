from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import torch
import time

BOARD_SIZE = 3
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE

PLAYER_X = 1  # Agent
PLAYER_O = -1  # Opponent
EMPTY = 0

WIN_MASKS: Tuple[int, ...] = (
    0b111000000,
    0b000111000,
    0b000000111,
    0b100100100,
    0b010010010,
    0b001001001,
    0b100010001,
    0b001010100,
)
FULL_MASK = (1 << NUM_ACTIONS) - 1
INDEX_TO_MASK: Tuple[int, ...] = tuple(1 << idx for idx in range(NUM_ACTIONS))

OpponentStrategy = Callable[[torch.Tensor, List[int]], int]

_MINIMAX_PROFILE_ENABLED: bool = False
_MINIMAX_PROFILE_STATS: Dict[str, Dict[str, float]] = {}


def enable_minimax_profiling(enabled: bool) -> None:
    """Toggle minimax profiling and reset stats when enabling."""
    global _MINIMAX_PROFILE_ENABLED
    _MINIMAX_PROFILE_ENABLED = enabled
    if enabled:
        reset_minimax_profile()


def reset_minimax_profile() -> None:
    """Clear collected minimax profiling statistics."""
    _MINIMAX_PROFILE_STATS.clear()


def get_minimax_profile_stats() -> Dict[str, Dict[str, float]]:
    """Return a shallow copy of the minimax profiling statistics."""
    return {name: dict(values) for name, values in _MINIMAX_PROFILE_STATS.items()}


def _record_minimax_time(section: str, duration: float) -> None:
    if not _MINIMAX_PROFILE_ENABLED:
        return
    stats = _MINIMAX_PROFILE_STATS.setdefault(
        section, {"count": 0, "total_seconds": 0.0}
    )
    stats["count"] += 1
    stats["total_seconds"] += duration


def _board_tensor_to_bits(board: torch.Tensor) -> Tuple[int, int]:
    flat = board.view(-1)
    agent_bits = 0
    opponent_bits = 0
    for idx in range(NUM_ACTIONS):
        value = int(flat[idx].item())
        if value == PLAYER_X:
            agent_bits |= INDEX_TO_MASK[idx]
        elif value == PLAYER_O:
            opponent_bits |= INDEX_TO_MASK[idx]
    return agent_bits, opponent_bits


def _bits_to_board_tensor(agent_bits: int, opponent_bits: int, device: torch.device) -> torch.Tensor:
    board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.int8, device=device)
    flat = board.view(-1)
    for idx, mask in enumerate(INDEX_TO_MASK):
        if agent_bits & mask:
            flat[idx] = PLAYER_X
        elif opponent_bits & mask:
            flat[idx] = PLAYER_O
    return board


def _bits_to_obs(agent_bits: int, opponent_bits: int, current_player: int, device: torch.device) -> torch.Tensor:
    agent_plane = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=device)
    opponent_plane = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=device)
    for idx, mask in enumerate(INDEX_TO_MASK):
        if agent_bits & mask:
            agent_plane[idx] = 1.0
        elif opponent_bits & mask:
            opponent_plane[idx] = 1.0
    current = torch.tensor([1.0 if current_player == PLAYER_X else 0.0], dtype=torch.float32, device=device)
    return torch.cat([agent_plane, opponent_plane, current])


def _check_winner_bits(agent_bits: int, opponent_bits: int) -> int:
    for mask in WIN_MASKS:
        if agent_bits & mask == mask:
            return PLAYER_X
        if opponent_bits & mask == mask:
            return PLAYER_O
    return EMPTY


def _board_full(agent_bits: int, opponent_bits: int) -> bool:
    return (agent_bits | opponent_bits) == FULL_MASK


def _legal_actions_from_bits(agent_bits: int, opponent_bits: int) -> List[int]:
    occupied = agent_bits | opponent_bits
    return [idx for idx, mask in enumerate(INDEX_TO_MASK) if not (occupied & mask)]


@lru_cache(maxsize=None)
def _solve_state(agent_bits: int, opponent_bits: int, is_agent_turn: bool) -> Tuple[int, int]:
    winner = _check_winner_bits(agent_bits, opponent_bits)
    if winner == PLAYER_X:
        return 1, -1
    if winner == PLAYER_O:
        return -1, -1
    if _board_full(agent_bits, opponent_bits):
        return 0, -1

    legal_actions = _legal_actions_from_bits(agent_bits, opponent_bits)
    if not legal_actions:
        return 0, -1

    if is_agent_turn:
        best_value = -2
        best_action = legal_actions[0]
        for action in legal_actions:
            value, _ = _solve_state(agent_bits | INDEX_TO_MASK[action], opponent_bits, False)
            if value > best_value:
                best_value = value
                best_action = action
            if best_value == 1:
                break
        return best_value, best_action

    best_value = 2
    best_action = legal_actions[0]
    for action in legal_actions:
        value, _ = _solve_state(agent_bits, opponent_bits | INDEX_TO_MASK[action], True)
        if value < best_value:
            best_value = value
            best_action = action
        if best_value == -1:
            break
    return best_value, best_action


# Precompute the full minimax table once at import time for instant lookups.
_solve_state(0, 0, True)


@dataclass
class StepResult:
    observation: torch.Tensor
    reward: float
    done: bool


class TicTacToeEnv:
    """Single-agent Tic-Tac-Toe environment with a random opponent."""

    def __init__(self, device: torch.device | None = None, opponent_strategy: OpponentStrategy | None = None) -> None:
        self.device = device if device is not None else torch.device("cpu")
        self.current_player = PLAYER_X
        self.opponent_strategy = opponent_strategy
        self._agent_bits = 0
        self._opponent_bits = 0

    def reset(self) -> torch.Tensor:
        self.current_player = PLAYER_X
        self._agent_bits = 0
        self._opponent_bits = 0
        return self._current_obs()

    def legal_actions(self) -> List[int]:
        return _legal_actions_from_bits(self._agent_bits, self._opponent_bits)

    def step(self, action: int) -> StepResult:
        legal = self.legal_actions()
        if action not in legal:
            # Illegal move: immediate loss for the agent
            return StepResult(self._current_obs(), reward=-1.0, done=True)

        self._place_mark(action, PLAYER_X)
        winner = _check_winner_bits(self._agent_bits, self._opponent_bits)
        if winner == PLAYER_X:
            return StepResult(self._current_obs(), reward=1.0, done=True)

        if _board_full(self._agent_bits, self._opponent_bits):
            return StepResult(self._current_obs(), reward=0.0, done=True)

        opponent_action = self._select_opponent_action()
        self._place_mark(opponent_action, PLAYER_O)
        winner = _check_winner_bits(self._agent_bits, self._opponent_bits)
        if winner == PLAYER_O:
            return StepResult(self._current_obs(), reward=-1.0, done=True)

        if _board_full(self._agent_bits, self._opponent_bits):
            return StepResult(self._current_obs(), reward=0.0, done=True)

        return StepResult(self._current_obs(), reward=0.0, done=False)

    def _select_opponent_action(self) -> int:
        legal = self.legal_actions()
        if not legal:
            raise RuntimeError("No legal actions available for opponent.")
        if self.opponent_strategy is None:
            choice = torch.randint(len(legal), (1,), device=self.device).item()
            return legal[choice]
        board_tensor = self.to_board_tensor()
        action = self.opponent_strategy(board_tensor, legal)
        if action not in legal:
            raise ValueError("Opponent strategy produced illegal action.")
        return action

    def _place_mark(self, action: int, player: int) -> None:
        mask = INDEX_TO_MASK[action]
        if player == PLAYER_X:
            self._agent_bits |= mask
        else:
            self._opponent_bits |= mask

    def _current_obs(self) -> torch.Tensor:
        return _bits_to_obs(self._agent_bits, self._opponent_bits, PLAYER_X, self.device)

    def to_board_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        target = device if device is not None else torch.device("cpu")
        return _bits_to_board_tensor(self._agent_bits, self._opponent_bits, target)


def minimax_opponent_action(board: torch.Tensor, legal_actions: List[int]) -> int:
    start = time.perf_counter() if _MINIMAX_PROFILE_ENABLED else None
    agent_bits, opponent_bits = _board_tensor_to_bits(board)
    _, action = _solve_state(agent_bits, opponent_bits, False)
    if action not in legal_actions or action == -1:
        action = legal_actions[0]
    if start is not None:
        _record_minimax_time("minimax_opponent_lookup", time.perf_counter() - start)
    return action


def minimax_agent_action(board: torch.Tensor, legal_actions: List[int]) -> int:
    start = time.perf_counter() if _MINIMAX_PROFILE_ENABLED else None
    agent_bits, opponent_bits = _board_tensor_to_bits(board)
    _, action = _solve_state(agent_bits, opponent_bits, True)
    if action not in legal_actions or action == -1:
        action = legal_actions[0]
    if start is not None:
        _record_minimax_time("minimax_agent_lookup", time.perf_counter() - start)
    return action


def _minimax(board: torch.Tensor, is_agent_turn: bool) -> int:
    start = time.perf_counter() if _MINIMAX_PROFILE_ENABLED else None
    agent_bits, opponent_bits = _board_tensor_to_bits(board)
    value, _ = _solve_state(agent_bits, opponent_bits, is_agent_turn)
    if start is not None:
        _record_minimax_time("minimax_value_lookup", time.perf_counter() - start)
    return value
