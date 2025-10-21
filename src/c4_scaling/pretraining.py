from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .datasets import TicTacToeImitationDataset
from .policy_gradient import PolicyNetwork


@dataclass
class PretrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    shuffle: bool = True


def pretrain_policy(
    policy: PolicyNetwork,
    dataset: TicTacToeImitationDataset,
    config: PretrainingConfig,
    device: torch.device,
    log_fn: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> Dict[str, float]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot pretrain policy")

    policy = policy.to(device)
    policy.train()

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for observations, actions in dataloader:
            obs = observations.to(device)
            target = actions.to(device)

            logits = policy(obs)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits.detach(), dim=-1)
            total_correct += int((preds == target).sum().item())
            total_loss += float(loss.detach().cpu().item()) * obs.size(0)
            total_examples += obs.size(0)

        avg_loss = total_loss / max(total_examples, 1)
        accuracy = total_correct / max(total_examples, 1)
        if log_fn is not None:
            log_fn({"pretrain/loss": avg_loss, "pretrain/accuracy": accuracy}, epoch)

    return {"loss": avg_loss, "accuracy": accuracy}
