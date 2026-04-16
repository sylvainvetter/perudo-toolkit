"""
M4 — Trained CFR policy: serialisable strategy table.

A Policy wraps the time-averaged strategy accumulated during CFR training.
It can be saved to disk (pickle) and loaded back for deployment in CFRBot.

The average strategy is stored as strategy_sum per info state.
At query time we normalise: probs[a] = strategy_sum[a] / sum(strategy_sum).
For unseen states (not visited during training) we return a uniform
distribution over legal actions as a safe fallback.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from perudo.m4.infostate import N_ACTIONS


@dataclass
class Policy:
    """
    Time-averaged CFR strategy table.

    Attributes:
        strategy_sum : info_key → accumulated strategy weight vector
        n_iters      : number of training iterations used to build this policy
    """

    strategy_sum: dict[tuple[Any, ...], np.ndarray]
    n_iters: int

    # ------------------------------------------------------------------ query

    def get_probs(self, info_key: tuple[Any, ...], mask: np.ndarray) -> np.ndarray:
        """
        Return a normalised probability vector over N_ACTIONS.

        Illegal actions always get probability 0.
        Falls back to uniform over legal actions for unseen states.
        """
        s = self.strategy_sum.get(info_key)
        if s is not None:
            masked: np.ndarray = np.where(mask, s, 0.0)
            total = float(masked.sum())
            if total > 0:
                return masked / total

        n_legal = int(mask.sum())
        if n_legal == 0:
            return np.zeros(N_ACTIONS)
        result: np.ndarray = np.where(mask, 1.0 / n_legal, 0.0)
        return result

    def knows(self, info_key: tuple[Any, ...]) -> bool:
        """Return True if this info state was visited during training."""
        return info_key in self.strategy_sum

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        """Serialise the policy to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> Policy:
        """Load a previously saved policy from disk."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Policy, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------ info

    @property
    def n_states(self) -> int:
        """Number of distinct information states covered by this policy."""
        return len(self.strategy_sum)

    def __repr__(self) -> str:
        return (
            f"Policy(n_states={self.n_states:,}, n_iters={self.n_iters:,})"
        )
