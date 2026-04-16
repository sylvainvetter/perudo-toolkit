"""
M3 — Strategy implementations for the Monte Carlo simulator.

Available strategies:
  RandomLegal   — uniformly picks among legal actions (raise / liar / exact)
  Honest        — bids the expected count for its best value; calls liar if forced
  ThresholdBot  — delegates to M2 recommender
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from perudo.core.rules import count_matching, is_valid_opening, is_valid_raise
from perudo.core.types import (
    Action,
    Bid,
    Exact,
    GameState,
    Liar,
    RaiseBid,
)
from perudo.m1 import expected_count, p_per_die
from perudo.m2 import RecommenderConfig, enumerate_valid_raises, recommend


class Strategy(ABC):
    """Abstract base for all Perudo strategies."""

    @abstractmethod
    def choose_action(
        self,
        game_state: GameState,
        rng: np.random.Generator,
    ) -> Action:
        """Return the chosen action for the current player."""

    def wants_percolateur(self, game_state: GameState) -> bool:
        """Return True to trigger Percolateur when eligible (1 die, opening)."""
        return True  # default: always trigger

    @property
    def name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# RandomLegal
# ---------------------------------------------------------------------------


class RandomLegal(Strategy):
    """
    Uniformly choose among legal actions.

    Actions enumerated: one raise candidate per face value (min-q) + Liar
    + Exact (if available). Each has equal probability of being selected.
    """

    def choose_action(self, game_state: GameState, rng: np.random.Generator) -> Action:
        prev = game_state.round.current_bid
        pid = game_state.round.current_player_id
        total = sum(p.dice_count for p in game_state.players)
        player = next(p for p in game_state.players if p.id == pid)

        actions: list[Action] = [
            RaiseBid(b) for b in enumerate_valid_raises(prev, pid, total)
        ]
        if prev is not None:
            actions.append(Liar())
            if not player.exact_used:
                actions.append(Exact())

        return actions[int(rng.integers(0, len(actions)))]


# ---------------------------------------------------------------------------
# Honest
# ---------------------------------------------------------------------------


class Honest(Strategy):
    """
    Always bid the rounded expected count for the most-held face value.

    Falls back to Liar if no honest raise is possible over the current bid.
    """

    def choose_action(self, game_state: GameState, rng: np.random.Generator) -> Action:
        prev = game_state.round.current_bid
        pid = game_state.round.current_player_id
        perco = game_state.round.percolateur
        player = next(p for p in game_state.players if p.id == pid)
        own = player.dice
        n_unk = sum(p.dice_count for p in game_state.players if p.id != pid)

        best_bid: Bid | None = None
        best_exp = -1.0

        for v in range(1, 7):
            p = p_per_die(v, percolateur=perco)
            own_count = count_matching(own, v, joker_active=not perco)
            exp = expected_count(n_unk, p, own_count)
            q = max(1, round(exp))
            candidate = Bid(q, v, pid)

            valid = (
                is_valid_opening(candidate)
                if prev is None
                else is_valid_raise(candidate, prev)
            )
            if valid and exp > best_exp:
                best_exp = exp
                best_bid = candidate

        if best_bid is not None:
            return RaiseBid(best_bid)

        # No honest raise over current bid → call liar
        if prev is not None:
            return Liar()

        # Fallback opening (shouldn't happen)
        return RaiseBid(Bid(1, 3, pid))

    def wants_percolateur(self, game_state: GameState) -> bool:
        return False  # Honest never triggers: jokers help honest bidders


# ---------------------------------------------------------------------------
# ThresholdBot
# ---------------------------------------------------------------------------


class ThresholdBot(Strategy):
    """
    Use M2 recommender with configurable thresholds.

    Default thresholds follow the game-theory calibration in RecommenderConfig.
    """

    def __init__(self, config: RecommenderConfig | None = None) -> None:
        self.config = config

    def choose_action(self, game_state: GameState, rng: np.random.Generator) -> Action:
        rec = recommend(game_state, self.config)
        if rec.best_action == "raise":
            assert rec.bid_if_raise is not None
            return RaiseBid(rec.bid_if_raise)
        if rec.best_action == "exact":
            return Exact()
        return Liar()

    @property
    def name(self) -> str:
        if self.config is None:
            return "ThresholdBot"
        return (
            f"ThresholdBot(l={self.config.threshold_liar:.2f}"
            f",e={self.config.threshold_exact:.2f})"
        )
