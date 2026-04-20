"""
M2 — Action recommender for Perudo.

Decision logic v1 (thresholds calibrable via M3 — S7/S8):
  1. Compute p_true  = P(current bid is satisfied  | own dice)
  2. Compute p_exact = P(current bid is exact       | own dice)
  3. p_true  < cfg.threshold_liar              → recommend Liar
  4. p_exact > cfg.threshold_exact AND exact still available → recommend Exact
  5. Otherwise enumerate valid raises, score each, recommend the best Raise

The score for a raise candidate is:
    score = p_true(raised_bid) − lambda_risk × max(0, q − E[T])

With lambda_risk = 0 (default), score = p_true: the most probable raise wins.

Adaptive thresholds (S8 multi-player calibration, 81 configs × 500 games each):
  n_players=3 → liar=0.30, exact=0.45   (win rate 67.0%)
  n_players=4 → liar=0.35, exact=0.40   (win rate 65.6%)
  n_players=5 → liar=0.30, exact=0.50   (win rate 64.4%)
  n_players=6 → liar=0.30, exact=0.30   (win rate 61.8%)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from perudo.core.types import Bid, GameState, Player
from perudo.m1 import BidStats, bid_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommenderConfig:
    """
    Tunable decision thresholds.

    Defaults calibrated by S7 grid-search (81 configs × 500 games vs [Honest, Honest],
    validated at 2 000 games, confirmed across 3 opponent scenarios):
    - threshold_liar = 0.30: challenge aggressively — wins +15 pp over the Nash
      break-even (0.50) against Honest-style opponents who bid rounded expectations.
      Robust across mixed and symmetric ThresholdBot fields (see docs/steps/STEP_S7.md).
    - threshold_exact = 0.40: claim exact when highly concentrated distribution
      (conservative — exact is one-shot and losing costs a die).
    - lambda_risk = 0.0: no risk penalty by default.
    """

    threshold_liar: float = 0.30
    """Recommend Liar when P(current bid true) < this value. Calibrated at 0.30 (S7)."""

    threshold_exact: float = 0.40
    """Recommend Exact when P(current bid exact) > this value and exact is available."""

    lambda_risk: float = 0.0
    """Penalty weight on (q − E[T]) in the raise score. 0 = pure probability."""


_DEFAULT_CONFIG = RecommenderConfig()

# ---------------------------------------------------------------------------
# Adaptive thresholds — calibrated per player count (S8)
# ---------------------------------------------------------------------------

_ADAPTIVE_CONFIGS: dict[int, RecommenderConfig] = {
    3: RecommenderConfig(threshold_liar=0.30, threshold_exact=0.45),
    4: RecommenderConfig(threshold_liar=0.35, threshold_exact=0.40),
    5: RecommenderConfig(threshold_liar=0.30, threshold_exact=0.50),
    6: RecommenderConfig(threshold_liar=0.30, threshold_exact=0.30),
}


def config_for_n_players(n_players: int) -> RecommenderConfig:
    """Return the calibrated config for the given player count.

    Falls back to the closest calibrated value for n < 3 or n > 6.
    """
    if n_players <= 3:
        return _ADAPTIVE_CONFIGS[3]
    if n_players >= 6:
        return _ADAPTIVE_CONFIGS[6]
    return _ADAPTIVE_CONFIGS.get(n_players, _DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaiseCandidate:
    """A legal raise option with its probability stats and composite score."""

    bid: Bid
    stats: BidStats
    score: float


@dataclass
class Rationale:
    """Numerical justification attached to every recommendation."""

    p_current_bid_true: float  # P(T >= q) for the bid currently on the table
    p_current_bid_exact: float  # P(T == q) for the bid currently on the table
    expected_total: float  # E[T] given own dice and unknown count
    alternatives_considered: list[RaiseCandidate]
    score: float  # decision score for the chosen action


@dataclass(frozen=True)
class Recommendation:
    """The output of the M2 recommender."""

    best_action: Literal["raise", "liar", "exact"]
    bid_if_raise: Bid | None  # non-None iff best_action == "raise"
    rationale: Rationale


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_player(game_state: GameState) -> Player:
    pid = game_state.round.current_player_id
    for p in game_state.players:
        if p.id == pid:
            return p
    raise ValueError(f"Current player id={pid} not found in game_state.players")


def _n_unknown(game_state: GameState) -> int:
    pid = game_state.round.current_player_id
    return sum(p.dice_count for p in game_state.players if p.id != pid)


def _total_dice(game_state: GameState) -> int:
    return sum(p.dice_count for p in game_state.players)


def _min_q_for_value(prev_bid: Bid, v: int) -> int:
    """Minimum valid quantity when raising to face value *v* over *prev_bid*."""
    pq, pv = prev_bid.quantity, prev_bid.value
    if pv == 1:
        return pq + 1 if v == 1 else 2 * pq + 1
    if v == 1:
        return math.ceil(pq / 2) + 1
    # Standard → Standard: value can never decrease (Q-G)
    if v > pv:
        return pq       # same quantity OK with higher value (Q-G)
    if v == pv:
        return pq + 1   # strictly more required for same value
    return 9999         # v < pv: illegal — sentinel > any realistic total_dice


def enumerate_valid_raises(
    prev_bid: Bid | None,
    player_id: int,
    total_dice: int,
) -> list[Bid]:
    """
    Return one raise candidate per face value (the minimum valid quantity for each).

    Using the minimum q per value maximises P(bid true), making these the best
    candidates under the default score function.

    For an opening bid (prev_bid=None) returns Bid(1, v, player_id) for each v.
    """
    if prev_bid is None:
        return [Bid(1, v, player_id) for v in range(1, 7)]

    candidates: list[Bid] = []
    for v in range(1, 7):
        min_q = _min_q_for_value(prev_bid, v)
        if min_q <= total_dice:
            candidates.append(Bid(min_q, v, player_id))
    return candidates


def _score(bid: Bid, stats: BidStats, lambda_risk: float) -> float:
    """Score for a raise candidate: p_true penalised by bid excess over E[T]."""
    distance = max(0.0, bid.quantity - stats.expected)
    return stats.p_true - lambda_risk * distance


def _build_candidates(
    player_id: int,
    total_dice: int,
    prev_bid: Bid | None,
    own_dice: list[int],
    n_unk: int,
    percolateur: bool,
    lambda_risk: float,
) -> list[RaiseCandidate]:
    return [
        RaiseCandidate(
            bid=bid,
            stats=(s := bid_stats(bid, own_dice, n_unk, percolateur=percolateur)),
            score=_score(bid, s, lambda_risk),
        )
        for bid in enumerate_valid_raises(prev_bid, player_id, total_dice)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend(
    game_state: GameState,
    config: RecommenderConfig | None = None,
) -> Recommendation:
    """
    Recommend the best action for the current player.

    Reads own dice from game_state.players[current_player_id].dice.
    Opponent dice counts are read from game_state but their values are
    treated as unknown (only count matters for probability estimates).

    Args:
        game_state: Full game state snapshot.
        config:     Decision thresholds; uses defaults when None.

    Returns:
        Recommendation with best_action, optional bid_if_raise, and rationale.
    """
    player = _get_player(game_state)
    own = player.dice
    n_unk = _n_unknown(game_state)
    total = _total_dice(game_state)
    n_players = len(game_state.players)
    cfg = config if config is not None else config_for_n_players(n_players)
    prev = game_state.round.current_bid
    perco = game_state.round.percolateur
    pid = game_state.round.current_player_id

    candidates = _build_candidates(pid, total, prev, own, n_unk, perco, cfg.lambda_risk)

    # ── Opening bid: only raise is possible ──────────────────────────────────
    if prev is None:
        # Tie-break by own_count (announce what you hold)
        best = (
            max(candidates, key=lambda c: (c.score, c.stats.own_count))
            if candidates
            else None
        )
        return Recommendation(
            best_action="raise",
            bid_if_raise=best.bid if best else None,
            rationale=Rationale(
                p_current_bid_true=0.0,
                p_current_bid_exact=0.0,
                expected_total=0.0,
                alternatives_considered=candidates,
                score=best.score if best else 0.0,
            ),
        )

    cur = bid_stats(prev, own, n_unk, percolateur=perco)

    # ── Liar ─────────────────────────────────────────────────────────────────
    if cur.p_true < cfg.threshold_liar:
        return Recommendation(
            best_action="liar",
            bid_if_raise=None,
            rationale=Rationale(
                p_current_bid_true=cur.p_true,
                p_current_bid_exact=cur.p_exact,
                expected_total=cur.expected,
                alternatives_considered=candidates,
                score=1.0 - cur.p_true,
            ),
        )

    # ── Exact ─────────────────────────────────────────────────────────────────
    if cur.p_exact > cfg.threshold_exact and not player.exact_used:
        return Recommendation(
            best_action="exact",
            bid_if_raise=None,
            rationale=Rationale(
                p_current_bid_true=cur.p_true,
                p_current_bid_exact=cur.p_exact,
                expected_total=cur.expected,
                alternatives_considered=candidates,
                score=cur.p_exact,
            ),
        )

    # ── Raise ─────────────────────────────────────────────────────────────────
    best_raise = max(candidates, key=lambda c: c.score) if candidates else None
    if best_raise is None:
        # No valid raise available: fall back to Liar
        return Recommendation(
            best_action="liar",
            bid_if_raise=None,
            rationale=Rationale(
                p_current_bid_true=cur.p_true,
                p_current_bid_exact=cur.p_exact,
                expected_total=cur.expected,
                alternatives_considered=candidates,
                score=1.0 - cur.p_true,
            ),
        )

    return Recommendation(
        best_action="raise",
        bid_if_raise=best_raise.bid,
        rationale=Rationale(
            p_current_bid_true=cur.p_true,
            p_current_bid_exact=cur.p_exact,
            expected_total=cur.expected,
            alternatives_considered=candidates,
            score=best_raise.score,
        ),
    )
