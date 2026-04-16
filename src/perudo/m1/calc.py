"""
M1 — Probability calculator for Perudo bids.

All calculations model the total count T = own_count + X, where:
  - own_count  is deterministic (player's visible dice)
  - X ~ Binomial(n_unknown, p_die) models the unknown dice

p_die depends on the announced value and whether Percolateur is active:
  - Normal play, value != 1 : p = 2/6  (exact match 1/6 + joker 1/6)
  - Normal play, value == 1 : p = 1/6  (only literal 1s count)
  - Percolateur,  any value : p = 1/6  (jokers disabled)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

from perudo.core.rules import count_matching
from perudo.core.types import Bid

# ---------------------------------------------------------------------------
# Per-die probability
# ---------------------------------------------------------------------------


def p_per_die(value: int, *, percolateur: bool) -> float:
    """
    Probability that a single unknown die contributes to an announcement of *value*.

    Returns 2/6 in normal play for value in 2..6 (die matches or is a joker),
    1/6 otherwise (Percolateur active, or value == 1).
    """
    if percolateur or value == 1:
        return 1.0 / 6.0
    return 2.0 / 6.0


# ---------------------------------------------------------------------------
# Low-level probability functions
# ---------------------------------------------------------------------------


def p_at_least(q: int, n: int, p: float, own_count: int) -> float:
    """
    P(T >= q) where T = own_count + Binomial(n, p).

    Returns 1.0 if own_count alone already satisfies the bid (own_count >= q).
    Returns 0.0 if the bid needs more dice than the total possible (q > n + own_count).
    """
    needed = q - own_count
    if needed <= 0:
        return 1.0
    if needed > n:
        return 0.0
    # P(X >= needed) = sf(needed - 1, n, p)  where X ~ Binom(n, p)
    return float(binom.sf(needed - 1, n, p))


def p_exactly(q: int, n: int, p: float, own_count: int) -> float:
    """
    P(T == q) where T = own_count + Binomial(n, p).

    Returns 0.0 if q < own_count or q > n + own_count.
    """
    needed = q - own_count
    if needed < 0 or needed > n:
        return 0.0
    return float(binom.pmf(needed, n, p))


def expected_count(n: int, p: float, own_count: int) -> float:
    """E[T] = own_count + n * p."""
    return own_count + n * p


def distribution(n: int, p: float, own_count: int) -> NDArray[np.float64]:
    """
    Full probability mass function of T = own_count + Binomial(n, p).

    Returns a 1-D array *dist* of length (n + own_count + 1) where
    dist[k] = P(T == k) for k in 0 .. n + own_count.

    The array sums to 1.0 (within floating-point precision).
    """
    total_max = n + own_count
    result = np.zeros(total_max + 1)
    if n == 0:
        # Deterministic: T = own_count with probability 1
        result[own_count] = 1.0
        return result
    pmf_values = binom.pmf(np.arange(n + 1), n, p)
    # T = own_count + k for k in 0..n  →  indices own_count..own_count+n
    result[own_count : own_count + n + 1] = pmf_values
    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BidStats:
    """Probability statistics for a bid given a player's hand."""

    p_true: float  # P(T >= q) — probability the bid is at least satisfied
    p_exact: float  # P(T == q) — probability the bid is exactly satisfied
    expected: float  # E[T]      — expected total count
    own_count: int  # deterministic contribution from the player's own dice
    n_unknown: int  # number of hidden opponent dice
    p_die: float  # probability each hidden die contributes to the count


def bid_stats(
    bid: Bid,
    own_dice: list[int],
    n_unknown: int,
    *,
    percolateur: bool,
) -> BidStats:
    """
    Compute probability statistics for *bid* given the player's hand.

    Args:
        bid:        The (quantity, value) announcement to evaluate.
        own_dice:   The player's current dice values.
        n_unknown:  Number of dice held by opponents (hidden).
        percolateur: True if the Percolateur rule is active (no jokers).

    Returns:
        BidStats with P(true), P(exact), E[total], and auxiliary fields.
    """
    joker_active = not percolateur
    own = count_matching(own_dice, bid.value, joker_active=joker_active)
    p = p_per_die(bid.value, percolateur=percolateur)
    return BidStats(
        p_true=p_at_least(bid.quantity, n_unknown, p, own),
        p_exact=p_exactly(bid.quantity, n_unknown, p, own),
        expected=expected_count(n_unknown, p, own),
        own_count=own,
        n_unknown=n_unknown,
        p_die=p,
    )
