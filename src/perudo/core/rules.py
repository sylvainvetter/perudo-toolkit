"""
Perudo game engine: bid validation and round resolution.

All raise rules are identical in normal play and Percolateur mode.
The *joker_active* flag only affects die *counting*, not which raises are legal.

Raise validity rules (section 2.2 + Q-G):
  prev Standard → new Standard : q' > q  OR  (q' == q AND v' > v)
  prev Standard → new Perudo   : q' >= ceil(q / 2) + 1
  prev Perudo   → new Perudo   : q' > q
  prev Perudo   → new Standard : q' >= 2 * q + 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Bid

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def count_matching(dice: list[int], value: int, *, joker_active: bool) -> int:
    """
    Count dice that contribute to an announcement of *value*.

    joker_active=True (normal play, value != 1): dice showing 1 are jokers
      and count as *value*.
    joker_active=False (Percolateur, or value == 1): only exact matches count.
    """
    exact = sum(1 for d in dice if d == value)
    if joker_active and value != 1:
        return exact + sum(1 for d in dice if d == 1)
    return exact


# ---------------------------------------------------------------------------
# Bid validation
# ---------------------------------------------------------------------------


def is_valid_bid(bid: Bid) -> bool:
    """Structural validity: quantity >= 1 and value in 1..6."""
    return bid.quantity >= 1 and 1 <= bid.value <= 6


def is_valid_opening(bid: Bid) -> bool:
    """Any structurally valid bid may open a round (including Perudo / value=1)."""
    return is_valid_bid(bid)


def is_valid_raise(new_bid: Bid, prev_bid: Bid) -> bool:
    """
    Return True if *new_bid* is a legal raise over *prev_bid*.

    See module docstring for the full rule table.
    """
    if not is_valid_bid(new_bid):
        return False

    q, v = new_bid.quantity, new_bid.value
    pq, pv = prev_bid.quantity, prev_bid.value

    if pv == 1:  # previous bid was Perudo (value=1)
        if v == 1:  # Perudo → Perudo
            return q > pq
        else:  # Perudo → Standard
            return q >= 2 * pq + 1
    else:  # previous bid was Standard (value 2..6)
        if v == 1:  # Standard → Perudo
            return q >= math.ceil(pq / 2) + 1
        else:  # Standard → Standard
            return q > pq or (q == pq and v > pv)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of a Liar or Exact challenge."""

    total_matching: int
    bid_was_true: bool  # T >= q  (challenger was wrong / exact caller may be right)
    bid_was_exact: bool  # T == q
    loser_id: int | None  # player who loses 1 die; None when Exact succeeds


def resolve_liar(
    current_bid: Bid,
    challenger_id: int,
    all_dice: list[int],
    *,
    percolateur: bool,
) -> ResolutionResult:
    """
    Resolve a Liar challenge against *current_bid*.

    T >= q → challenger was wrong → challenger loses 1 die.
    T <  q → bidder was lying     → bidder loses 1 die.

    *all_dice* must be the flat list of every die on the table (all players).
    """
    total = count_matching(all_dice, current_bid.value, joker_active=not percolateur)
    bid_was_true = total >= current_bid.quantity
    return ResolutionResult(
        total_matching=total,
        bid_was_true=bid_was_true,
        bid_was_exact=(total == current_bid.quantity),
        loser_id=challenger_id if bid_was_true else current_bid.player_id,
    )


def resolve_exact(
    current_bid: Bid,
    caller_id: int,
    all_dice: list[int],
    *,
    percolateur: bool,
) -> ResolutionResult:
    """
    Resolve an Exact claim on *current_bid*.

    T == q → caller gains 1 die (max 5, handled by game engine); loser_id = None.
    T != q → caller loses 1 die; loser_id = caller_id.

    *all_dice* must be the flat list of every die on the table (all players).
    """
    total = count_matching(all_dice, current_bid.value, joker_active=not percolateur)
    bid_was_exact = total == current_bid.quantity
    return ResolutionResult(
        total_matching=total,
        bid_was_true=(total >= current_bid.quantity),
        bid_was_exact=bid_was_exact,
        loser_id=None if bid_was_exact else caller_id,
    )
