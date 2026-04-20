"""
M4 — Compact information-state representation using probability buckets.

Instead of storing exact die values (which creates millions of states), we
abstract the information state to probability buckets computed from the
current bid and own dice.  This reduces the state space from ~100k to ~1800,
enabling convergence with 10k-50k training episodes.

Info key (for non-opening states only, i.e. bid_q > 0):
  p_true_bucket  : P(bid is true  | own dice) → 10 levels (0-9, each 10 pp)
  p_exact_bucket : P(bid is exact | own dice) → 6  levels (0-5, each 10 pp)
  n_dice         : own dice count              → 5  values (1-5)
  n_active       : active players at table     → 5  values (2-6)
  exact_avail    : Exact still usable          → bool
  perco          : Percolateur active          → bool
  round_phase    : bids so far this round      → 3  values (1 / 2-3 / 4+)

Total: 10 × 6 × 5 × 5 × 2 × 2 × 3 = 18 000 states.
In practice perco=False always during training → ~9 000 reachable states.

n_active (number of players still alive) replaces the old total_bucket proxy
(>15 / 8-15 / ≤7 dice total). It is a direct, more expressive signal:
head-to-head (2p) vs full table (6p) demand radically different strategies.

Opening bids (bid_q == 0) use the full face distribution so the agent can
learn value-specific opening strategies.

Action indices (unchanged):
  0  Liar
  1  Exact
  2  Raise v=1 … 7  Raise v=6
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from perudo.core.types import Bid, Exact, Liar, RaiseBid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ACTIONS = 8
ACTION_LIAR = 0
ACTION_EXACT = 1
ACTION_RAISE_BASE = 2  # index for v=1; v=k → index k+1


# ---------------------------------------------------------------------------
# Info key (bucketed probabilities)
# ---------------------------------------------------------------------------


def make_opening_key(
    face_counts: np.ndarray,
    n_active: int,
    perco: bool,
) -> tuple:
    """
    Build an info-state key for opening bids (bid_q == 0).

    Captures the full face distribution so the agent can learn value-specific
    opening strategies (e.g. "I have three 4s → bid (1,4)").

    Key: ("open", face_counts_6tuple, n_active, perco)
    Prefix "open" ensures no collision with the 7-int non-opening keys.

    n_active: number of players still alive (2-6).

    State count: ≤ 462 face-count vectors × 5 player counts × 2 perco = 4 620
    In practice perco=False always during training → ≤ 2 310 reachable.
    """
    return ("open", tuple(map(int, face_counts)), n_active, perco)


def make_info_key(
    own_dice: list[int],
    bid_q: int,
    bid_v: int,
    total_dice: int,
    exact_avail: bool,
    perco: bool,
    n_bids: int = 1,
    n_active: int = 4,
) -> tuple[Any, ...]:
    """
    Build a compact hashable info-state key for a non-opening decision.

    Computes P(bid true) and P(bid exact) from M1 and buckets them.
    Should only be called when bid_q > 0 (there is a bid on the table).

    Args:
        n_bids  : number of bids placed this round so far (bucketed to
                  round_phase: 1=fresh / 2-3=escalating / 4+=pressure).
        n_active: number of players still alive (2-6). Replaces the old
                  total_bucket proxy — more expressive and directly available.
    """
    from perudo.m4._tables import binom_pmf, binom_sf, own_counts_from_faces

    n_unk = max(0, total_dice - len(own_dice))
    joker = not perco
    p_idx = 0 if (perco or bid_v == 1) else 1

    face_counts = np.bincount(np.array(own_dice, dtype=np.int32),
                               minlength=7)[1:].astype(np.int32)
    own_counts = own_counts_from_faces(face_counts, joker)
    own_cur = int(own_counts[bid_v - 1])
    needed = bid_q - own_cur

    p_true = binom_sf(needed - 1, n_unk, p_idx)
    p_exact = binom_pmf(needed, n_unk, p_idx)

    p_true_bucket = min(9, int(p_true * 10))   # 0..9  (each bucket = 10 pp)
    p_exact_bucket = min(5, int(p_exact * 10))  # 0..5  (each bucket = 10 pp)
    n_dice = len(own_dice)                       # 1..5
    # 0=fresh (1 bid) / 1=escalating (2-3) / 2=pressure (4+)
    round_phase = 0 if n_bids == 1 else (1 if n_bids <= 3 else 2)

    return (
        p_true_bucket, p_exact_bucket, n_dice,
        n_active, exact_avail, perco, round_phase,
    )


# ---------------------------------------------------------------------------
# Action helpers (unchanged — still need bid_q / bid_v for legal raises)
# ---------------------------------------------------------------------------


def min_q_raise(bid_q: int, bid_v: int, new_v: int) -> int:
    """Minimum legal quantity when raising to new_v over (bid_q, bid_v)."""
    if bid_q == 0:
        return 1
    pq, pv = bid_q, bid_v
    if pv == 1:
        return pq + 1 if new_v == 1 else 2 * pq + 1
    if new_v == 1:
        return math.ceil(pq / 2) + 1
    return pq if new_v > pv else pq + 1


def legal_mask(
    bid_q: int,
    bid_v: int,
    total_dice: int,
    exact_avail: bool,
) -> np.ndarray:
    """Boolean mask of length N_ACTIONS — True where the action is legal."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if bid_q > 0:
        mask[ACTION_LIAR] = True
        if exact_avail:
            mask[ACTION_EXACT] = True
    for v in range(1, 7):
        if min_q_raise(bid_q, bid_v, v) <= total_dice:
            mask[ACTION_RAISE_BASE + v - 1] = True
    return mask


def decode_action(
    idx: int,
    bid_q: int,
    bid_v: int,
    player_id: int,
) -> Liar | Exact | RaiseBid:
    """Convert an action index to the corresponding game action."""
    if idx == ACTION_LIAR:
        return Liar()
    if idx == ACTION_EXACT:
        return Exact()
    v = idx - ACTION_RAISE_BASE + 1  # 2→v=1, 3→v=2, …, 7→v=6
    min_q = min_q_raise(bid_q, bid_v, v)
    return RaiseBid(Bid(quantity=min_q, value=v, player_id=player_id))
