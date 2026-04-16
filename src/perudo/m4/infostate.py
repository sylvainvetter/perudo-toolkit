"""
M4 — Information-state representation and action encoding for CFR.

An information-state key captures everything the current player knows:
  - own_dice     : sorted tuple of the player's own die values
  - bid_q, bid_v : current bid quantity and face value (0, 0 = opening)
  - total_dice   : total dice on the table (game-phase proxy)
  - exact_avail  : True if the player hasn't spent their Exact action
  - percolateur  : True if the Percolateur rule is active this round

Action indices
  0  Liar      legal when there is a bid on the table
  1  Exact     legal when bid on table AND exact not yet used
  2  Raise v=1 raise to minimum legal quantity with face value 1
  3  Raise v=2 ...
  ...
  7  Raise v=6
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
ACTION_RAISE_BASE = 2  # action index for face value 1; value v → index v+1


# ---------------------------------------------------------------------------
# Information-state key
# ---------------------------------------------------------------------------


def make_info_key(
    own_dice: list[int],
    bid_q: int,
    bid_v: int,
    total_dice: int,
    exact_avail: bool,
    percolateur: bool,
) -> tuple[Any, ...]:
    """Return a hashable key representing the current information state."""
    return (tuple(sorted(own_dice)), bid_q, bid_v, total_dice, exact_avail, percolateur)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _min_q_raise(bid_q: int, bid_v: int, new_v: int) -> int:
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
    """Boolean mask of length N_ACTIONS: True where the action is legal."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if bid_q > 0:
        mask[ACTION_LIAR] = True
        if exact_avail:
            mask[ACTION_EXACT] = True
    for v in range(1, 7):
        if _min_q_raise(bid_q, bid_v, v) <= total_dice:
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
    v = idx - ACTION_RAISE_BASE + 1  # 2 → v=1, 3 → v=2, …, 7 → v=6
    min_q = _min_q_raise(bid_q, bid_v, v)
    return RaiseBid(Bid(quantity=min_q, value=v, player_id=player_id))
