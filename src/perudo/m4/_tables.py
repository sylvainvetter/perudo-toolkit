"""
M4 — Pre-computed lookup tables for the CFR hot path.

Replaces scipy.stats.binom calls and _min_q_raise logic with O(1) array
lookups, eliminating the Python function-call overhead on every decision.

Tables built once at module import (< 5 ms).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom as _binom

# ---------------------------------------------------------------------------
# Binomial CDF tables
# Max total dice = 6 players × 5 dice = 30
# ---------------------------------------------------------------------------

_MAX_N = 30          # max n_unknown
_P0 = 1.0 / 6.0     # p per die for value==1 or perco
_P1 = 2.0 / 6.0     # p per die for value in 2..6, no perco

# Shape: (max_n+1, max_n+2)
# [n, k] = P(Binom(n, p) > k)  — indexed by (n_unk, needed-1)
# k ranges -1..n  →  we store k+1 in axis-1  (k=-1 → idx 0, k=n → idx n+1)
_ns = np.arange(_MAX_N + 1)
_ks = np.arange(_MAX_N + 2) - 1     # -1 .. _MAX_N

# SF[p_idx][n, k+1] = P(X > k)  with X ~ Binom(n, p)
_SF = np.zeros((2, _MAX_N + 1, _MAX_N + 2))
_PMF = np.zeros((2, _MAX_N + 1, _MAX_N + 2))

for _pi, _p in enumerate([_P0, _P1]):
    for _n in range(_MAX_N + 1):
        for _k in range(-1, _MAX_N + 1):
            _SF[_pi, _n, _k + 1] = float(_binom.sf(_k, _n, _p))
            _PMF[_pi, _n, _k + 1] = float(_binom.pmf(_k, _n, _p))


def binom_sf(k: int, n: int, p_idx: int) -> float:
    """P(X > k) for X ~ Binom(n, p) using pre-computed table."""
    n = min(n, _MAX_N)
    k = max(-1, min(k, _MAX_N))
    return float(_SF[p_idx, n, k + 1])


def binom_pmf(k: int, n: int, p_idx: int) -> float:
    """P(X == k) for X ~ Binom(n, p) using pre-computed table."""
    n = min(n, _MAX_N)
    k = max(-1, min(k, _MAX_N))
    return float(_PMF[p_idx, n, k + 1])


# ---------------------------------------------------------------------------
# min_q_raise table
# Shape: (max_q+1, 7, 7)  — indexed by (bid_q, bid_v, new_v)
# ---------------------------------------------------------------------------

_MAX_Q = _MAX_N  # max quantity ever bid


def _compute_min_q(bid_q: int, bid_v: int, new_v: int) -> int:
    if bid_q == 0:
        return 1
    pq, pv = bid_q, bid_v
    if pv == 1:
        return pq + 1 if new_v == 1 else 2 * pq + 1
    if new_v == 1:
        return math.ceil(pq / 2) + 1
    if new_v > pv:
        return pq       # same quantity OK with higher value (Q-G)
    if new_v == pv:
        return pq + 1   # strictly more required for same value
    return 9999         # new_v < pv: illegal — value can never decrease


_MIN_Q = np.zeros((_MAX_Q + 1, 7, 7), dtype=np.int32)
for _bq in range(_MAX_Q + 1):
    for _bv in range(1, 7):
        for _nv in range(1, 7):
            _MIN_Q[_bq, _bv, _nv] = _compute_min_q(_bq, _bv, _nv)


def min_q_table(bid_q: int, bid_v: int, new_v: int) -> int:
    """Minimum legal quantity for raise — O(1) table lookup."""
    if bid_q > _MAX_Q:
        return bid_q + 1  # safety fallback for unusually large bids
    return int(_MIN_Q[bid_q, bid_v, new_v])


# ---------------------------------------------------------------------------
# Vectorized own_counts from dice face-value counts
# ---------------------------------------------------------------------------

def own_counts_from_faces(face_counts: np.ndarray, joker: bool) -> np.ndarray:
    """
    Compute own contribution for each face value 1..6 given pre-counted faces.

    Args:
        face_counts : int array of shape (6,), face_counts[i] = count of (i+1)s
        joker       : True in normal play (ones count as jokers for v=2..6)

    Returns:
        int array of shape (6,), own_counts[i] = own contribution for value i+1
    """
    if joker:
        result = face_counts.copy()
        result[1:] += face_counts[0]   # joker ones added to v=2..6
        return result
    return face_counts.copy()
