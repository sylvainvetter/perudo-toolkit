"""M1 — Probability calculator for Perudo bids."""

from .calc import (
    BidStats,
    bid_stats,
    distribution,
    expected_count,
    p_at_least,
    p_exactly,
    p_per_die,
)

__all__ = [
    "BidStats",
    "bid_stats",
    "distribution",
    "expected_count",
    "p_at_least",
    "p_exactly",
    "p_per_die",
]
