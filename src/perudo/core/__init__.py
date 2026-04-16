"""Public API for the Perudo core engine (types + rules)."""

from .rules import (
    ResolutionResult,
    count_matching,
    is_valid_bid,
    is_valid_opening,
    is_valid_raise,
    resolve_exact,
    resolve_liar,
)
from .types import (
    Action,
    Bid,
    Exact,
    GameState,
    Liar,
    Player,
    RaiseBid,
    RoundState,
)

__all__ = [
    # Types
    "Action",
    "Bid",
    "Exact",
    "GameState",
    "Liar",
    "Player",
    "RaiseBid",
    "RoundState",
    # Rules
    "ResolutionResult",
    "count_matching",
    "is_valid_bid",
    "is_valid_opening",
    "is_valid_raise",
    "resolve_exact",
    "resolve_liar",
]
