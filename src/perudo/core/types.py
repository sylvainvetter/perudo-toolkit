"""Core data types for the Perudo game engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bid:
    """An announced (quantity, value) pair made by a player."""

    quantity: int  # >= 1
    value: int  # 1..6  (1 = Perudo / joker face)
    player_id: int


@dataclass(frozen=True)
class RaiseBid:
    """Action: raise the current bid to a new Bid."""

    bid: Bid


@dataclass(frozen=True)
class Liar:
    """Action: challenge the current bid (contester — « menteur »)."""


@dataclass(frozen=True)
class Exact:
    """Action: claim the count is exactly right (one-shot per game per player)."""


# Discriminated union — dispatch with isinstance()
type Action = RaiseBid | Liar | Exact


@dataclass
class Player:
    """One participant in the game."""

    id: int
    dice: list[int]  # current die values; length == dice_count
    exact_used: bool = False  # True once the player has spent their Exact action

    @property
    def dice_count(self) -> int:
        return len(self.dice)

    @property
    def eliminated(self) -> bool:
        return len(self.dice) == 0


@dataclass
class RoundState:
    """State of the current bidding round."""

    bids: list[Bid]  # chronological list of bids this round
    current_player_id: int  # whose turn it is to act
    percolateur: bool  # True → Percolateur rule active (no jokers)
    starter_id: int  # player who opened this round

    @property
    def current_bid(self) -> Bid | None:
        """Last bid placed, or None if the round just started."""
        return self.bids[-1] if self.bids else None


@dataclass
class GameState:
    """Complete game state snapshot."""

    players: list[Player]
    round: RoundState
    turn_order: list[int]  # IDs of active (non-eliminated) players, in order
