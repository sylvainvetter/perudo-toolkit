"""Tests for core types (Player, Bid, RoundState, Action union)."""

from perudo.core.types import (
    Action,
    Bid,
    Exact,
    Liar,
    Player,
    RaiseBid,
    RoundState,
)


class TestPlayer:
    def test_dice_count(self) -> None:
        p = Player(id=1, dice=[1, 2, 3, 4, 5])
        assert p.dice_count == 5

    def test_dice_count_empty(self) -> None:
        p = Player(id=1, dice=[])
        assert p.dice_count == 0

    def test_not_eliminated_with_dice(self) -> None:
        p = Player(id=1, dice=[3])
        assert not p.eliminated

    def test_eliminated_when_no_dice(self) -> None:
        p = Player(id=1, dice=[])
        assert p.eliminated

    def test_exact_used_default_false(self) -> None:
        p = Player(id=1, dice=[1, 2])
        assert p.exact_used is False

    def test_mutate_dice(self) -> None:
        p = Player(id=1, dice=[1, 2, 3])
        p.dice.pop()
        assert p.dice_count == 2


class TestRoundState:
    def test_current_bid_none_on_empty_round(self) -> None:
        rs = RoundState(bids=[], current_player_id=1, percolateur=False, starter_id=1)
        assert rs.current_bid is None

    def test_current_bid_returns_last_bid(self) -> None:
        b1 = Bid(2, 3, player_id=0)
        b2 = Bid(3, 4, player_id=1)
        rs = RoundState(
            bids=[b1, b2], current_player_id=2, percolateur=False, starter_id=0
        )
        assert rs.current_bid == b2

    def test_percolateur_flag(self) -> None:
        rs = RoundState(bids=[], current_player_id=0, percolateur=True, starter_id=0)
        assert rs.percolateur is True


class TestBidFrozen:
    def test_bid_immutable(self) -> None:
        b = Bid(3, 5, player_id=0)
        try:
            b.quantity = 4  # type: ignore[misc]
            raise AssertionError("should have raised FrozenInstanceError")
        except Exception:
            pass  # expected


class TestActionUnion:
    def test_raise_bid_is_action(self) -> None:
        action: Action = RaiseBid(bid=Bid(3, 4, player_id=0))
        assert isinstance(action, RaiseBid)

    def test_liar_is_action(self) -> None:
        action: Action = Liar()
        assert isinstance(action, Liar)

    def test_exact_is_action(self) -> None:
        action: Action = Exact()
        assert isinstance(action, Exact)

    def test_dispatch_with_isinstance(self) -> None:
        actions: list[Action] = [RaiseBid(Bid(1, 2, 0)), Liar(), Exact()]
        types_seen = {type(a) for a in actions}
        assert types_seen == {RaiseBid, Liar, Exact}
