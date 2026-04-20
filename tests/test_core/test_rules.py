"""
Tests for the Perudo rule engine.

Covers:
- count_matching (with/without jokers, Percolateur)
- is_valid_bid / is_valid_opening
- is_valid_raise (all four transition types + property tests)
- resolve_liar
- resolve_exact
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from perudo.core.rules import (
    count_matching,
    is_valid_bid,
    is_valid_opening,
    is_valid_raise,
    resolve_exact,
    resolve_liar,
)
from perudo.core.types import Bid

# ---------------------------------------------------------------------------
# count_matching
# ---------------------------------------------------------------------------


class TestCountMatching:
    def test_exact_match_no_jokers(self) -> None:
        assert count_matching([2, 3, 2, 5, 2], 2, joker_active=False) == 3

    def test_jokers_counted_for_non_one(self) -> None:
        # two 3s + one joker (1) → 3
        assert count_matching([3, 3, 1, 4, 5], 3, joker_active=True) == 3

    def test_jokers_not_counted_when_value_is_one(self) -> None:
        # value=1 → only literal 1s count, even with joker_active=True
        assert count_matching([1, 1, 3, 4, 5], 1, joker_active=True) == 2

    def test_percolateur_ignores_jokers(self) -> None:
        # joker_active=False: the 1 does NOT count toward 3s
        assert count_matching([1, 3, 3, 3], 3, joker_active=False) == 3
        assert count_matching([1, 3, 3], 3, joker_active=False) == 2

    def test_empty_dice_returns_zero(self) -> None:
        assert count_matching([], 4, joker_active=True) == 0

    def test_all_jokers(self) -> None:
        # five 1s, asking for value=5 with jokers → all 5 count
        assert count_matching([1, 1, 1, 1, 1], 5, joker_active=True) == 5

    def test_all_jokers_no_joker_active(self) -> None:
        assert count_matching([1, 1, 1, 1, 1], 5, joker_active=False) == 0


@given(
    dice=st.lists(st.integers(min_value=1, max_value=6), max_size=30),
    value=st.integers(min_value=1, max_value=6),
)
def test_count_never_exceeds_total_dice(dice: list[int], value: int) -> None:
    assert 0 <= count_matching(dice, value, joker_active=True) <= len(dice)
    assert 0 <= count_matching(dice, value, joker_active=False) <= len(dice)


@given(
    dice=st.lists(st.integers(min_value=1, max_value=6), max_size=30),
    value=st.integers(min_value=2, max_value=6),
)
def test_joker_active_gte_inactive(dice: list[int], value: int) -> None:
    """With jokers on, count can only be >= count with jokers off."""
    assert count_matching(dice, value, joker_active=True) >= count_matching(
        dice, value, joker_active=False
    )


# ---------------------------------------------------------------------------
# is_valid_bid / is_valid_opening
# ---------------------------------------------------------------------------


class TestIsValidBid:
    @pytest.mark.parametrize("q,v", [(1, 1), (1, 6), (5, 3), (100, 1)])
    def test_valid(self, q: int, v: int) -> None:
        assert is_valid_bid(Bid(q, v, player_id=0))

    @pytest.mark.parametrize("q,v", [(0, 3), (-1, 3), (1, 0), (1, 7), (1, -1)])
    def test_invalid(self, q: int, v: int) -> None:
        assert not is_valid_bid(Bid(q, v, player_id=0))

    def test_opening_delegates_to_valid_bid(self) -> None:
        assert is_valid_opening(Bid(1, 1, player_id=0))
        assert not is_valid_opening(Bid(0, 3, player_id=0))


# ---------------------------------------------------------------------------
# is_valid_raise
# ---------------------------------------------------------------------------


class TestIsValidRaiseStandardToStandard:
    """Standard (v != 1) → Standard (v' != 1)."""

    def test_higher_quantity_same_value(self) -> None:
        assert is_valid_raise(Bid(4, 3, 0), Bid(3, 3, 1))

    def test_much_higher_quantity(self) -> None:
        assert is_valid_raise(Bid(10, 2, 0), Bid(1, 2, 1))

    def test_same_quantity_higher_value(self) -> None:
        # Q-G confirmed: (3, 6) after (3, 5) is valid
        assert is_valid_raise(Bid(3, 6, 0), Bid(3, 5, 1))

    def test_same_quantity_same_value_invalid(self) -> None:
        assert not is_valid_raise(Bid(3, 5, 0), Bid(3, 5, 1))

    def test_lower_quantity_same_value_invalid(self) -> None:
        assert not is_valid_raise(Bid(2, 5, 0), Bid(3, 5, 1))

    def test_lower_quantity_higher_value_invalid(self) -> None:
        # q' < pq is always invalid regardless of value
        assert not is_valid_raise(Bid(2, 6, 0), Bid(3, 5, 1))

    def test_same_quantity_lower_value_invalid(self) -> None:
        assert not is_valid_raise(Bid(3, 4, 0), Bid(3, 5, 1))


class TestIsValidRaiseStandardToPerudo:
    """Standard (v != 1) → Perudo (v' = 1): q' >= ceil(pq / 2) + 1."""

    def test_q5_minimum_is_4(self) -> None:
        # ceil(5/2)+1 = 3+1 = 4
        assert is_valid_raise(Bid(4, 1, 0), Bid(5, 3, 1))
        assert not is_valid_raise(Bid(3, 1, 0), Bid(5, 3, 1))

    def test_q4_minimum_is_3(self) -> None:
        # ceil(4/2)+1 = 2+1 = 3
        assert is_valid_raise(Bid(3, 1, 0), Bid(4, 3, 1))
        assert not is_valid_raise(Bid(2, 1, 0), Bid(4, 3, 1))

    def test_q1_minimum_is_2(self) -> None:
        # ceil(1/2)+1 = 1+1 = 2
        assert is_valid_raise(Bid(2, 1, 0), Bid(1, 4, 1))
        assert not is_valid_raise(Bid(1, 1, 0), Bid(1, 4, 1))

    def test_q6_minimum_is_4(self) -> None:
        # ceil(6/2)+1 = 3+1 = 4
        assert is_valid_raise(Bid(4, 1, 0), Bid(6, 2, 1))
        assert not is_valid_raise(Bid(3, 1, 0), Bid(6, 2, 1))


class TestIsValidRaisePerudoToPerudo:
    """Perudo (v=1) → Perudo (v'=1): q' > pq."""

    def test_higher_quantity(self) -> None:
        assert is_valid_raise(Bid(4, 1, 0), Bid(3, 1, 1))

    def test_same_quantity_invalid(self) -> None:
        assert not is_valid_raise(Bid(3, 1, 0), Bid(3, 1, 1))

    def test_lower_quantity_invalid(self) -> None:
        assert not is_valid_raise(Bid(2, 1, 0), Bid(3, 1, 1))


class TestIsValidRaisePerudoToStandard:
    """Perudo (v=1) → Standard (v'!=1): q' >= 2*pq + 1."""

    def test_pq3_minimum_is_7(self) -> None:
        assert is_valid_raise(Bid(7, 2, 0), Bid(3, 1, 1))
        assert not is_valid_raise(Bid(6, 2, 0), Bid(3, 1, 1))

    def test_pq1_minimum_is_3(self) -> None:
        assert is_valid_raise(Bid(3, 4, 0), Bid(1, 1, 1))
        assert not is_valid_raise(Bid(2, 4, 0), Bid(1, 1, 1))

    def test_pq2_minimum_is_5(self) -> None:
        assert is_valid_raise(Bid(5, 6, 0), Bid(2, 1, 1))
        assert not is_valid_raise(Bid(4, 6, 0), Bid(2, 1, 1))


class TestIsValidRaiseStructural:
    def test_zero_quantity_always_invalid(self) -> None:
        assert not is_valid_raise(Bid(0, 3, 0), Bid(1, 2, 1))

    def test_value_out_of_range_invalid(self) -> None:
        assert not is_valid_raise(Bid(5, 7, 0), Bid(1, 2, 1))
        assert not is_valid_raise(Bid(5, 0, 0), Bid(1, 2, 1))


# Property: Standard → Standard exhaustive check
@given(
    pq=st.integers(min_value=1, max_value=20),
    pv=st.integers(min_value=2, max_value=6),
    q=st.integers(min_value=1, max_value=20),
    v=st.integers(min_value=2, max_value=6),
)
def test_std_to_std_matches_formula(pq: int, pv: int, q: int, v: int) -> None:
    """is_valid_raise for std→std: value can never decrease (Q-G rule).
    v' > v : q' >= q suffices (same quantity OK with higher value)
    v' == v: q' > q required
    v' < v : always illegal
    """
    if v > pv:
        expected = q >= pq
    elif v == pv:
        expected = q > pq
    else:
        expected = False
    assert is_valid_raise(Bid(q, v, 0), Bid(pq, pv, 1)) == expected


# Property: Standard → Perudo exhaustive check
@given(
    pq=st.integers(min_value=1, max_value=20),
    pv=st.integers(min_value=2, max_value=6),
    q=st.integers(min_value=1, max_value=30),
)
def test_std_to_perudo_matches_formula(pq: int, pv: int, q: int) -> None:
    expected = q >= math.ceil(pq / 2) + 1
    assert is_valid_raise(Bid(q, 1, 0), Bid(pq, pv, 1)) == expected


# Property: monotonicity — if (q,v) is valid, then (q+1, v) is also valid
@given(
    pq=st.integers(min_value=1, max_value=15),
    pv=st.integers(min_value=2, max_value=6),
    q=st.integers(min_value=1, max_value=15),
    v=st.integers(min_value=2, max_value=6),
)
def test_std_std_raising_quantity_preserves_validity(
    pq: int, pv: int, q: int, v: int
) -> None:
    prev = Bid(pq, pv, 1)
    if is_valid_raise(Bid(q, v, 0), prev):
        assert is_valid_raise(Bid(q + 1, v, 0), prev)


# ---------------------------------------------------------------------------
# resolve_liar
# ---------------------------------------------------------------------------


class TestResolveLiar:
    def test_bid_true_challenger_loses(self) -> None:
        # bid: 2 threes; total with joker: 3 >= 2 → challenger loses
        result = resolve_liar(
            Bid(2, 3, player_id=1),
            challenger_id=2,
            all_dice=[3, 3, 3],
            percolateur=False,
        )
        assert result.bid_was_true is True
        assert result.loser_id == 2
        assert result.total_matching == 3

    def test_bid_false_bidder_loses(self) -> None:
        # bid: 4 threes; dice: [3,2,4,5] → total=1 < 4 → bidder (1) loses
        result = resolve_liar(
            Bid(4, 3, player_id=1),
            challenger_id=2,
            all_dice=[3, 2, 4, 5],
            percolateur=False,
        )
        assert result.bid_was_true is False
        assert result.loser_id == 1
        assert result.total_matching == 1

    def test_jokers_counted_in_normal_play(self) -> None:
        # bid: 3 fours; dice: [4,4,1] → 2 exact + 1 joker = 3 ≥ 3 → challenger loses
        result = resolve_liar(
            Bid(3, 4, player_id=1),
            challenger_id=2,
            all_dice=[4, 4, 1],
            percolateur=False,
        )
        assert result.bid_was_true is True
        assert result.bid_was_exact is True
        assert result.loser_id == 2

    def test_percolateur_jokers_inactive(self) -> None:
        # Percolateur: [4,4,1] → only 2 fours (1 not a joker) < 3 → bidder loses
        result = resolve_liar(
            Bid(3, 4, player_id=1),
            challenger_id=2,
            all_dice=[4, 4, 1],
            percolateur=True,
        )
        assert result.bid_was_true is False
        assert result.loser_id == 1
        assert result.total_matching == 2

    def test_bid_exact_is_true_and_exact(self) -> None:
        result = resolve_liar(
            Bid(3, 3, player_id=1),
            challenger_id=2,
            all_dice=[3, 3, 3, 5, 5],
            percolateur=False,
        )
        assert result.bid_was_exact is True
        assert result.bid_was_true is True

    def test_perudo_bid_only_ones_count(self) -> None:
        # bid: 2 ones; dice: [1,1,3,4,5] → 2 ones == 2 → challenger loses
        result = resolve_liar(
            Bid(2, 1, player_id=1),
            challenger_id=2,
            all_dice=[1, 1, 3, 4, 5],
            percolateur=False,
        )
        assert result.bid_was_true is True
        assert result.total_matching == 2


# ---------------------------------------------------------------------------
# resolve_exact
# ---------------------------------------------------------------------------


class TestResolveExact:
    def test_exact_correct_loser_is_none(self) -> None:
        # bid: 2 fives; dice: [5,5,2] → total=2 == 2 → caller wins
        result = resolve_exact(
            Bid(2, 5, player_id=1),
            caller_id=3,
            all_dice=[5, 5, 2],
            percolateur=False,
        )
        assert result.bid_was_exact is True
        assert result.bid_was_true is True
        assert result.loser_id is None

    def test_exact_wrong_over_caller_loses(self) -> None:
        # bid: 2 fives; dice: [5,5,5] → total=3 != 2 → caller (3) loses
        result = resolve_exact(
            Bid(2, 5, player_id=1),
            caller_id=3,
            all_dice=[5, 5, 5],
            percolateur=False,
        )
        assert result.bid_was_exact is False
        assert result.loser_id == 3

    def test_exact_wrong_under_caller_loses(self) -> None:
        # bid: 4 fives; dice: [5,5,2] → total=2 != 4 → caller loses
        result = resolve_exact(
            Bid(4, 5, player_id=1),
            caller_id=3,
            all_dice=[5, 5, 2],
            percolateur=False,
        )
        assert result.bid_was_exact is False
        assert result.bid_was_true is False
        assert result.loser_id == 3

    def test_exact_percolateur_no_jokers(self) -> None:
        # bid: 3 fours; dice: [4,4,1] → percolateur: total=2 != 3 → caller loses
        result = resolve_exact(
            Bid(3, 4, player_id=1),
            caller_id=2,
            all_dice=[4, 4, 1],
            percolateur=True,
        )
        assert result.bid_was_exact is False
        assert result.loser_id == 2

    def test_exact_perudo_value(self) -> None:
        # bid: 2 ones; dice: [1,1,3] → total=2 == 2 → caller wins
        result = resolve_exact(
            Bid(2, 1, player_id=0),
            caller_id=1,
            all_dice=[1, 1, 3],
            percolateur=False,
        )
        assert result.bid_was_exact is True
        assert result.loser_id is None


# ---------------------------------------------------------------------------
# Property: resolve_liar consistency
# ---------------------------------------------------------------------------


@given(
    q=st.integers(min_value=1, max_value=30),
    v=st.integers(min_value=1, max_value=6),
    dice=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=30),
    percolateur=st.booleans(),
)
def test_resolve_liar_loser_is_one_of_two_players(
    q: int, v: int, dice: list[int], percolateur: bool
) -> None:
    """loser_id must always be either bidder (0) or challenger (1)."""
    result = resolve_liar(
        Bid(q, v, player_id=0),
        challenger_id=1,
        all_dice=dice,
        percolateur=percolateur,
    )
    assert result.loser_id in {0, 1}


@given(
    q=st.integers(min_value=1, max_value=30),
    v=st.integers(min_value=1, max_value=6),
    dice=st.lists(st.integers(min_value=1, max_value=6), min_size=0, max_size=30),
    percolateur=st.booleans(),
)
def test_resolve_liar_bid_true_iff_total_gte_q(
    q: int, v: int, dice: list[int], percolateur: bool
) -> None:
    result = resolve_liar(
        Bid(q, v, player_id=0),
        challenger_id=1,
        all_dice=dice,
        percolateur=percolateur,
    )
    assert result.bid_was_true == (result.total_matching >= q)
    assert result.bid_was_exact == (result.total_matching == q)
