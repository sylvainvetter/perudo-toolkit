"""
Tests for M1 — Probability calculator.

Required invariants (section 4.1 of CLAUDE.md):
  1. Sum of distribution ≈ 1.0 (tolerance 1e-9)
  2. P(T >= q) + P(T < q) ≈ 1
  3. Edge case n=0 → deterministic distribution
  4. own_count >= q → P(T >= q) = 1.0
  5. Monotonicity: P(T >= q) strictly decreasing in q (when 0 < P < 1)

Also covers:
  - p_per_die values for all combinations
  - p_exactly consistency with distribution
  - expected_count formula
  - bid_stats integration with own_dice
  - Performance: p_at_least < 1 ms for n <= 30
"""

import time

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from perudo.core.types import Bid
from perudo.m1 import (
    BidStats,
    bid_stats,
    distribution,
    expected_count,
    p_at_least,
    p_exactly,
    p_per_die,
)

TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# p_per_die
# ---------------------------------------------------------------------------


class TestPPerDie:
    def test_normal_play_non_one_is_two_sixths(self) -> None:
        for v in range(2, 7):
            assert p_per_die(v, percolateur=False) == pytest.approx(2 / 6)

    def test_normal_play_value_one_is_one_sixth(self) -> None:
        assert p_per_die(1, percolateur=False) == pytest.approx(1 / 6)

    def test_percolateur_all_values_one_sixth(self) -> None:
        for v in range(1, 7):
            assert p_per_die(v, percolateur=True) == pytest.approx(1 / 6)


# ---------------------------------------------------------------------------
# p_at_least
# ---------------------------------------------------------------------------


class TestPAtLeast:
    def test_own_count_satisfies_bid(self) -> None:
        # own has 3 matching, q=2: no uncertainty needed → P=1
        assert p_at_least(2, n=10, p=1 / 3, own_count=3) == 1.0

    def test_own_count_exactly_meets_bid(self) -> None:
        assert p_at_least(5, n=10, p=1 / 3, own_count=5) == 1.0

    def test_impossible_bid(self) -> None:
        # q=10, n=5, own=3 → need 7 from 5 dice → impossible
        assert p_at_least(10, n=5, p=1 / 3, own_count=3) == 0.0

    def test_no_unknown_dice_bid_false(self) -> None:
        # n=0, own=2, q=3 → P=0
        assert p_at_least(3, n=0, p=1 / 3, own_count=2) == 0.0

    def test_no_unknown_dice_bid_true(self) -> None:
        # n=0, own=3, q=3 → P=1
        assert p_at_least(3, n=0, p=1 / 3, own_count=3) == 1.0

    def test_certain_dice(self) -> None:
        # p=1.0: all unknown dice always contribute → P(T>=q) = 1 if q <= n+own
        assert p_at_least(5, n=5, p=1.0, own_count=0) == pytest.approx(1.0)

    def test_impossible_dice(self) -> None:
        # p=0.0, own=0: no dice can ever contribute → P=0 unless q<=0
        assert p_at_least(1, n=10, p=0.0, own_count=0) == pytest.approx(0.0)

    def test_q_zero_always_one(self) -> None:
        assert p_at_least(0, n=10, p=1 / 3, own_count=0) == 1.0

    def test_known_value(self) -> None:
        # 5 dice, p=1/3, own=0, q=2 → P(Binom(5,1/3) >= 2)
        from scipy.stats import binom

        expected = float(binom.sf(1, 5, 1 / 3))
        assert p_at_least(2, n=5, p=1 / 3, own_count=0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# p_exactly
# ---------------------------------------------------------------------------


class TestPExactly:
    def test_below_own_count_is_zero(self) -> None:
        # own=5, q=3 → T >= 5 always → P(T==3)=0
        assert p_exactly(3, n=10, p=1 / 3, own_count=5) == pytest.approx(0.0)

    def test_above_max_is_zero(self) -> None:
        # own=0, n=3, q=5 → T <= 3 always
        assert p_exactly(5, n=3, p=1 / 3, own_count=0) == pytest.approx(0.0)

    def test_n_zero_own_count_matches(self) -> None:
        # n=0, own=3, q=3 → T=3 always → P(T==3)=1
        assert p_exactly(3, n=0, p=1 / 3, own_count=3) == pytest.approx(1.0)

    def test_n_zero_own_count_no_match(self) -> None:
        assert p_exactly(2, n=0, p=1 / 3, own_count=3) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        from scipy.stats import binom

        # own=1, n=5, p=1/3, q=3 → need 2 from 5 → P(X==2)
        expected = float(binom.pmf(2, 5, 1 / 3))
        assert p_exactly(3, n=5, p=1 / 3, own_count=1) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# expected_count
# ---------------------------------------------------------------------------


class TestExpectedCount:
    def test_formula(self) -> None:
        assert expected_count(n=12, p=1 / 3, own_count=2) == pytest.approx(2 + 12 / 3)

    def test_no_unknown(self) -> None:
        assert expected_count(n=0, p=1 / 3, own_count=4) == pytest.approx(4.0)

    def test_p_zero(self) -> None:
        assert expected_count(n=10, p=0.0, own_count=3) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# distribution — invariant 1: sums to 1
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_sums_to_one(self) -> None:
        dist = distribution(n=10, p=1 / 3, own_count=2)
        assert abs(dist.sum() - 1.0) < TOLERANCE

    def test_n_zero_deterministic(self) -> None:
        # Invariant 3: n=0 → mass entirely at own_count
        dist = distribution(n=0, p=1 / 3, own_count=3)
        assert dist[3] == pytest.approx(1.0)
        assert abs(dist.sum() - 1.0) < TOLERANCE

    def test_length(self) -> None:
        dist = distribution(n=7, p=1 / 4, own_count=2)
        assert len(dist) == 7 + 2 + 1  # 0 .. n + own_count

    def test_p_exactly_matches_distribution(self) -> None:
        n, p, own = 8, 1 / 3, 2
        dist = distribution(n=n, p=p, own_count=own)
        for q in range(n + own + 1):
            assert dist[q] == pytest.approx(p_exactly(q, n, p, own), abs=1e-12)

    def test_p_at_least_matches_distribution_suffix_sum(self) -> None:
        n, p, own = 8, 1 / 3, 2
        dist = distribution(n=n, p=p, own_count=own)
        for q in range(n + own + 2):
            suffix = float(dist[q:].sum()) if q <= n + own else 0.0
            assert p_at_least(q, n, p, own) == pytest.approx(suffix, abs=1e-10)

    def test_nonnegative(self) -> None:
        dist = distribution(n=15, p=2 / 6, own_count=0)
        assert (dist >= 0).all()


# ---------------------------------------------------------------------------
# Required invariants (property-based)
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=0, max_value=25),
    p=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_subnormal=False),
    own_count=st.integers(min_value=0, max_value=5),
)
def test_invariant_distribution_sums_to_one(n: int, p: float, own_count: int) -> None:
    """Invariant 1: sum of distribution ≈ 1.0."""
    dist = distribution(n=n, p=p, own_count=own_count)
    assert abs(dist.sum() - 1.0) < TOLERANCE


@given(
    q=st.integers(min_value=0, max_value=30),
    n=st.integers(min_value=0, max_value=25),
    p=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_subnormal=False),
    own_count=st.integers(min_value=0, max_value=5),
)
def test_invariant_complement(q: int, n: int, p: float, own_count: int) -> None:
    """Invariant 2: P(T >= q) + P(T < q) ≈ 1."""
    p_true = p_at_least(q, n, p, own_count)
    dist = distribution(n=n, p=p, own_count=own_count)
    p_less = float(dist[:q].sum()) if q > 0 else 0.0
    assert abs(p_true + p_less - 1.0) < TOLERANCE


@given(
    n=st.integers(min_value=0, max_value=25),
    p=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_subnormal=False),
    own_count=st.integers(min_value=0, max_value=5),
    q=st.integers(min_value=1, max_value=10),
)
def test_invariant_own_count_satisfies(
    n: int, p: float, own_count: int, q: int
) -> None:
    """Invariant 4: own_count >= q → P(T >= q) = 1.0."""
    assume(own_count >= q)
    assert p_at_least(q, n, p, own_count) == 1.0


@given(
    n=st.integers(min_value=0, max_value=25),
    p=st.floats(min_value=1e-6, max_value=1.0 - 1e-6, allow_nan=False),
    own_count=st.integers(min_value=0, max_value=5),
    q=st.integers(min_value=0, max_value=29),
)
def test_invariant_monotonicity(n: int, p: float, own_count: int, q: int) -> None:
    """Invariant 5: P(T >= q) >= P(T >= q+1)."""
    assert p_at_least(q, n, p, own_count) >= p_at_least(q + 1, n, p, own_count)


# ---------------------------------------------------------------------------
# bid_stats integration
# ---------------------------------------------------------------------------


class TestBidStats:
    def test_returns_bid_stats(self) -> None:
        stats = bid_stats(
            Bid(3, 4, player_id=0),
            own_dice=[4, 4, 1, 2, 6],
            n_unknown=10,
            percolateur=False,
        )
        assert isinstance(stats, BidStats)

    def test_own_count_normal_play(self) -> None:
        # dice [4,4,1,2,6]: value=4 with jokers → 2 fours + 1 joker = 3
        stats = bid_stats(
            Bid(3, 4, player_id=0),
            own_dice=[4, 4, 1, 2, 6],
            n_unknown=10,
            percolateur=False,
        )
        assert stats.own_count == 3

    def test_own_count_percolateur(self) -> None:
        # Same dice, Percolateur: joker inactive → only 2 fours
        stats = bid_stats(
            Bid(3, 4, player_id=0),
            own_dice=[4, 4, 1, 2, 6],
            n_unknown=10,
            percolateur=True,
        )
        assert stats.own_count == 2

    def test_p_die_normal(self) -> None:
        stats = bid_stats(
            Bid(3, 4, player_id=0),
            own_dice=[],
            n_unknown=10,
            percolateur=False,
        )
        assert stats.p_die == pytest.approx(2 / 6)

    def test_p_die_perudo_value(self) -> None:
        stats = bid_stats(
            Bid(2, 1, player_id=0),
            own_dice=[1, 2],
            n_unknown=10,
            percolateur=False,
        )
        assert stats.p_die == pytest.approx(1 / 6)
        assert stats.own_count == 1  # only the literal 1 counts

    def test_p_true_and_p_exact_consistent(self) -> None:
        stats = bid_stats(
            Bid(3, 3, player_id=0),
            own_dice=[3, 1],
            n_unknown=12,
            percolateur=False,
        )
        assert stats.p_true >= stats.p_exact

    def test_bid_already_met(self) -> None:
        # own dice already exceed bid quantity
        stats = bid_stats(
            Bid(2, 5, player_id=0),
            own_dice=[5, 5, 5],
            n_unknown=10,
            percolateur=False,
        )
        assert stats.p_true == pytest.approx(1.0)
        assert stats.own_count == 3

    def test_expected_value(self) -> None:
        # own=[2,2], value=2, n_unknown=12, normal play
        # own_count=2, p_die=2/6, expected = 2 + 12*(2/6) = 2+4 = 6
        stats = bid_stats(
            Bid(1, 2, player_id=0),
            own_dice=[2, 2],
            n_unknown=12,
            percolateur=False,
        )
        assert stats.expected == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Performance: p_at_least < 1 ms (spec section 4.1)
# ---------------------------------------------------------------------------


def test_p_at_least_performance() -> None:
    """p_at_least must complete in < 1 ms for n <= 30 (spec requirement)."""
    n_calls = 1000
    start = time.perf_counter()
    for _ in range(n_calls):
        p_at_least(4, n=30, p=2 / 6, own_count=1)
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_calls
    assert elapsed_ms < 1.0, (
        f"p_at_least took {elapsed_ms:.3f} ms on average (limit 1 ms)"
    )
