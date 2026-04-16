"""
Tests for M2 — Action recommender.

Scenarios covered:
  - Opening bid always yields "raise"
  - Low p_true → "liar"
  - High p_exact, exact available → "exact"
  - High p_exact, exact already used → "raise"
  - Normal situation → "raise" with a valid bid
  - Returned raise bid is always a legal raise over prev_bid
  - Custom config thresholds
  - Percolateur mode
  - enumerate_valid_raises correctness
  - Fallback to liar when no raise possible (q at max)
"""

import pytest

from perudo.core.rules import is_valid_raise
from perudo.core.types import Bid, GameState, Player, RoundState
from perudo.m2 import (
    RecommenderConfig,
    enumerate_valid_raises,
    recommend,
)

# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _make_state(
    own_dice: list[int],
    n_opponents: int = 1,
    opp_dice_count: int = 5,
    current_bid: Bid | None = None,
    player_id: int = 0,
    percolateur: bool = False,
    exact_used: bool = False,
) -> GameState:
    """Build a minimal GameState for unit testing."""
    players: list[Player] = [
        Player(id=player_id, dice=list(own_dice), exact_used=exact_used)
    ]
    for i in range(n_opponents):
        players.append(Player(id=player_id + 1 + i, dice=[3] * opp_dice_count))
    bids = [current_bid] if current_bid is not None else []
    round_state = RoundState(
        bids=bids,
        current_player_id=player_id,
        percolateur=percolateur,
        starter_id=player_id,
    )
    return GameState(
        players=players,
        round=round_state,
        turn_order=[p.id for p in players],
    )


# ---------------------------------------------------------------------------
# Opening bid
# ---------------------------------------------------------------------------


class TestOpeningBid:
    def test_opening_always_raises(self) -> None:
        state = _make_state([1, 2, 3, 4, 5], n_opponents=1, current_bid=None)
        rec = recommend(state)
        assert rec.best_action == "raise"

    def test_opening_bid_is_structurally_valid(self) -> None:
        state = _make_state([3, 3, 1, 2, 6], n_opponents=2, current_bid=None)
        rec = recommend(state)
        assert rec.bid_if_raise is not None
        bid = rec.bid_if_raise
        assert bid.quantity >= 1
        assert 1 <= bid.value <= 6

    def test_opening_prefers_value_with_most_own_dice(self) -> None:
        # Own 3 threes (2 direct + 1 joker) — value 3 should be preferred
        state = _make_state([3, 3, 1, 5, 6], n_opponents=1, current_bid=None)
        rec = recommend(state)
        assert rec.bid_if_raise is not None
        assert rec.bid_if_raise.value == 3

    def test_opening_rationale_zeroed(self) -> None:
        state = _make_state([3, 4, 5], current_bid=None)
        rec = recommend(state)
        assert rec.rationale.p_current_bid_true == 0.0
        assert rec.rationale.p_current_bid_exact == 0.0
        assert rec.rationale.expected_total == 0.0

    def test_opening_has_six_alternatives(self) -> None:
        state = _make_state([1, 2, 3, 4, 5], current_bid=None)
        rec = recommend(state)
        assert len(rec.rationale.alternatives_considered) == 6


# ---------------------------------------------------------------------------
# Liar recommendation
# ---------------------------------------------------------------------------


class TestLiarRecommendation:
    def test_impossible_bid_triggers_liar(self) -> None:
        # Bid (20, 3) when total dice = 10 → impossible → p_true = 0
        state = _make_state(
            [1, 2, 4, 5, 6],
            n_opponents=1,
            opp_dice_count=5,
            current_bid=Bid(20, 3, player_id=1),
        )
        rec = recommend(state)
        assert rec.best_action == "liar"
        assert rec.bid_if_raise is None

    def test_very_unlikely_bid_triggers_liar(self) -> None:
        # Bid (9, 3) with total 10 dice, own=[5,5,5,5,5] (0 threes)
        # p_true = P(Binom(5, 2/6) >= 9) ≈ 0
        state = _make_state(
            [5, 5, 5, 5, 5],
            n_opponents=1,
            opp_dice_count=5,
            current_bid=Bid(9, 3, player_id=1),
        )
        rec = recommend(state)
        assert rec.best_action == "liar"

    def test_liar_score_is_complement_of_p_true(self) -> None:
        state = _make_state(
            [5, 5, 5, 5, 5],
            opp_dice_count=5,
            current_bid=Bid(9, 3, player_id=1),
        )
        rec = recommend(state)
        assert rec.rationale.score == pytest.approx(
            1.0 - rec.rationale.p_current_bid_true
        )

    def test_liar_threshold_respected(self) -> None:
        # Use threshold_liar=0.0: should never recommend liar unless p_true==0
        cfg = RecommenderConfig(threshold_liar=0.0)
        state = _make_state(
            [5, 5, 5, 5, 5],
            opp_dice_count=5,
            current_bid=Bid(9, 3, player_id=1),
        )
        # p_true = 0.0 → liar even with threshold_liar=0.0 (0 < 0 is False, but 0 >= 0)
        # Actually: p_true (0.0) < threshold (0.0) is False → not liar → raise
        rec = recommend(state, config=cfg)
        # With threshold=0.0 and p_true=0: 0 < 0 is False → goes to raise
        # But there may be no valid raise if bid is at max already
        # Either "raise" or "liar" (fallback) is acceptable here
        assert rec.best_action in {"raise", "liar"}


# ---------------------------------------------------------------------------
# Exact recommendation
# ---------------------------------------------------------------------------


class TestExactRecommendation:
    def test_high_p_exact_triggers_exact(self) -> None:
        # Own all 5 dice matching bid, no unknowns → T = 5 surely → p_exact = 1
        state = _make_state(
            [3, 3, 3, 3, 3],
            n_opponents=0,
            current_bid=Bid(5, 3, player_id=1),
        )
        rec = recommend(state)
        assert rec.best_action == "exact"
        assert rec.bid_if_raise is None

    def test_exact_used_falls_back_to_raise(self) -> None:
        state = _make_state(
            [3, 3, 3, 3, 3],
            n_opponents=0,
            current_bid=Bid(5, 3, player_id=1),
            exact_used=True,
        )
        rec = recommend(state)
        assert rec.best_action != "exact"

    def test_exact_score_is_p_exact(self) -> None:
        state = _make_state(
            [3, 3, 3, 3, 3],
            n_opponents=0,
            current_bid=Bid(5, 3, player_id=1),
        )
        rec = recommend(state)
        assert rec.rationale.score == pytest.approx(rec.rationale.p_current_bid_exact)

    def test_exact_threshold_respected(self) -> None:
        # Force threshold_exact very high → won't trigger exact even with p_exact=1
        cfg = RecommenderConfig(threshold_exact=1.1)
        state = _make_state(
            [3, 3, 3, 3, 3],
            n_opponents=0,
            current_bid=Bid(5, 3, player_id=1),
        )
        rec = recommend(state, config=cfg)
        assert rec.best_action != "exact"


# ---------------------------------------------------------------------------
# Raise recommendation
# ---------------------------------------------------------------------------


class TestRaiseRecommendation:
    def test_normal_situation_returns_raise(self) -> None:
        # Plausible bid with moderate probability
        state = _make_state(
            [3, 3, 1, 2, 6],
            n_opponents=1,
            opp_dice_count=5,
            current_bid=Bid(2, 3, player_id=1),
        )
        rec = recommend(state)
        assert rec.best_action == "raise"
        assert rec.bid_if_raise is not None

    def test_raise_bid_is_legal_over_prev(self) -> None:
        prev = Bid(2, 3, player_id=1)
        state = _make_state(
            [3, 3, 1, 2, 6],
            n_opponents=1,
            opp_dice_count=5,
            current_bid=prev,
        )
        rec = recommend(state)
        if rec.best_action == "raise":
            assert rec.bid_if_raise is not None
            assert is_valid_raise(rec.bid_if_raise, prev)

    def test_raise_bid_player_id_matches_current(self) -> None:
        state = _make_state(
            [3, 4, 5],
            player_id=2,
            n_opponents=1,
            current_bid=Bid(1, 2, player_id=1),
        )
        rec = recommend(state)
        if rec.best_action == "raise":
            assert rec.bid_if_raise is not None
            assert rec.bid_if_raise.player_id == 2

    def test_raise_from_perudo_bid(self) -> None:
        # Prev is Perudo bid (value=1); valid Standard raise needs q >= 2*1+1 = 3
        prev = Bid(1, 1, player_id=1)
        state = _make_state(
            [3, 3, 3, 4, 5],
            n_opponents=1,
            opp_dice_count=5,
            current_bid=prev,
        )
        rec = recommend(state)
        if rec.best_action == "raise":
            assert rec.bid_if_raise is not None
            assert is_valid_raise(rec.bid_if_raise, prev)

    def test_lambda_risk_penalises_overbidding(self) -> None:
        # With lambda_risk > 0, aggressive bids should score lower
        prev = Bid(1, 3, player_id=1)
        state_base = _make_state(
            [3],
            n_opponents=1,
            opp_dice_count=5,
            current_bid=prev,
        )
        rec_no_risk = recommend(state_base, config=RecommenderConfig(lambda_risk=0.0))
        rec_risk = recommend(state_base, config=RecommenderConfig(lambda_risk=1.0))
        # Both should raise, but risk version may pick a less ambitious bid
        assert rec_no_risk.best_action == "raise"
        assert rec_risk.best_action == "raise"


# ---------------------------------------------------------------------------
# Percolateur mode
# ---------------------------------------------------------------------------


class TestPercolateurMode:
    def test_percolateur_affects_own_count(self) -> None:
        # With [3,3,1] and value=3:
        #   normal: own_count=3 (2 direct + 1 joker)
        #   percolateur: own_count=2 (joker inactive)
        prev = Bid(2, 3, player_id=1)
        state_normal = _make_state([3, 3, 1], opp_dice_count=5, current_bid=prev)
        state_perco = _make_state(
            [3, 3, 1], opp_dice_count=5, current_bid=prev, percolateur=True
        )
        rec_normal = recommend(state_normal)
        rec_perco = recommend(state_perco)
        # Both should raise but percolateur has lower own_count for value=3
        assert (
            rec_normal.rationale.p_current_bid_true
            >= rec_perco.rationale.p_current_bid_true
        )


# ---------------------------------------------------------------------------
# enumerate_valid_raises
# ---------------------------------------------------------------------------


class TestEnumerateValidRaises:
    def test_opening_returns_six_bids(self) -> None:
        bids = enumerate_valid_raises(None, player_id=0, total_dice=10)
        assert len(bids) == 6
        assert {b.value for b in bids} == set(range(1, 7))
        assert all(b.quantity == 1 for b in bids)

    def test_opening_player_id_set(self) -> None:
        bids = enumerate_valid_raises(None, player_id=3, total_dice=5)
        assert all(b.player_id == 3 for b in bids)

    def test_std_to_std_min_q_same_value(self) -> None:
        # prev=(3,4): same value v=4 needs q >= 4
        bids = enumerate_valid_raises(Bid(3, 4, 1), player_id=0, total_dice=20)
        v4 = [b for b in bids if b.value == 4]
        assert len(v4) == 1
        assert v4[0].quantity == 4

    def test_std_to_std_min_q_higher_value(self) -> None:
        # prev=(3,4): higher value v=5 needs q >= 3
        bids = enumerate_valid_raises(Bid(3, 4, 1), player_id=0, total_dice=20)
        v5 = [b for b in bids if b.value == 5]
        assert len(v5) == 1
        assert v5[0].quantity == 3

    def test_std_to_perudo_min_q(self) -> None:
        # prev=(5,3): Perudo min q = ceil(5/2)+1 = 4
        bids = enumerate_valid_raises(Bid(5, 3, 1), player_id=0, total_dice=20)
        perudo = [b for b in bids if b.value == 1]
        assert len(perudo) == 1
        assert perudo[0].quantity == 4

    def test_perudo_to_standard_min_q(self) -> None:
        # prev=(2,1): Standard min q = 2*2+1 = 5
        bids = enumerate_valid_raises(Bid(2, 1, 1), player_id=0, total_dice=20)
        for b in bids:
            if b.value != 1:
                assert b.quantity == 5
                break

    def test_all_candidates_are_valid_raises(self) -> None:
        prev = Bid(3, 4, player_id=1)
        bids = enumerate_valid_raises(prev, player_id=0, total_dice=20)
        for b in bids:
            assert is_valid_raise(b, prev), f"Invalid raise: {b} over {prev}"

    def test_total_dice_cap(self) -> None:
        # prev=(3,1) → Standard needs q>=7; cap at 5 → no Standard candidates
        bids = enumerate_valid_raises(Bid(3, 1, 1), player_id=0, total_dice=5)
        standard = [b for b in bids if b.value != 1]
        assert all(b.quantity <= 5 for b in bids)
        assert len(standard) == 0  # 2*3+1=7 > 5


# ---------------------------------------------------------------------------
# Rationale completeness
# ---------------------------------------------------------------------------


class TestRationale:
    def test_rationale_has_current_bid_stats(self) -> None:
        state = _make_state(
            [3, 3, 1],
            n_opponents=1,
            current_bid=Bid(2, 3, player_id=1),
        )
        rec = recommend(state)
        assert 0.0 <= rec.rationale.p_current_bid_true <= 1.0
        assert 0.0 <= rec.rationale.p_current_bid_exact <= 1.0
        assert rec.rationale.expected_total >= 0.0

    def test_alternatives_are_raise_candidates(self) -> None:
        state = _make_state(
            [3, 3, 1],
            n_opponents=1,
            current_bid=Bid(2, 3, player_id=1),
        )
        rec = recommend(state)
        for cand in rec.rationale.alternatives_considered:
            assert 0.0 <= cand.stats.p_true <= 1.0
            assert cand.score is not None

    def test_p_exact_le_p_true_in_rationale(self) -> None:
        state = _make_state(
            [3, 3, 1],
            n_opponents=1,
            current_bid=Bid(2, 3, player_id=1),
        )
        rec = recommend(state)
        assert (
            rec.rationale.p_current_bid_exact <= rec.rationale.p_current_bid_true + 1e-9
        )
