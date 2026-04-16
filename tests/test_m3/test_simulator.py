"""
Tests for M3 — Monte Carlo simulator.

Invariants verified:
  1. Every game has exactly one winner.
  2. All games complete (no infinite loops).
  3. Reproducibility: same seed → same results.
  4. Performance: 100-game run < 10 s (smoke test).
  5. Each round ends with exactly one Liar or Exact action.
  6. Every raise bid is a valid raise over the previous bid.
  7. Liar/Exact can only be called when there is a current bid.
  8. SimulationResults win counts sum to n_games.
  9. Wilson CI bounds are within [0, 1] and lo <= hi.
  10. CSV and Markdown outputs are written when output_dir provided.
"""

from __future__ import annotations

import time
from pathlib import Path

from perudo.core.rules import is_valid_raise
from perudo.m3 import (
    Honest,
    RandomLegal,
    ThresholdBot,
    run_simulation,
)
from perudo.m3.reporter import StrategyStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42
_TWO_PLAYERS = [RandomLegal(), RandomLegal()]
_THREE_PLAYERS = [RandomLegal(), Honest(), ThresholdBot()]


# ---------------------------------------------------------------------------
# Invariant 1 + 2: every game finishes with one winner
# ---------------------------------------------------------------------------


class TestGameCompletion:
    def test_two_players_has_winner(self) -> None:
        results = run_simulation(20, _TWO_PLAYERS, seed=_SEED)
        assert all(g.winner_id in {0, 1} for g in results.game_records)

    def test_three_players_has_winner(self) -> None:
        results = run_simulation(20, _THREE_PLAYERS, seed=_SEED)
        assert all(g.winner_id in {0, 1, 2} for g in results.game_records)

    def test_no_game_has_zero_rounds(self) -> None:
        results = run_simulation(20, _TWO_PLAYERS, seed=_SEED)
        assert all(g.n_rounds >= 1 for g in results.game_records)


# ---------------------------------------------------------------------------
# Invariant 3: reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_winner_sequence(self) -> None:
        r1 = run_simulation(50, _TWO_PLAYERS, seed=_SEED)
        r2 = run_simulation(50, _TWO_PLAYERS, seed=_SEED)
        winners1 = [g.winner_id for g in r1.game_records]
        winners2 = [g.winner_id for g in r2.game_records]
        assert winners1 == winners2

    def test_different_seeds_differ(self) -> None:
        r1 = run_simulation(50, _TWO_PLAYERS, seed=0)
        r2 = run_simulation(50, _TWO_PLAYERS, seed=99)
        winners1 = [g.winner_id for g in r1.game_records]
        winners2 = [g.winner_id for g in r2.game_records]
        assert winners1 != winners2


# ---------------------------------------------------------------------------
# Invariant 5: each round ends with Liar or Exact
# ---------------------------------------------------------------------------


class TestRoundStructure:
    def test_each_round_ends_with_liar_or_exact(self) -> None:
        results = run_simulation(10, _TWO_PLAYERS, seed=_SEED)
        # Group bid_records by (game_id, round_id)
        from collections import defaultdict

        round_actions: dict[tuple[int, int], list[str]] = defaultdict(list)
        for b in results.bid_records:
            round_actions[(b.game_id, b.round_id)].append(b.action_type)

        for actions in round_actions.values():
            # Last action in a round must be "liar" or "exact"
            assert actions[-1] in {"liar", "exact"}, (
                f"Round ended with action: {actions[-1]}"
            )

    def test_liar_exact_only_after_bid(self) -> None:
        results = run_simulation(10, _THREE_PLAYERS, seed=_SEED)
        from collections import defaultdict

        round_actions: dict[tuple[int, int], list[str]] = defaultdict(list)
        for b in results.bid_records:
            round_actions[(b.game_id, b.round_id)].append(b.action_type)

        for actions in round_actions.values():
            # First action must be a raise (can't call liar/exact with no bid)
            assert actions[0] == "raise", (
                f"First action in round was not raise: {actions[0]}"
            )


# ---------------------------------------------------------------------------
# Invariant 6: all raises are legal
# ---------------------------------------------------------------------------


class TestRaiseLegality:
    def test_all_raises_are_valid(self) -> None:
        from collections import defaultdict

        from perudo.core.types import Bid

        results = run_simulation(10, _THREE_PLAYERS, seed=_SEED)

        # Reconstruct per-round bid sequences
        rounds: dict[tuple[int, int], list] = defaultdict(list)
        for b in results.bid_records:
            rounds[(b.game_id, b.round_id)].append(b)

        for actions in rounds.values():
            prev_bid = None
            for b in actions:
                if b.action_type == "raise":
                    assert b.quantity is not None
                    assert b.value is not None
                    current_bid = Bid(b.quantity, b.value, b.player_id)
                    if prev_bid is not None:
                        assert is_valid_raise(current_bid, prev_bid), (
                            f"Illegal raise {current_bid} over {prev_bid}"
                        )
                    prev_bid = current_bid


# ---------------------------------------------------------------------------
# Invariant 8: win counts sum to n_games
# ---------------------------------------------------------------------------


class TestStats:
    def test_win_counts_sum_to_n_games(self) -> None:
        results = run_simulation(100, _TWO_PLAYERS, seed=_SEED)
        total_wins = sum(s.wins for s in results.strategy_stats)
        assert total_wins == results.n_games

    def test_win_rates_in_range(self) -> None:
        results = run_simulation(100, _TWO_PLAYERS, seed=_SEED)
        for s in results.strategy_stats:
            assert 0.0 <= s.win_rate <= 1.0


# ---------------------------------------------------------------------------
# Invariant 9: Wilson CI
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_ci_bounds_valid(self) -> None:
        for wins in [0, 1, 50, 99, 100]:
            s = StrategyStats(name="test", n_games=100, wins=wins)
            lo, hi = s.wilson_ci()
            assert 0.0 <= lo <= hi <= 1.0

    def test_ci_empty(self) -> None:
        s = StrategyStats(name="test", n_games=0, wins=0)
        lo, hi = s.wilson_ci()
        assert lo == 0.0 and hi == 1.0


# ---------------------------------------------------------------------------
# Invariant 10: file output
# ---------------------------------------------------------------------------


class TestFileOutput:
    def test_csv_files_created(self, tmp_path: Path) -> None:
        run_simulation(5, _TWO_PLAYERS, seed=_SEED, output_dir=tmp_path)
        assert (tmp_path / "game_records.csv").exists()
        assert (tmp_path / "bid_records.csv").exists()

    def test_markdown_created(self, tmp_path: Path) -> None:
        run_simulation(5, _TWO_PLAYERS, seed=_SEED, output_dir=tmp_path)
        md = tmp_path / "summary.md"
        assert md.exists()
        content = md.read_text(encoding="utf-8")
        assert "Win rates" in content
        assert "RandomLegal" in content

    def test_game_records_csv_has_correct_rows(self, tmp_path: Path) -> None:
        import csv as csv_mod

        n = 5
        run_simulation(n, _TWO_PLAYERS, seed=_SEED, output_dir=tmp_path)
        with (tmp_path / "game_records.csv").open(encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == n


# ---------------------------------------------------------------------------
# Invariant 4: performance smoke test
# ---------------------------------------------------------------------------


def test_performance_100_games() -> None:
    """100 games with 3 players must complete in under 10 seconds."""
    start = time.perf_counter()
    run_simulation(100, _THREE_PLAYERS, seed=_SEED)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"100-game simulation took {elapsed:.2f} s (limit 10 s)"


# ---------------------------------------------------------------------------
# Strategy-specific tests
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_honest_never_triggers_percolateur(self) -> None:
        """Honest.wants_percolateur always returns False."""
        from perudo.core.types import GameState, Player, RoundState
        from perudo.m3.strategies import Honest

        h = Honest()
        state = GameState(
            players=[Player(id=0, dice=[3])],
            round=RoundState(
                bids=[], current_player_id=0, percolateur=False, starter_id=0
            ),
            turn_order=[0],
        )
        assert h.wants_percolateur(state) is False

    def test_threshold_bot_name_with_config(self) -> None:
        from perudo.m2 import RecommenderConfig

        cfg = RecommenderConfig(threshold_liar=0.45, threshold_exact=0.35)
        bot = ThresholdBot(cfg)
        assert "0.45" in bot.name
        assert "0.35" in bot.name

    def test_random_legal_name(self) -> None:
        assert RandomLegal().name == "RandomLegal"
