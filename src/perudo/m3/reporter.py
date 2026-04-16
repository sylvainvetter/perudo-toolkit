"""
M3 — Simulation reporter: records, stats, CSV and Markdown output.

Wilson 95 % confidence interval is used for all win-rate estimates.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Record types (written by the simulator)
# ---------------------------------------------------------------------------


@dataclass
class BidRecord:
    """One action taken during a game."""

    game_id: int
    round_id: int
    turn_id: int
    player_id: int
    action_type: str  # "raise" | "liar" | "exact"
    quantity: int | None  # non-None for "raise"
    value: int | None  # non-None for "raise"


@dataclass
class GameRecord:
    """Outcome of one complete game."""

    game_id: int
    winner_id: int
    n_rounds: int
    strategy_names: list[str]  # index == player_id


# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------


@dataclass
class StrategyStats:
    """Win-rate statistics for one strategy."""

    name: str
    n_games: int
    wins: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.n_games if self.n_games else 0.0

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score 95 % confidence interval for win rate."""
        n = self.n_games
        p = self.win_rate
        if n == 0:
            return (0.0, 1.0)
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class SimulationResults:
    """Aggregate output of run_simulation."""

    n_games: int
    n_players: int
    strategy_names: list[str]
    game_records: list[GameRecord]
    bid_records: list[BidRecord]
    strategy_stats: list[StrategyStats] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.strategy_stats:
            self._compute_stats()

    def _compute_stats(self) -> None:
        wins = dict.fromkeys(range(self.n_players), 0)
        for g in self.game_records:
            if g.winner_id >= 0:
                wins[g.winner_id] += 1
        self.strategy_stats = [
            StrategyStats(
                name=self.strategy_names[i],
                n_games=self.n_games,
                wins=wins[i],
            )
            for i in range(self.n_players)
        ]

    @property
    def avg_rounds(self) -> float:
        if not self.game_records:
            return 0.0
        return sum(g.n_rounds for g in self.game_records) / len(self.game_records)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(results: SimulationResults, output_dir: Path) -> Path:
    """Write game_records.csv and bid_records.csv to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    games_path = output_dir / "game_records.csv"
    with games_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game_id", "winner_id", "winner_strategy", "n_rounds"])
        for g in results.game_records:
            winner_name = (
                results.strategy_names[g.winner_id] if g.winner_id >= 0 else "none"
            )
            writer.writerow([g.game_id, g.winner_id, winner_name, g.n_rounds])

    bids_path = output_dir / "bid_records.csv"
    with bids_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "round_id",
                "turn_id",
                "player_id",
                "action_type",
                "quantity",
                "value",
            ]
        )
        for b in results.bid_records:
            writer.writerow(
                [
                    b.game_id,
                    b.round_id,
                    b.turn_id,
                    b.player_id,
                    b.action_type,
                    b.quantity or "",
                    b.value or "",
                ]
            )

    return output_dir


def write_markdown(results: SimulationResults, output_path: Path) -> Path:
    """Write a Markdown summary report to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Simulation Summary\n")
    lines.append(f"**Games:** {results.n_games:,}  ")
    lines.append(f"**Players:** {results.n_players}  ")
    lines.append(f"**Average rounds per game:** {results.avg_rounds:.1f}\n")

    lines.append("## Win rates\n")
    lines.append("| Strategy | Wins | Win rate | 95 % CI |")
    lines.append("|---|---:|---:|---|")
    for s in results.strategy_stats:
        lo, hi = s.wilson_ci()
        lines.append(
            f"| {s.name} | {s.wins:,} | {s.win_rate:.1%} | [{lo:.1%}, {hi:.1%}] |"
        )

    lines.append("\n## Notes\n")
    lines.append(
        "- Win rates use Wilson score 95 % confidence intervals.\n"
        "- `RandomLegal` serves as a baseline (uniform random legal actions).\n"
        "- `Honest` bids the rounded expected count; never triggers Percolateur.\n"
        "- `ThresholdBot` uses the M2 recommender (threshold_liar=0.50, "
        "threshold_exact=0.40).\n"
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
