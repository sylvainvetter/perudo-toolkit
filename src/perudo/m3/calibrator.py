"""
M3 — Grid-search calibrator for M2 recommender thresholds.

Sweeps threshold_liar × threshold_exact, runs N games per config point
against a fixed set of opponents, and reports win rates with Wilson CIs.

Usage:
    from perudo.m3.calibrator import calibrate, write_calibration_report
    results = calibrate(n_games=500, seed=42)
    write_calibration_report(results, Path("results/calibration.md"))
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perudo.m2 import RecommenderConfig
from perudo.m3.reporter import StrategyStats
from perudo.m3.simulator import run_simulation
from perudo.m3.strategies import Honest, Strategy, ThresholdBot

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

_LIAR_VALUES: list[float] = [round(v, 2) for v in np.arange(0.30, 0.75, 0.05)]
_EXACT_VALUES: list[float] = [round(v, 2) for v in np.arange(0.15, 0.60, 0.05)]

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    """One cell of the calibration grid."""

    threshold_liar: float
    threshold_exact: float
    win_rate: float
    ci_lo: float
    ci_hi: float
    n_games: int

    def wilson_width(self) -> float:
        return self.ci_hi - self.ci_lo


@dataclass
class CalibrationResults:
    """Full grid-search output."""

    points: list[CalibrationPoint]
    n_games_per_config: int
    opponent_names: list[str]
    liar_values: list[float]
    exact_values: list[float]

    @property
    def best(self) -> CalibrationPoint:
        return max(self.points, key=lambda p: p.win_rate)

    def get(self, liar: float, exact: float) -> CalibrationPoint | None:
        for p in self.points:
            liar_ok = abs(p.threshold_liar - liar) < 1e-9
            if liar_ok and abs(p.threshold_exact - exact) < 1e-9:
                return p
        return None


# ---------------------------------------------------------------------------
# Core calibration function
# ---------------------------------------------------------------------------


def calibrate(
    n_games: int = 500,
    *,
    liar_values: list[float] | None = None,
    exact_values: list[float] | None = None,
    opponents: list[Strategy] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> CalibrationResults:
    """
    Run a grid search over (threshold_liar, threshold_exact).

    For each grid point, simulates *n_games* games with:
        [ThresholdBot(config), *opponents]

    Args:
        n_games:       Games per grid point.
        liar_values:   threshold_liar grid (default: 0.30..0.70 step 0.05).
        exact_values:  threshold_exact grid (default: 0.15..0.55 step 0.05).
        opponents:     Fixed opponent strategies (default: [Honest(), Honest()]).
        seed:          Base RNG seed; each config gets seed + deterministic offset.
        verbose:       Print progress.

    Returns:
        CalibrationResults with all grid points ranked by win rate.
    """
    lv = liar_values or _LIAR_VALUES
    ev = exact_values or _EXACT_VALUES
    opps: list[Strategy] = opponents if opponents is not None else [Honest(), Honest()]
    opp_names = [s.name for s in opps]

    total = len(lv) * len(ev)
    points: list[CalibrationPoint] = []
    done = 0

    rng_meta = np.random.default_rng(seed)

    for liar in lv:
        for exact in ev:
            # Unique seed per config for reproducibility
            config_seed = int(rng_meta.integers(0, 2**31))
            config = RecommenderConfig(threshold_liar=liar, threshold_exact=exact)
            strategies: list[Strategy] = [ThresholdBot(config)] + list(opps)

            results = run_simulation(n_games, strategies, seed=config_seed)
            bot_stats: StrategyStats = results.strategy_stats[0]
            lo, hi = bot_stats.wilson_ci()

            points.append(
                CalibrationPoint(
                    threshold_liar=liar,
                    threshold_exact=exact,
                    win_rate=bot_stats.win_rate,
                    ci_lo=lo,
                    ci_hi=hi,
                    n_games=n_games,
                )
            )

            done += 1
            if verbose:
                best_so_far = max(points, key=lambda p: p.win_rate)
                print(
                    f"  [{done:3d}/{total}] liar={liar:.2f} exact={exact:.2f} "
                    f"= {bot_stats.win_rate:.1%}  "
                    f"(best so far: {best_so_far.win_rate:.1%} "
                    f"at liar={best_so_far.threshold_liar:.2f} "
                    f"exact={best_so_far.threshold_exact:.2f})"
                )

    return CalibrationResults(
        points=points,
        n_games_per_config=n_games,
        opponent_names=opp_names,
        liar_values=lv,
        exact_values=ev,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_calibration_csv(results: CalibrationResults, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold_liar", "threshold_exact", "win_rate", "ci_lo", "ci_hi"])
        for p in sorted(results.points, key=lambda p: -p.win_rate):
            w.writerow(
                [
                    p.threshold_liar,
                    p.threshold_exact,
                    round(p.win_rate, 4),
                    round(p.ci_lo, 4),
                    round(p.ci_hi, 4),
                ]
            )


def write_calibration_report(results: CalibrationResults, path: Path) -> None:
    """Write a Markdown report with ASCII heatmap and best-config summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    best = results.best
    lv = results.liar_values
    ev = results.exact_values

    lines: list[str] = []
    lines.append("# Calibration M2 — Résultats\n")
    lines.append(f"**Parties par config :** {results.n_games_per_config:,}  ")
    lines.append(f"**Adversaires :** {', '.join(results.opponent_names)}  ")
    lines.append(f"**Configs testées :** {len(results.points)}\n")

    lines.append("## Meilleure configuration\n")
    lines.append("| Paramètre | Valeur |")
    lines.append("|---|---|")
    lines.append(f"| `threshold_liar` | **{best.threshold_liar:.2f}** |")
    lines.append(f"| `threshold_exact` | **{best.threshold_exact:.2f}** |")
    lines.append(f"| Taux de victoire | **{best.win_rate:.1%}** |")
    lines.append(f"| IC 95 % Wilson | [{best.ci_lo:.1%}, {best.ci_hi:.1%}] |\n")

    # Top 10
    lines.append("## Top 10 configurations\n")
    lines.append("| threshold_liar | threshold_exact | Win rate | IC 95 % |")
    lines.append("|---|---|---|---|")
    for p in sorted(results.points, key=lambda p: -p.win_rate)[:10]:
        marker = " ← **optimal**" if p is best else ""
        lines.append(
            f"| {p.threshold_liar:.2f} | {p.threshold_exact:.2f} "
            f"| {p.win_rate:.1%} | [{p.ci_lo:.1%}, {p.ci_hi:.1%}] |{marker}"
        )

    # ASCII heatmap
    lines.append("\n## Heatmap (taux de victoire)\n")
    col_w = 7
    header = "liar \\ exact |" + "".join(f" {e:.2f} ".ljust(col_w) for e in ev)
    lines.append("```")
    lines.append(header)
    lines.append("-" * len(header))
    for liar in lv:
        row = f"  {liar:.2f}       |"
        for exact in ev:
            pt = results.get(liar, exact)
            if pt:
                cell = f"{pt.win_rate:.1%}"
                row += f" {cell} " if pt is not best else f"[{cell}]"
            else:
                row += "  —    "
        lines.append(row)
    lines.append("```")
    lines.append(
        f"\n`[xx.x%]` = meilleure config "
        f"(liar={best.threshold_liar:.2f}, exact={best.threshold_exact:.2f})\n"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
