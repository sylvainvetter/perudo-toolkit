"""
Calibration multi-joueurs — grid search threshold_liar x threshold_exact
pour 3, 4, 5 et 6 joueurs.

Usage :
    python scripts/calibrate_multi.py
    python scripts/calibrate_multi.py --players 4 5 --games 300
    python scripts/calibrate_multi.py --players 3 --games 1000 --out results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ajoute le src/ au path si besoin
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perudo.m2 import RecommenderConfig
from perudo.m3 import Honest, RandomLegal, ThresholdBot, run_simulation
from perudo.m3.calibrator import (
    CalibrationResults,
    calibrate,
    write_calibration_csv,
    write_calibration_report,
)
from perudo.m3.strategies import Strategy


def _build_opponents(n_players: int) -> list[Strategy]:
    """Adversaires fixes : rempli avec Honest, dernier avec RandomLegal."""
    if n_players < 2:
        raise ValueError("n_players >= 2")
    opps: list[Strategy] = []
    for i in range(n_players - 1):
        opps.append(Honest() if i < n_players - 2 else RandomLegal())
    return opps


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration multi-joueurs Perudo")
    parser.add_argument(
        "--players", nargs="+", type=int, default=[3, 4, 5, 6],
        help="Nombres de joueurs a tester (ex: 3 4 5 6)"
    )
    parser.add_argument(
        "--games", type=int, default=500,
        help="Parties par point de grille (defaut: 500)"
    )
    parser.add_argument(
        "--out", type=str, default="results",
        help="Dossier de sortie (defaut: results/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aleatoire (defaut: 42)"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_configs = 9 * 9  # grille 9x9

    print("=" * 60)
    print("  Calibration M2 — grid search multi-joueurs")
    print("=" * 60)
    print(f"  Joueurs      : {args.players}")
    print(f"  Parties/cfg  : {args.games:,}")
    print(f"  Configs      : {n_configs}")
    print(f"  Total parties: {len(args.players) * n_configs * args.games:,}")
    print(f"  Sortie       : {out}/")
    print("=" * 60)
    print()

    all_best: list[tuple[int, CalibrationResults]] = []

    for n in args.players:
        opps = _build_opponents(n)
        opp_names = " + ".join(s.name for s in opps)
        print(f"--- {n} joueurs (ThresholdBot vs {opp_names}) ---")
        print()

        t0 = time.perf_counter()

        # Monkey-patch run_simulation inside calibrate to show progress
        # On passe verbose=False au calibrateur et on affiche nous-memes
        results = _calibrate_verbose(n_games=args.games, opponents=opps, seed=args.seed)

        elapsed = time.perf_counter() - t0
        best = results.best
        print(f"\n  Termine en {elapsed:.0f}s")
        print(f"  Meilleure config : liar={best.threshold_liar:.2f}  "
              f"exact={best.threshold_exact:.2f}  "
              f"-> {best.win_rate:.1%}  "
              f"IC=[{best.ci_lo:.1%}, {best.ci_hi:.1%}]")
        print()

        write_calibration_csv(results, out / f"calibration_{n}p.csv")
        write_calibration_report(results, out / f"calibration_{n}p.md")

        all_best.append((n, results))

    # Synthese
    print("=" * 60)
    print("  SYNTHESE — Meilleure config par nombre de joueurs")
    print("=" * 60)
    print(f"  {'Joueurs':>8}  {'liar':>6}  {'exact':>6}  {'Win rate':>10}  {'IC 95%'}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*20}")
    for n, res in all_best:
        b = res.best
        print(f"  {n:>8}  {b.threshold_liar:>6.2f}  {b.threshold_exact:>6.2f}"
              f"  {b.win_rate:>10.1%}  [{b.ci_lo:.1%}, {b.ci_hi:.1%}]")
    print()
    print(f"  Rapports : {out}/calibration_Xp.md")


def _calibrate_verbose(
    n_games: int,
    opponents: list[Strategy],
    seed: int,
) -> CalibrationResults:
    """Calibrate avec barre de progression dans le terminal."""
    import numpy as np
    from perudo.m3.calibrator import (
        CalibrationPoint,
        CalibrationResults,
        _EXACT_VALUES,
        _LIAR_VALUES,
    )
    from perudo.m3.reporter import StrategyStats

    lv = _LIAR_VALUES
    ev = _EXACT_VALUES
    total = len(lv) * len(ev)
    points: list[CalibrationPoint] = []
    rng_meta = np.random.default_rng(seed)
    opp_names = [s.name for s in opponents]

    for idx, liar in enumerate(lv):
        for exact in ev:
            config_seed = int(rng_meta.integers(0, 2**31))
            config = RecommenderConfig(threshold_liar=liar, threshold_exact=exact)
            strats: list[Strategy] = [ThresholdBot(config)] + list(opponents)

            done = len(points)
            filled = int(30 * done / total)
            bar = "#" * filled + "-" * (30 - filled)
            best_so_far = (max(points, key=lambda p: p.win_rate).win_rate
                           if points else 0.0)
            sys.stdout.write(
                f"\r  [{bar}] {done+1}/{total}  "
                f"liar={liar:.2f} exact={exact:.2f}  "
                f"best so far: {best_so_far:.1%}   "
            )
            sys.stdout.flush()

            results = run_simulation(n_games, strats, seed=config_seed, verbose=False)
            bot_stats: StrategyStats = results.strategy_stats[0]
            lo, hi = bot_stats.wilson_ci()
            points.append(CalibrationPoint(
                threshold_liar=liar,
                threshold_exact=exact,
                win_rate=bot_stats.win_rate,
                ci_lo=lo,
                ci_hi=hi,
                n_games=n_games,
            ))

    return CalibrationResults(
        points=points,
        n_games_per_config=n_games,
        opponent_names=opp_names,
        liar_values=lv,
        exact_values=ev,
    )


if __name__ == "__main__":
    main()
