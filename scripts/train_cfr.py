"""
CFR self-play training script.

Trains one CFRBot per player-count configuration via self-play, evaluates
against ThresholdBot every --eval-every iterations, and saves the best
policy to models/cfr_Np.pkl.

Usage:
    python scripts/train_cfr.py
    python scripts/train_cfr.py --iters 50000 --players 4
    python scripts/train_cfr.py --iters 200000 --players 3 4 5 6 --out models/
    python scripts/train_cfr.py --iters 100000 --players 4 --eval-every 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perudo.m3 import ThresholdBot, run_simulation
from perudo.m4 import CFRBot, CFRTrainer, Policy


def evaluate(
    policy: Policy,
    n_players: int,
    n_games: int = 2000,
    seed: int = 999,
) -> tuple[float, float, float, float]:
    """Run CFRBot vs ThresholdBot and return (win_rate, ci_lo, ci_hi, fallback_rate)."""
    bot = CFRBot(policy)
    opponents = [ThresholdBot() for _ in range(n_players - 1)]
    results = run_simulation(n_games, [bot] + opponents, seed=seed)
    s = results.strategy_stats[0]
    lo, hi = s.wilson_ci()
    return s.win_rate, lo, hi, bot.fallback_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFR bot for Perudo")
    parser.add_argument(
        "--iters", type=int, default=100_000,
        help="Total self-play iterations per player count (default: 100 000)",
    )
    parser.add_argument(
        "--players", nargs="+", type=int, default=[4],
        help="Player counts to train (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed (default: 42)",
    )
    parser.add_argument(
        "--out", type=str, default="models",
        help="Output directory for policy files (default: models/)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=10_000,
        help="Evaluate every N iterations (default: 10 000)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=2000,
        help="Games per evaluation run (default: 2 000)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  CFR Self-Play Training — Perudo")
    print("=" * 62)
    print(f"  Joueurs      : {args.players}")
    print(f"  Itérations   : {args.iters:,}")
    print(f"  Eval tous les: {args.eval_every:,} iters ({args.eval_games} parties)")
    print(f"  Sortie       : {out}/")
    print("=" * 62)

    for n_players in args.players:
        print(f"\n{'─' * 62}")
        print(f"  {n_players} joueurs — démarrage")
        print(f"{'─' * 62}\n")

        trainer = CFRTrainer()
        best_win_rate = 0.0
        best_policy: Policy | None = None
        t0 = time.perf_counter()
        chunk = args.eval_every

        done = 0
        chunk_idx = 0
        while done < args.iters:
            n = min(chunk, args.iters - done)
            trainer.train(
                n,
                n_players,
                seed=args.seed + chunk_idx * 7,
                verbose=True,
            )
            done += n
            chunk_idx += 1

            policy = trainer.to_policy()
            wr, lo, hi, fb = evaluate(
                policy, n_players, n_games=args.eval_games, seed=args.seed + 999
            )
            elapsed = time.perf_counter() - t0
            marker = ""
            if wr > best_win_rate:
                best_win_rate = wr
                best_policy = policy
                marker = "  ← meilleur"

            print(
                f"  iter={trainer.n_iters:>7,}  states={policy.n_states:>7,}  "
                f"win={wr:.1%} [{lo:.1%},{hi:.1%}]  "
                f"fallback={fb:.1%}  t={elapsed:.0f}s{marker}"
            )

        # Save best policy
        assert best_policy is not None
        path = out / f"cfr_{n_players}p.pkl"
        best_policy.save(path)
        print(f"\n  Modèle sauvegardé : {path}")
        print(f"  Info-states : {best_policy.n_states:,}")
        print(f"  Win rate final vs ThresholdBot : {best_win_rate:.1%}")

    print("\n" + "=" * 62)
    print("  Entraînement terminé.")
    print("  Lancez le serveur et ouvrez /sim pour tester CFRBot.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
