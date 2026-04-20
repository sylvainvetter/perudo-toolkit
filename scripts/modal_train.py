"""
Modal cloud training script — CFR self-play.

Parallelisme sur deux axes :
  - Axis 1 : un container par player count (3/4/5/6 joueurs)
  - Axis 2 : N_WORKERS containers par player count, chacun fait iters/N_WORKERS
             iterations, les strategy_sum sont additionnees a la fin.

CFR est lineairement additif : sum(strategy_sum_i) == run unique equivalent.
Avec 8 workers x 4 player counts = 32 containers en parallele (defaut).

Self-play active par defaut (selfplay_every=50_000) : les adversaires passent
progressivement de ThresholdBot a la strategie CFR apprise, rendant le bot
plus robuste contre des joueurs variés.

Usage:
    python -m modal run scripts/modal_train.py
    python -m modal run scripts/modal_train.py --iters 500000
    python -m modal run scripts/modal_train.py --iters 1000000 --players 4,5,6
    python -m modal run scripts/modal_train.py --workers 16
    python -m modal run scripts/modal_train.py --no-selfplay
"""

from __future__ import annotations

import pickle
from pathlib import Path

import modal

N_WORKERS = 8  # containers par player count (defaut)

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App("perudo-cfr")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2.0", "scipy>=1.13")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(
        Path(__file__).parent.parent / "src",
        remote_path="/root/src",
    )
)


# ---------------------------------------------------------------------------
# Remote worker : train one shard
# ---------------------------------------------------------------------------


@app.function(image=image, cpu=4.0, timeout=7200)
def train_shard(
    n_players: int,
    n_iters: int,
    seed: int,
    worker_id: int,
    selfplay_every: int = 50_000,
) -> bytes:
    """
    Train one shard of iterations.  Returns pickled (strategy_sum, n_iters).
    Multiple shards are merged by the local entrypoint.
    """
    from perudo.m4 import CFRTrainer  # type: ignore[import]

    print(
        f"[{n_players}p w{worker_id}] debut — {n_iters:,} iters seed={seed}"
        f"  selfplay_every={selfplay_every}",
        flush=True,
    )
    trainer = CFRTrainer()
    chunk = 10_000
    done = 0
    chunk_idx = 0
    while done < n_iters:
        n = min(chunk, n_iters - done)
        trainer.train(
            n, n_players, seed=seed + chunk_idx * 7,
            selfplay_every=selfplay_every, verbose=False,
        )
        done += n
        chunk_idx += 1
    print(
        f"[{n_players}p w{worker_id}] termine — "
        f"{trainer.n_iters:,} iters  {len(trainer.strategy_sum):,} etats",
        flush=True,
    )
    return pickle.dumps((trainer.strategy_sum, trainer.n_iters))


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    iters: int = 1_000_000,
    players: str = "3,4,5,6",
    workers: int = N_WORKERS,
    no_selfplay: bool = False,
    cross_eval: bool = True,
    eval_games: int = 5000,
) -> None:
    """
    Lance workers x len(player_counts) containers en parallele.
    Fusionne les strategy_sum et sauvegarde les policies.

    Options:
        --iters         Iterations totales par player count (defaut: 1 000 000)
        --players       Comptes de joueurs separes par virgules (defaut: 3,4,5,6)
        --workers       Containers par player count (defaut: 8)
        --no-selfplay   Desactiver le self-play (adversaires = ThresholdBot fixe)
        --cross-eval    Evaluer nouveau vs ancien modele (defaut: True)
        --eval-games    Parties pour l'evaluation finale (defaut: 5 000)
    """
    import numpy as np

    from perudo.m4.cfr import cross_eval as do_cross_eval  # type: ignore[import]
    from perudo.m4.cfr import fast_eval  # type: ignore[import]
    from perudo.m4.policy import Policy  # type: ignore[import]

    player_counts = [int(p.strip()) for p in players.split(",")]
    iters_per_worker = max(1000, iters // workers)
    total_iters = iters_per_worker * workers
    selfplay_every = 0 if no_selfplay else 50_000
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    print("=" * 66)
    print("  CFR Training — Modal cloud (parallel workers)")
    print("=" * 66)
    print(f"  Joueurs       : {player_counts}")
    print(f"  Iterations    : {total_iters:,}  ({workers} w x {iters_per_worker:,})")
    print(f"  Containers    : {len(player_counts) * workers} en parallele")
    print(f"  Self-play     : {'desactive' if no_selfplay else f'tous les {selfplay_every:,} iters'}")
    print(f"  Cross-eval    : {'oui' if cross_eval else 'non'} ({eval_games} parties)")
    print("=" * 66 + "\n")

    # Build all jobs : (n_players, n_iters, seed, worker_id, selfplay_every)
    jobs = [
        (n, iters_per_worker, 42 + w * 1000, w, selfplay_every)
        for n in player_counts
        for w in range(workers)
    ]

    # Run all in parallel, collect shards
    shards_flat = list(train_shard.starmap(jobs))

    # Group shards by player count, merge strategy_sums, evaluate, save
    print("\n" + "=" * 66)
    for n_players in player_counts:
        indices = [
            i for i, (n, _, _, _, _) in enumerate(jobs) if n == n_players
        ]
        merged_sum: dict = {}
        merged_iters = 0
        for idx in indices:
            shard_sum, shard_iters = pickle.loads(shards_flat[idx])  # noqa: S301
            merged_iters += shard_iters
            for key, vec in shard_sum.items():
                if key in merged_sum:
                    merged_sum[key] = merged_sum[key] + vec
                else:
                    merged_sum[key] = vec.copy()

        policy_new = Policy(
            strategy_sum={k: np.array(v) for k, v in merged_sum.items()},
            n_iters=merged_iters,
        )

        # Evaluate new model vs ThresholdBot
        wr, lo, hi = fast_eval(policy_new, n_players, eval_games, seed=999)
        print(
            f"  [{n_players}p] vs ThresholdBot : "
            f"win={wr:.1%} [{lo:.1%},{hi:.1%}]"
            f"  {policy_new.n_states:,} etats  {merged_iters:,} iters"
        )

        # Cross-eval: new vs old model (if old model exists)
        old_path = out_dir / f"cfr_{n_players}p.pkl"
        if cross_eval and old_path.exists():
            try:
                policy_old = Policy.load(old_path)
                xwr, xlo, xhi = do_cross_eval(
                    policy_new, policy_old, n_players, eval_games, seed=998
                )
                arrow = "✓ MEILLEUR" if xwr > 0.50 else "✗ MOINS BON"
                print(
                    f"  [{n_players}p] vs ancien CFR  : "
                    f"win={xwr:.1%} [{xlo:.1%},{xhi:.1%}]  {arrow}"
                )
            except Exception as e:
                print(f"  [{n_players}p] cross-eval echoue : {e}")

        path = out_dir / f"cfr_{n_players}p.pkl"
        policy_new.save(path)
        print(f"  [{n_players}p] Modele sauvegarde : {path.name}")

    print("=" * 66)
    print("  Modeles prets. Relancez le serveur pour activer CFRBot.")
    print("=" * 66 + "\n")
