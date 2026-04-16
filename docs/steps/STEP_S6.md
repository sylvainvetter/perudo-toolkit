# Bilan S6 — Module M3 : simulateur Monte Carlo

*Date : 2026-04-16*

## Ce qui est fait

- [x] **`src/perudo/m3/strategies.py`** — 3 stratégies pluggables :
  - `RandomLegal` — choisit uniformément parmi les actions légales
  - `Honest` — enchère l'espérance arrondie de la meilleure valeur ; `wants_percolateur=False`
  - `ThresholdBot` — délègue au recommandeur M2 ; nom dynamique avec les seuils

- [x] **`src/perudo/m3/simulator.py`** — boucle de jeu complète :
  - `_SimPlayer` — état mutable par joueur (dés, exact_used)
  - `_run_single_game` — 1 partie complète : lancer des dés, Percolateur, tours, résolution
  - `run_simulation(n_games, strategies, *, seed, output_dir)` → `SimulationResults`
  - Règles : Liar (perdant ouvre), Exact (annonceur ouvre — Q-F), élimination, Percolateur

- [x] **`src/perudo/m3/reporter.py`** — types de sortie et génération de rapports :
  - `BidRecord`, `GameRecord` — enregistrement par tour/partie
  - `StrategyStats` avec `win_rate` + `wilson_ci()` (intervalle de Wilson 95 %)
  - `SimulationResults` — agrégation automatique dans `__post_init__`
  - `write_csv` → `game_records.csv` + `bid_records.csv`
  - `write_markdown` → `summary.md`

- [x] **`src/perudo/m3/__init__.py`** — re-exports complets

- [x] **19 tests** (156 au total, 0 échec) — couverture **96 %** sur le module entier
  - Invariants : un seul gagnant, reproductibilité, légalité des enchères,
    structure des manches, CI Wilson, performances, sorties fichiers

- [x] **Simulation 10 000 parties** (3 joueurs, seed=42) → `results/`

## Résultats 10 000 parties

| Stratégie | Victoires | Taux | IC Wilson 95 % |
|---|---:|---:|---|
| RandomLegal | 1 065 | 10,7 % | [10,1 %, 11,3 %] |
| Honest | 4 053 | 40,5 % | [39,6 %, 41,5 %] |
| ThresholdBot | 4 882 | 48,8 % | [47,8 %, 49,8 %] |

Durée moyenne d'une partie : **13,1 tours** — Temps total : 99 s pour 10 000 parties.

## Décisions prises

| Décision | Raison |
|---|---|
| Percolateur = starter avec 1 dé + `wants_percolateur=True` | Règle Q-D : l'ouvreur décide |
| Perdant ouvre la manche suivante (Liar) | Règle standard Perudo |
| Annonceur Exact ouvre toujours | Q-F, qu'il ait gagné ou perdu |
| Wilson CI pour les taux de victoire | Plus fiable que ±σ/√n pour les proportions extrêmes |
| `seed=42` reproductible | Permet de comparer les stratégies de façon déterministe |

## Limites connues

- **Performance** : 99 s pour 10k parties vs. cible 60 s (spec). La boucle Python pure
  est le goulot ; une vectorisation numpy complète permettrait de réduire à ~10 s (S7/S8).
- **ThresholdBot vs Honest** : ThresholdBot domine (48,8 % vs 40,5 %). Les seuils M2
  (threshold_liar=0.50, threshold_exact=0.40) semblent déjà bien calibrés.
- `RandomLegal` (10,7 %) confirme que la stratégie aléatoire est nettement sous-optimale.

## Ce qui reste

- [ ] S7 : Calibration des seuils M2 via grille (threshold_liar × threshold_exact)
- [ ] S8 : Documentation finale

## Commandes de vérification

```bash
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
python -m mypy
python -m pytest -v
```
