# Bilan S5 — Module M2 : recommandeur d'action

*Date : 2026-04-16*

## Ce qui est fait

- [x] **`src/perudo/m2/recommender.py`** — logique de décision v1 :
  - `RecommenderConfig(threshold_liar, threshold_exact, lambda_risk)` — paramètres calibrables
  - `enumerate_valid_raises(prev_bid, player_id, total_dice)` — 1 candidat par valeur (min q)
  - `recommend(game_state, config)` → `Recommendation` avec `best_action`, `bid_if_raise`, `rationale`
  - Logique : Liar si p_true < seuil → Exact si p_exact > seuil ET dispo → Raise meilleur score

- [x] **30 tests** (137 au total, 0 échec) — couverture **95 %** sur M2
  - Opening bid, liar, exact, raise, Percolateur, enumerate, rationale
  - Vérification que chaque raise recommandé est légal (`is_valid_raise`)

- [x] **`docs/m2.md`** — documentation API

## Décisions prises

| Décision | Raison |
|---|---|
| 1 candidat par valeur (min_q) | Maximise p_true par valeur, suffisant pour v1 ; élargissable pour S7 |
| Score = `p_true - λ * max(0, q - E[T])` | λ=0 → pur p_true ; λ>0 → pénalise les enchères audacieuses |
| Tie-break ouverture par `own_count` | Annoncer ce qu'on tient est la stratégie dominante à ouverture |
| `min_value=1e-9` dans les tests Hypothesis | Scipy overflow sur p ≤ 2e-308 (floats normaux extrêmes hors domaine de jeu) |

## Limites connues (S7 calibration)

- Seuils par défaut (0.25 / 0.35) non calibrés — seront ajustés via M3
- Un seul candidat par valeur : stratégie peut être améliorée (S7)
- `lambda_risk = 0` par défaut : aucune pénalité risque

## Ce qui reste

- [ ] S6 : Module M3 — simulateur Monte Carlo + stratégies + stats 10k parties
- [ ] S7 : Calibration des seuils M2
- [ ] S8 : Documentation finale

## Commandes de vérification

```bash
python -m ruff check src/ tests/
python -m mypy
python -m pytest -v
```
