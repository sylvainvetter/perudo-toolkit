# Bilan S3 — Core : types + moteur de règles + tests

*Date : 2026-04-16*

## Ce qui est fait

- [x] **`src/perudo/core/types.py`** — tous les types du domaine :
  - `Bid(quantity, value, player_id)` — annonce (frozen dataclass)
  - `RaiseBid`, `Liar`, `Exact` — variantes d'action (frozen dataclasses)
  - `type Action = RaiseBid | Liar | Exact` — union discriminée (Python 3.12 `type` keyword)
  - `Player(id, dice, exact_used)` — avec propriétés `dice_count` et `eliminated`
  - `RoundState(bids, current_player_id, percolateur, starter_id)` — avec propriété `current_bid`
  - `GameState(players, round, turn_order)`

- [x] **`src/perudo/core/rules.py`** — moteur de règles :
  - `count_matching(dice, value, *, joker_active)` — comptage avec/sans jokers
  - `is_valid_bid(bid)` — validité structurelle
  - `is_valid_opening(bid)` — toute annonce valide peut ouvrir (y compris Perudo)
  - `is_valid_raise(new_bid, prev_bid)` — les 4 transitions : std→std, std→perudo, perudo→perudo, perudo→std
  - `resolve_liar(bid, challenger_id, all_dice, *, percolateur)` → `ResolutionResult`
  - `resolve_exact(bid, caller_id, all_dice, *, percolateur)` → `ResolutionResult`

- [x] **68 tests** (68 passed, 0 failed) — couverture **100 %** sur le core
  - Tests unitaires pour chaque fonction et chaque transition de surenchère
  - Tests de propriété Hypothesis : formule exhaustive std→std, std→perudo, monotonie, invariants resolve_liar
  - CI locale complète : ruff check ✓ / ruff format ✓ / mypy strict ✓ / pytest ✓

## Décisions prises

| Décision | Raison |
|---|---|
| `type Action = ...` (PEP 695) | ruff UP040 l'impose sur Python 3.12 ; plus lisible que TypeAlias |
| `joker_active` en kwarg-only (`*`) | Évite les erreurs d'ordre d'argument sur un bool |
| `loser_id: int | None` dans `ResolutionResult` | `None` signifie « exact réussi, pas de perdant » — sémantique explicite |
| Règles de surenchère identiques en Percolateur | Confirmé Q-D : seul le comptage change, pas la validation |

## Ce qui reste

- [ ] S4 : Module M1 — calculateur de probabilité (`p_at_least`, `p_exactly`, `expected_count`, `distribution`)
- [ ] S5 : Module M2 — recommandeur d'action
- [ ] S6 : Module M3 — simulateur Monte Carlo
- [ ] S7 : Calibration des seuils M2
- [ ] S8 : Documentation finale

## Commandes de vérification

```bash
python -m ruff check src/ tests/
python -m ruff format --check src/ tests/
python -m mypy
python -m pytest tests/test_core/ -v
```
