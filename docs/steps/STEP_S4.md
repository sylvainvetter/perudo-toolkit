# Bilan S4 — Module M1 : calculateur de probabilité

*Date : 2026-04-16*

## Ce qui est fait

- [x] **`src/perudo/m1/calc.py`** — toutes les fonctions spécifiées section 4.1 :
  - `p_per_die(value, *, percolateur)` — probabilité par dé inconnu (1/6 ou 2/6)
  - `p_at_least(q, n, p, own_count)` → P(T ≥ q) via `scipy.stats.binom.sf`
  - `p_exactly(q, n, p, own_count)` → P(T == q) via `scipy.stats.binom.pmf`
  - `expected_count(n, p, own_count)` → E[T] = own_count + n·p
  - `distribution(n, p, own_count)` → vecteur numpy PMF complet
  - `bid_stats(bid, own_dice, n_unknown, *, percolateur)` → `BidStats` (API haut niveau)

- [x] **39 tests** (107 au total, 0 échec) — couverture **100 %** sur M1
  - Invariants 1-5 de la section 4.1 vérifiés par Hypothesis
  - Test de performance : `p_at_least` < 1 ms pour n ≤ 30 ✓
  - Tests d'intégration `bid_stats` couvrant normal play, Percolateur, Perudo (valeur=1)

- [x] **`docs/m1.md`** — documentation API avec exemples

## Décisions prises

| Décision | Raison |
|---|---|
| `allow_subnormal=False` dans les tests Hypothesis | Les floats dénormalisés (p ≈ 1e-308) déclenchent un overflow dans `scipy.stats.binom`. Hors domaine de jeu réel (p ∈ {1/6, 2/6}) |
| `NDArray[np.float64]` comme type de retour | Mypy strict l'exige ; `numpy.typing.NDArray` est la forme correcte |
| `scipy-stubs` ajouté en dépendance de dev | Nécessaire pour mypy strict sur les imports scipy |

## Ce qui reste

- [ ] S5 : Module M2 — recommandeur d'action (consomme M1)
- [ ] S6 : Module M3 — simulateur Monte Carlo
- [ ] S7 : Calibration des seuils M2
- [ ] S8 : Documentation finale

## Commandes de vérification

```bash
python -m ruff check src/ tests/
python -m mypy
python -m pytest -v
```
