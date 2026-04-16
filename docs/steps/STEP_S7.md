# Bilan S7 — Calibration des seuils M2

*Date : 2026-04-16*

## Ce qui est fait

- [x] **`src/perudo/m3/calibrator.py`** — moteur de grid-search :
  - `calibrate(n_games, liar_values, exact_values, opponents, seed)` → `CalibrationResults`
  - `write_calibration_csv` / `write_calibration_report` (heatmap ASCII + top 10)

- [x] **Grid search** : 81 configs × 500 parties = 40 500 parties (~5 min)
  - Adversaires : [Honest(), Honest()]
  - Grille : threshold_liar ∈ [0.30…0.70 pas 0.05], threshold_exact ∈ [0.15…0.55 pas 0.05]

- [x] **Validation** : top 5 configs à 2 000 parties

- [x] **Test de robustesse** sur 3 scénarios (2 000 parties chacun) :
  - Scénario A : vs [Honest, Honest]
  - Scénario B : vs [ThresholdBot(0.50), Honest]
  - Scénario C : vs [ThresholdBot(0.50), ThresholdBot(0.50)]

- [x] **Mise à jour** `RecommenderConfig` : défaut → liar=0.30, exact=0.40

## Résultats

### Meilleure configuration : liar=0.30, exact=0.40

| Scénario | liar=0.30 | liar=0.40 | liar=0.45 | liar=0.50 (ancien défaut) |
|---|---|---|---|---|
| vs [Honest, Honest] | **59.4 %** | 50.5 % | 43.9 % | 43.6 % |
| vs [ThresholdBot(0.50), Honest] | **30.0 %** | 27.2 % | 20.5 % | 20.1 % |
| vs [ThresholdBot(0.50), ThresholdBot(0.50)] | **54.9 %** | 45.6 % | 34.0 % | 34.7 % |

**liar=0.30 domine dans les 3 scénarios** — c'est la config robuste.

### Validation top 5 (2 000 parties, seed=1234, vs [Honest, Honest])

| threshold_liar | threshold_exact | Win rate | IC 95 % |
|---|---|---|---|
| **0.30** | **0.40** | **59.5 %** | [57.3 %, 61.6 %] |
| 0.30 | 0.50 | 57.9 % | [55.7 %, 60.0 %] |
| 0.30 | 0.35 | 57.9 % | [55.7 %, 60.0 %] |
| 0.30 | 0.30 | 55.6 % | [53.5 %, 57.8 %] |
| 0.35 | 0.40 | 50.5 % | [48.3 %, 52.7 %] |

## Interprétation

**Pourquoi threshold_liar=0.30 gagne-t-il mieux que le Nash break-even (0.50) ?**

Le seuil Nash de 0.50 suppose un adversaire qui bid de façon optimale et aléatoire.
Honest bid systématiquement l'espérance arrondie de ses dés — ce qui signifie que ses
enchères sont vraies environ 50 % du temps mais souvent à la limite. En contestant
dès que P(vrai) < 0.30, ThresholdBot met une pression maximale sur ces bids marginaux
et gagne des dés bien plus souvent qu'il n'en perd.

**threshold_exact=0.40 confirmé** — inchangé par rapport au défaut initial.

## Décisions prises

| Décision | Raison |
|---|---|
| Nouveau défaut liar=0.30 | Optimal dans les 3 scénarios testés, gain +15 pp vs 0.50 |
| exact=0.40 maintenu | Validé dans le top 5, aucun autre seuil exact n'améliore significativement |
| lambda_risk=0 inchangé | Hors scope S7 ; gain marginal attendu |

## Ce qui reste

- [ ] S8 : Documentation finale
- [ ] Future : self-play par renforcement (potentiel 65-70 % vs humains)
- [ ] Future : modélisation des adversaires (Bayésien)
