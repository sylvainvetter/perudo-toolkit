# Bilan S2 — Squelette projet + CI

*Date : 2026-04-16*

## Ce qui est fait

- [x] Stack validée : **Python 3.12**, scipy, numpy, pytest, hypothesis, ruff, mypy
- [x] Structure de répertoires créée :
  ```
  src/perudo/{core,m1,m2,m3}/
  tests/{test_core,test_m1,test_m2,test_m3}/
  docs/
  .github/workflows/
  ```
- [x] `pyproject.toml` — configuration unifiée (build, dépendances, ruff, mypy, pytest, coverage)
- [x] `README.md` — description, installation, usage rapide, structure
- [x] `.github/workflows/ci.yml` — pipeline : lint → format check → mypy → pytest

## Ce qui reste

- [ ] S3 : types (`Die`, `Player`, `Bid`, `Action`, `RoundState`, `GameState`) + moteur de règles + tests unitaires
- [ ] S4 : Module M1 (calculateur de probabilité) + tests
- [ ] S5 : Module M2 (recommandeur) + tests
- [ ] S6 : Module M3 (simulateur Monte Carlo) + stratégies
- [ ] S7 : Calibration des seuils M2
- [ ] S8 : Documentation finale (`USAGE.md`, exemples)

## Décisions prises

| Décision | Raison |
|---|---|
| `hatchling` comme build backend | Plus simple que setuptools, pas de `setup.py` |
| `ruff` pour lint + format | Remplace flake8/black/isort en un seul outil, très rapide |
| `hypothesis` comme lib property-based | Meilleure intégration pytest, strategies personnalisables |
| `--cov-fail-under=0` pour l'instant | Le seuil sera relevé progressivement à chaque étape |
| Pas de git init automatique | À faire manuellement ou via la prochaine session |

## Comment lancer la CI localement

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
mypy
pytest
```
