# Perudo Toolkit

Boîte à outils pour le jeu de dés **Perudo** — probabilités, conseiller d'action, simulateur Monte Carlo, et interface web.

## Lancer l'application

```bash
pip install -e .
perudo
# → http://localhost:8000
```

Ou sans installation :

```bash
pip install -e .
uvicorn perudo.web.app:app --reload
```

## Pages

| Page | URL | Description |
|---|---|---|
| **Conseiller** | `/` | Entre tes dés et le bid en cours → reçois la meilleure action |
| **Probabilités** | `/proba` | Analyse détaillée d'une enchère (P(vrai), P(exact), espérance) |
| **Simulation** | `/sim` | Compare des stratégies sur N parties simulées |

## Modules

| Module | Rôle |
|---|---|
| **M1** `perudo.m1` | Calculateur de probabilité (loi binomiale) |
| **M2** `perudo.m2` | Recommandeur d'action avec seuils calibrables |
| **M3** `perudo.m3` | Simulateur Monte Carlo + stratégies |
| **Web** `perudo.web` | Interface FastAPI / Jinja2 |

## Résultats de référence (10 000 parties, 3 joueurs)

| Stratégie | Taux de victoire | IC 95 % |
|---|---|---|
| RandomLegal | 10,7 % | [10,1 %, 11,3 %] |
| Honest | 40,5 % | [39,6 %, 41,5 %] |
| ThresholdBot | 48,8 % | [47,8 %, 49,8 %] |

## Installation développement

```bash
pip install -e ".[dev]"

# Vérifications
ruff check src/ tests/
mypy
pytest
```

## Déploiement (Render.com)

1. Connecter le repo GitHub sur [render.com](https://render.com)
2. Type : **Web Service** · Runtime : **Python 3**
3. Build command : `pip install -e .`
4. Start command : `uvicorn perudo.web.app:app --host 0.0.0.0 --port $PORT`
5. Chaque `git push` redéploie automatiquement.

## Structure

```
src/perudo/
├── core/        Types de domaine, moteur de règles
├── m1/          Calculateur de probabilité
├── m2/          Recommandeur d'action
├── m3/          Simulateur Monte Carlo + stratégies
└── web/         Interface FastAPI (templates Jinja2 + Tailwind + Alpine.js)
```
