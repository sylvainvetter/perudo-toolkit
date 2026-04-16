# STACK.md — Choix de stack technique

> Comparatif des options envisagées pour le projet Perudo Toolkit.
> **Validation humaine requise avant de passer à S2.**

---

## Critères d'évaluation

| Critère | Poids | Détail |
|---|---|---|
| Vitesse de développement | Élevé | Projet exploratoire, itérations fréquentes |
| Écosystème stats/proba | Élevé | M1 repose sur la loi binomiale ; M3 sur la simulation |
| Qualité des outils de test | Élevé | > 85 % de couverture, tests de propriété (invariants probabilistes) |
| Performance runtime | Moyen | M1 < 1 ms ; M3 10k parties en < 60 s |
| Typage statique | Moyen | Aide à modéliser `GameState`, `Action`, variants |

---

## Option A — Python 3.12+ ✅ *Recommandée*

### Dépendances principales

| Rôle | Lib |
|---|---|
| Distribution binomiale (M1) | `scipy.stats.binom` |
| Calcul vectorisé (M3) | `numpy` |
| Tests unitaires | `pytest` |
| Tests de propriété | `hypothesis` |
| Linter + formatter | `ruff` |
| Vérification de types | `mypy` |

### Structure projet

```
perudo/
├── src/
│   └── perudo/
│       ├── core/       # Types, règles, validation
│       ├── m1/         # Calculateur de probabilité
│       ├── m2/         # Recommandeur
│       └── m3/         # Simulateur Monte Carlo
├── tests/
├── docs/
├── pyproject.toml      # Config unifiée (ruff, mypy, pytest)
└── README.md
```

### Analyse par critère

| Critère | Évaluation | Commentaire |
|---|---|---|
| Vitesse de dev | ★★★★★ | Syntaxe concise, REPL, itération rapide |
| Stats/proba | ★★★★★ | `scipy.stats.binom.sf` couvre tout M1 nativement en 1 ligne |
| Tests de propriété | ★★★★★ | `hypothesis` — leader du domaine, strategies personnalisables |
| Performance | ★★★★☆ | M1 < 0.1 ms avec scipy ; M3 < 60 s atteignable avec numpy |
| Typage | ★★★★☆ | `mypy` + dataclasses/`@dataclass` ou `pydantic` pour les types |

### Points forts spécifiques au projet

- `scipy.stats.binom.sf(q-1, n, p)` = P(X ≥ q) — implémente `p_at_least` en une ligne
- `hypothesis.strategies` peut générer des `GameState` valides pour tester l'invariant de surenchère
- `numpy` permet de vectoriser la simulation de N parties simultanément dans M3

---

## Option B — TypeScript (Node.js 20+)

### Dépendances principales

| Rôle | Lib |
|---|---|
| Tests | `jest` + `ts-jest` |
| Tests de propriété | `fast-check` |
| Linter | `eslint` + `prettier` |
| Types | natif TypeScript |
| Stats | *(pas d'équivalent scipy — à implémenter)* |

### Analyse par critère

| Critère | Évaluation | Commentaire |
|---|---|---|
| Vitesse de dev | ★★★★☆ | Bon, mais toolchain plus lourde |
| Stats/proba | ★★☆☆☆ | Pas de lib binomiale mature ; calcul CDF à écrire manuellement |
| Tests de propriété | ★★★★☆ | `fast-check` solide |
| Performance | ★★★★☆ | Comparable à Python pour la simulation |
| Typage | ★★★★★ | Union types natifs, idéal pour `Action = Raise | Liar | Exact` |

**Inconvénient bloquant** : l'absence d'équivalent scipy implique d'implémenter la CDF binomiale from scratch — source d'erreurs pour un module critique (M1).

---

## Option C — Rust (édition 2021)

### Dépendances principales

| Rôle | Crate |
|---|---|
| Stats | `statrs` |
| RNG reproductible | `rand` + `rand_chacha` |
| Tests de propriété | `proptest` |
| Benchmarks | `criterion` |

### Analyse par critère

| Critère | Évaluation | Commentaire |
|---|---|---|
| Vitesse de dev | ★★☆☆☆ | Borrow checker, compilation lente, overhead élevé |
| Stats/proba | ★★★☆☆ | `statrs` correct mais moins mature que scipy |
| Tests de propriété | ★★★★☆ | `proptest` bon |
| Performance | ★★★★★ | M3 pourrait tourner en < 5 s pour 10k parties |
| Typage | ★★★★★ | Enums algébriques natifs, pattern matching exhaustif |

**Inconvénient** : pour un projet exploratoire avec calibration itérative, la lenteur de développement Rust est un frein significatif.

---

## Tableau comparatif résumé

| Critère | Python | TypeScript | Rust |
|---|---|---|---|
| Vitesse de dev | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| Écosystème stats | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| Tests de propriété | ★★★★★ | ★★★★☆ | ★★★★☆ |
| Performance M1 | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Performance M3 | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Typage | ★★★★☆ | ★★★★★ | ★★★★★ |
| **Score global** | **29/30** | **23/30** | **22/30** |

---

## Recommandation

**Option A — Python** est la stack recommandée pour ce projet.

Le critère décisif est l'écosystème stats : `scipy.stats.binom` implémente nativement toutes les fonctions de M1 (PMF, CDF, SF) avec une précision numérique validée. Pour un projet centré sur le calcul probabiliste et la simulation, ne pas repartir de zéro sur la distribution binomiale est un avantage majeur.

Les objectifs de performance (M1 < 1 ms, M3 < 60 s) sont atteignables sans Rust.

---

## Décision

- [x] **Option A — Python** *(choisie — 2026-04-16)*
- [ ] Option B — TypeScript
- [ ] Option C — Rust

*À cocher par l'utilisateur avant de passer à S2.*

---

*Document généré lors du démarrage S1 — 2026-04-16*
