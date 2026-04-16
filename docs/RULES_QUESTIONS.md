# RULES_QUESTIONS.md — Points de règles arbitrés

> Ce document liste les points ambigus identifiés lors de l'analyse des règles (cf. section 2.8 du CLAUDE.md),
> avec la décision définitive prise par l'utilisateur. Il fait foi pour toute l'implémentation.

---

## Q-D — Surenchère en manche Percolateur

| | |
|---|---|
| **Hypothèse initiale** | Valeur figée toute la manche, seule la quantité augmente |
| **Décision** | **Surenchère libre** : les règles de surenchère standard s'appliquent intégralement |
| **Différence avec une manche normale** | Les dés de valeur `1` ne comptent **pas** comme jokers (ils ne valent que 1) |

**Règle d'implémentation** : `is_valid_raise(bid, prev_bid, percolateur=True)` utilise exactement les mêmes contraintes que `percolateur=False`, mais `joker_active = False` lors du décompte des dés satisfaisant une annonce.

---

## Q-E — Plafond de dés après « exact » réussi

| | |
|---|---|
| **Hypothèse initiale** | Plafond à 5 dés |
| **Décision** | **Plafond à 5 dés (confirmé)** |
| **Précisions** | - « Exact » n'est utilisable qu'**une seule fois par partie** par joueur → récupération max d'**1 dé** au total<br>- Précondition : le joueur doit avoir **déjà perdu au moins 1 dé** pour pouvoir en récupérer un<br>- On ne peut jamais dépasser 5 dés |

**Règle d'implémentation** :
```
player.exact_used = True  # après usage
player.dice_count = min(player.dice_count + 1, 5)  # si exact réussi
```

---

## Q-F — Qui ouvre la manche suivante après « exact » réussi ?

| | |
|---|---|
| **Hypothèse initiale** | L'annonceur d'« exact » |
| **Décision** | **L'annonceur d'« exact » ouvre la manche suivante** (hypothèse confirmée) |

**Règle d'implémentation** : `round.starter_id = exact_caller_id` en cas d'exact réussi.

---

## Q-G — Surenchère à valeur supérieure : la quantité peut-elle rester égale ?

| | |
|---|---|
| **Hypothèse initiale** | Oui (`q' ≥ q` suffit quand `v' > v`) |
| **Décision** | **Oui, confirmé** — `(3 six)` est une surenchère valide après `(3 cinq)` |

**Règle d'implémentation** :
```
# Surenchère standard (v ≠ 1 → v' ≠ 1)
valid = (q' > q) OR (q' >= q AND v' > v)
```

---

## Récapitulatif des règles de surenchère (toutes variantes)

| Cas | Condition de validité |
|---|---|
| Standard → Standard | `q' > q` OU (`q' ≥ q` ET `v' > v`) |
| Standard → Perudo (`v'=1`) | `q' ≥ ceil(q / 2) + 1` |
| Perudo → Perudo | `q' > q` |
| Perudo → Standard (`v'≠1`) | `q' ≥ 2·q + 1` |
| Percolateur (toute transition) | Mêmes règles ci-dessus, `joker_active = False` |

---

*Document généré lors du démarrage S0 — 2026-04-16*
