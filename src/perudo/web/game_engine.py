"""
Game engine for the /play mode.

Converts between JSON-serializable state dicts and game logic.
All state is held client-side; each API call receives the full state
and returns the updated state after processing one human action +
all subsequent bot actions (until it's the human's turn again).

State dict keys:
  players      : list of {id, dice, n_dice, exact_used}
                 (bot dice are included so the server can make decisions)
  active       : sorted list of active player ids
  bids         : list of {quantity, value, player_id}
  current_player: whose turn it is (0 = human)
  starter      : who opened this round
  round_num    : current round number (1-indexed)
  perco        : percolateur active (bool)
  bot_types    : list of bot type strings for bots 1..n ("threshold" | "cfr")
  n_players    : total player count
"""

from __future__ import annotations

import random as _random
from typing import Any

import numpy as np

from perudo.core.rules import resolve_exact, resolve_liar
from perudo.core.types import Bid, Exact, Liar, RaiseBid
from perudo.m2 import config_for_n_players
from perudo.m4.cfr import _fast_action
from perudo.m4.infostate import (
    decode_action,
    legal_mask,
    make_info_key,
    make_opening_key,
)
from perudo.m4.policy import Policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_active(active: list[int], from_id: int) -> int:
    """Return next player id after from_id in the active list (circular)."""
    try:
        idx = active.index(from_id)
    except ValueError:
        candidates = [x for x in sorted(active) if x > from_id]
        return candidates[0] if candidates else active[0]
    return active[(idx + 1) % len(active)]


def _bot_action(
    player_state: dict[str, Any],
    bid_q: int,
    bid_v: int,
    total: int,
    perco: bool,
    pid: int,
    cfg: Any,
    rng: np.random.Generator,
    policy: Policy | None,
    n_bids: int,
) -> Liar | Exact | RaiseBid:
    """
    Get one bot action using CFR policy (if available) or ThresholdBot fallback.
    """
    face_counts = np.bincount(
        np.array(player_state["dice"], dtype=np.int32), minlength=7
    )[1:].astype(np.int32)
    n_own = int(player_state["n_dice"])

    if policy is not None:
        if bid_q == 0:
            info_key: tuple = make_opening_key(face_counts, total, perco)
            mask = legal_mask(0, 0, total, exact_avail=False)
        else:
            exact_avail = not player_state["exact_used"]
            info_key = make_info_key(
                player_state["dice"], bid_q, bid_v, total, exact_avail, perco, n_bids
            )
            mask = legal_mask(bid_q, bid_v, total, exact_avail)

        if policy.knows(info_key):
            probs = policy.get_probs(info_key, mask)
            legal = np.where(mask)[0]
            lp = probs[legal]
            s = lp.sum()
            lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
            idx = int(rng.choice(legal, p=lp))
            return decode_action(idx, bid_q, bid_v, pid)

    # Fallback: ThresholdBot
    return _fast_action(
        face_counts, n_own,
        bid_q, bid_v, total,
        bool(player_state["exact_used"]), perco, pid, cfg,
    )


def _roll_dice(n: int) -> list[int]:
    return sorted(_random.randint(1, 6) for _ in range(n))


def _bids_to_list(bids: list[Bid]) -> list[dict[str, int]]:
    return [{"quantity": b.quantity, "value": b.value, "player_id": b.player_id}
            for b in bids]


def _log_entry(
    player_id: int,
    action_label: str,
    detail: str = "",
    entry_type: str = "action",
) -> dict[str, Any]:
    label = "Vous" if player_id == 0 else f"Bot {player_id}"
    return {
        "type": entry_type,
        "player_id": player_id,
        "player_label": label,
        "action_label": action_label,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_game(
    n_bots: int,
    bot_types: list[str],
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """
    Create initial game state.

    Player 0 is always the human. Bots are players 1..n_bots.
    Player 0 always opens the first round.
    """
    if rng_seed is None:
        rng_seed = _random.randint(0, 2**31)
    _random.seed(rng_seed)

    n_players = 1 + n_bots
    players: list[dict[str, Any]] = []
    for i in range(n_players):
        players.append({
            "id": i,
            "dice": _roll_dice(5),
            "n_dice": 5,
            "exact_used": False,
        })

    return {
        "players": players,
        "active": list(range(n_players)),
        "bids": [],
        "current_player": 0,
        "starter": 0,
        "round_num": 1,
        "perco": False,
        "bot_types": bot_types,
        "n_players": n_players,
    }


def process_action(  # noqa: C901 — game loop is inherently complex
    state: dict[str, Any],
    action: dict[str, Any],
    cfr_policies: dict[int, Policy],
) -> dict[str, Any]:
    """
    Process one human action and run all subsequent bot actions.

    Continues until:
    - It's the human's turn (current_player == 0), OR
    - The game is over

    Returns:
        state         : updated game state
        log           : list of log entries (all actions this call)
        round_result  : {loser_id, gainer_id, total, bid_q, bid_v} or None
        game_over     : bool
        winner        : int or None
    """
    import copy

    state = copy.deepcopy(state)
    log: list[dict[str, Any]] = []
    rng = np.random.default_rng()

    n_players: int = state["n_players"]
    active: list[int] = list(state["active"])
    perco: bool = bool(state["perco"])
    cfg = config_for_n_players(n_players)
    n_policy = cfr_policies.get(n_players)

    bot_types: list[str] = state["bot_types"]

    def get_policy(pid: int) -> Policy | None:
        if pid == 0:
            return None
        bot_idx = pid - 1
        if bot_idx < len(bot_types) and bot_types[bot_idx] == "cfr":
            return n_policy
        return None

    # Reconstruct Bid objects from state
    bids: list[Bid] = [
        Bid(quantity=b["quantity"], value=b["value"], player_id=b["player_id"])
        for b in state["bids"]
    ]

    round_result: dict[str, Any] | None = None
    game_over = False
    winner: int | None = None

    # ------------------------------------------------------------------ #
    # Step 1: apply the human's action
    # ------------------------------------------------------------------ #
    assert state["current_player"] == 0, "process_action called out of turn"

    bid_q = bids[-1].quantity if bids else 0
    bid_v = bids[-1].value if bids else 0
    current: int

    if action["type"] == "raise":
        q = int(action["quantity"])
        v = int(action["value"])
        new_bid = Bid(quantity=q, value=v, player_id=0)
        bids.append(new_bid)
        log.append(_log_entry(0, f"Mise : {q} × {v}"))
        current = _next_active(active, 0)

    elif action["type"] == "liar":
        all_dice = [d for pid in active for d in state["players"][pid]["dice"]]
        result = resolve_liar(bids[-1], 0, all_dice, percolateur=perco)
        loser_id = result.loser_id
        detail = (
            f"Total {bids[-1].value}s = {result.total_matching} — "
            + ("vous perdez 1 dé" if loser_id == 0
               else f"Bot {loser_id} perd 1 dé" if loser_id is not None
               else "personne ne perd")
        )
        log.append(_log_entry(0, "Menteur !", detail))
        round_result = {
            "loser_id": loser_id, "gainer_id": None,
            "total": result.total_matching,
            "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
        }
        bids = []
        if loser_id is not None:
            _apply_die_loss(state, loser_id)
        active = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
        game_over, winner = _check_game_over(active)
        if not game_over:
            starter = _next_starter(active, loser_id)
            _start_new_round(state, active, starter, log)
            current = starter
        else:
            current = -1

    else:  # exact
        all_dice = [d for pid in active for d in state["players"][pid]["dice"]]
        result = resolve_exact(bids[-1], 0, all_dice, percolateur=perco)
        state["players"][0]["exact_used"] = True
        gainer_id: int | None = None
        loser_id_e: int | None = result.loser_id

        if result.loser_id is None:
            if state["players"][0]["n_dice"] < 5:
                gainer_id = 0
                state["players"][0]["dice"].append(0)
                state["players"][0]["n_dice"] += 1
                detail = (
                    f"Total {bids[-1].value}s = {result.total_matching} — "
                    "vous gagnez 1 dé !"
                )
            else:
                detail = (
                    f"Total {bids[-1].value}s = {result.total_matching} — "
                    "Exact ! (déjà au max, pas de dé gagné)"
                )
        else:
            _apply_die_loss(state, 0)
            detail = (
                f"Total {bids[-1].value}s = {result.total_matching} — "
                "vous perdez 1 dé"
            )

        log.append(_log_entry(0, "Exact !", detail))
        round_result = {
            "loser_id": loser_id_e, "gainer_id": gainer_id,
            "total": result.total_matching,
            "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
        }
        bids = []
        active = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
        game_over, winner = _check_game_over(active)
        if not game_over:
            starter = 0 if 0 in active else active[0]
            _start_new_round(state, active, starter, log)
            current = starter
        else:
            current = -1

    # ------------------------------------------------------------------ #
    # Step 2: run bots until human's turn or game over
    # ------------------------------------------------------------------ #
    while not game_over and current != 0:
        p = state["players"][current]
        all_dice_list = [d for pid in active for d in state["players"][pid]["dice"]]
        total = len(all_dice_list)
        bid_q_b = bids[-1].quantity if bids else 0
        bid_v_b = bids[-1].value if bids else 0

        bot_act = _bot_action(
            p, bid_q_b, bid_v_b, total, perco, current, cfg, rng,
            get_policy(current), len(bids),
        )

        if isinstance(bot_act, RaiseBid):
            b = bot_act.bid
            bids.append(b)
            log.append(_log_entry(current, f"Mise : {b.quantity} × {b.value}"))
            current = _next_active(active, current)

        elif isinstance(bot_act, Liar):
            all_dice_list = [d for pid in active for d in state["players"][pid]["dice"]]
            result_b = resolve_liar(bids[-1], current, all_dice_list, percolateur=perco)
            loser_id_b = result_b.loser_id
            detail_b = (
                f"Total {bids[-1].value}s = {result_b.total_matching} — "
                + ("vous perdez 1 dé" if loser_id_b == 0
                   else f"Bot {loser_id_b} perd 1 dé" if loser_id_b is not None
                   else "personne ne perd")
            )
            log.append(_log_entry(current, "Menteur !", detail_b))
            round_result = {
                "loser_id": loser_id_b, "gainer_id": None,
                "total": result_b.total_matching,
                "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
            }
            bids = []
            if loser_id_b is not None:
                _apply_die_loss(state, loser_id_b)
            active = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
            game_over, winner = _check_game_over(active)
            if not game_over:
                starter_b = _next_starter(active, loser_id_b)
                _start_new_round(state, active, starter_b, log)
                current = starter_b

        else:  # Exact
            all_dice_list = [d for pid in active for d in state["players"][pid]["dice"]]
            result_e = resolve_exact(bids[-1], current, all_dice_list, percolateur=perco)
            state["players"][current]["exact_used"] = True
            gainer_id_b: int | None = None
            loser_id_e2: int | None = result_e.loser_id

            if result_e.loser_id is None:
                if state["players"][current]["n_dice"] < 5:
                    gainer_id_b = current
                    state["players"][current]["dice"].append(0)
                    state["players"][current]["n_dice"] += 1
                    detail_e = (
                        f"Total {bids[-1].value}s = {result_e.total_matching} — "
                        f"Bot {current} gagne 1 dé !"
                    )
                else:
                    detail_e = (
                        f"Total {bids[-1].value}s = {result_e.total_matching} — "
                        f"Exact ! (déjà au max, pas de dé gagné)"
                    )
            else:
                _apply_die_loss(state, current)
                detail_e = (
                    f"Total {bids[-1].value}s = {result_e.total_matching} — "
                    f"Bot {current} perd 1 dé"
                )

            log.append(_log_entry(current, "Exact !", detail_e))
            round_result = {
                "loser_id": loser_id_e2, "gainer_id": gainer_id_b,
                "total": result_e.total_matching,
                "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
            }
            bids = []
            active = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
            game_over, winner = _check_game_over(active)
            if not game_over:
                starter_e = current if current in active else active[0]
                _start_new_round(state, active, starter_e, log)
                current = starter_e

    # ------------------------------------------------------------------ #
    # Finalise state
    # ------------------------------------------------------------------ #
    state["bids"] = _bids_to_list(bids)
    state["current_player"] = current if not game_over else -1
    state["active"] = active

    return {
        "state": state,
        "log": log,
        "round_result": round_result,
        "game_over": game_over,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_die_loss(state: dict[str, Any], player_id: int) -> None:
    p = state["players"][player_id]
    if p["dice"]:
        p["dice"] = p["dice"][:-1]
    p["n_dice"] = len(p["dice"])


def _check_game_over(active: list[int]) -> tuple[bool, int | None]:
    if len(active) <= 1:
        return True, (active[0] if active else None)
    return False, None


def _next_starter(active: list[int], loser_id: int | None) -> int:
    if loser_id is not None and loser_id in active:
        return loser_id
    return active[0]


def _start_new_round(
    state: dict[str, Any],
    active: list[int],
    starter: int,
    log: list[dict[str, Any]],
) -> None:
    """Roll new dice for all active players and reset round state."""
    for pid in active:
        n = state["players"][pid]["n_dice"]
        state["players"][pid]["dice"] = _roll_dice(n)

    state["bids"] = []
    state["current_player"] = starter
    state["starter"] = starter
    state["round_num"] = state.get("round_num", 1) + 1

    log.append({
        "type": "round_start",
        "player_id": -1,
        "player_label": "—",
        "action_label": f"Manche {state['round_num']}",
        "detail": "",
    })
