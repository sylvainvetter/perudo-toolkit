"""
WebSocket handler for multiplayer games.

Single endpoint: WS /ws/{room_code}/{token}

Phases:
  lobby   — waiting for players; creator can add/remove bots, then start
  playing — turn-based game; server runs bots immediately after human actions
  paused  — a human disconnected; waits up to PAUSE_TIMEOUT_SECS for reconnect
  finished — game over
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from perudo.m2 import config_for_n_players
from perudo.m4.policy import Policy
from perudo.web.game_engine import (
    _apply_die_loss,
    _bids_to_list,
    _bot_action,
    _check_game_over,
    _log_entry,
    _next_active,
    _next_starter,
    _roll_dice,
    _start_new_round,
)
from perudo.core.rules import resolve_exact, resolve_liar
from perudo.core.types import Bid, RaiseBid, Liar, Exact

from perudo.web.multiplayer.room_manager import (
    PAUSE_TIMEOUT_SECS,
    Room,
    PlayerSlot,
    room_manager,
)
from perudo.web.logging_setup import log_game_event, server_logger

# ---------------------------------------------------------------------------
# AFK constants
# ---------------------------------------------------------------------------

AFK_WARNING_SECS = 90   # warn at 90 s
AFK_TIMEOUT_SECS = 120  # auto-play at 120 s


# ---------------------------------------------------------------------------
# Main WS endpoint
# ---------------------------------------------------------------------------


async def handle_ws(
    websocket: WebSocket,
    room_code: str,
    token: str,
    cfr_policies: dict[int, Policy],
) -> None:
    room = room_manager.get_room(room_code)
    if room is None:
        await websocket.close(code=4004, reason="Room introuvable")
        return

    slot = room.slot_by_token(token)
    if slot is None:
        await websocket.close(code=4001, reason="Token invalide")
        return

    if slot.is_bot:
        await websocket.close(code=4001, reason="Token bot")
        return

    await websocket.accept()

    async with room.lock:
        slot.ws = websocket
        slot.connected = True

        if room.phase == "paused" and room.paused_player_id == slot.player_id:
            # Reconnection — resume the game
            room.phase = "playing"
            room.pause_deadline = None
            room.paused_player_id = None
            server_logger.info("[%s] %s reconnected (game resumed)", room.code, slot.pseudo)
            log_game_event(room.code, {"type": "reconnect", "player_id": slot.player_id, "pseudo": slot.pseudo})
            await _broadcast(room, {
                "type": "game_resumed",
                "player_id": slot.player_id,
                "pseudo": slot.pseudo,
            })
            pseudos = {s.player_id: s.pseudo for s in room.slots}
            # Re-send current game state to reconnecting player
            await _send(websocket, {
                "type": "game_start",
                "state": _filter_state(room.game_state, slot.player_id),
                "your_player_id": slot.player_id,
                "pseudos": pseudos,
                "resumed": True,
            })
            _reschedule_afk(room, cfr_policies)
        elif room.phase == "playing":
            server_logger.info("[%s] %s reconnected (game in progress)", room.code, slot.pseudo)
            pseudos = {s.player_id: s.pseudo for s in room.slots}
            # Late reconnect — resend state
            await _send(websocket, {
                "type": "game_start",
                "state": _filter_state(room.game_state, slot.player_id),
                "your_player_id": slot.player_id,
                "pseudos": pseudos,
                "resumed": True,
            })
        else:
            # Normal lobby join
            server_logger.info("[%s] %s connected (lobby, %d/%d)", room.code, slot.pseudo, len(room.slots), room.n_seats)
            await _broadcast(room, {
                "type": "lobby_update",
                **room.lobby_payload(),
            })

    try:
        async for raw in _ws_iter(websocket):
            msg = json.loads(raw)
            await _dispatch(websocket, room, slot, msg, cfr_policies)
    except WebSocketDisconnect:
        pass
    finally:
        await _on_disconnect(room, slot, cfr_policies)


# ---------------------------------------------------------------------------
# Message dispatcher
# ---------------------------------------------------------------------------


async def _dispatch(
    ws: WebSocket,
    room: Room,
    slot: PlayerSlot,
    msg: dict[str, Any],
    cfr_policies: dict[int, Policy],
) -> None:
    t = msg.get("type")

    async with room.lock:
        if room.phase == "lobby":
            if t == "add_bot" and slot.token == room.creator_token:
                bot_type = msg.get("bot_type", "threshold")
                if bot_type not in ("threshold", "cfr"):
                    return
                new_slot = room_manager.add_bot(room, bot_type)
                if new_slot:
                    await _broadcast(room, {"type": "lobby_update", **room.lobby_payload()})

            elif t == "remove_bot" and slot.token == room.creator_token:
                pid = int(msg.get("slot", -1))
                if room_manager.remove_bot(room, pid):
                    await _broadcast(room, {"type": "lobby_update", **room.lobby_payload()})

            elif t == "kick" and slot.token == room.creator_token:
                pid = int(msg.get("player_id", -1))
                if pid == slot.player_id:
                    return  # can't kick yourself
                target = room.slot_by_id(pid)
                if target and not target.is_bot:
                    kicked_ws = target.ws
                    room.slots = [s for s in room.slots if s.player_id != pid]
                    for i, s in enumerate(room.slots):
                        s.player_id = i
                    await _broadcast(room, {"type": "lobby_update", **room.lobby_payload()})
                    if kicked_ws:
                        try:
                            await kicked_ws.send_text(json.dumps({
                                "type": "kicked",
                                "message": "Vous avez été expulsé de la room.",
                            }))
                            await kicked_ws.close()
                        except Exception:
                            pass

            elif t == "start_game" and slot.token == room.creator_token:
                if len(room.slots) < 2:
                    await _send(ws, {"type": "error", "message": "Il faut au moins 2 joueurs."})
                    return
                _init_game(room)
                room.phase = "playing"
                pseudos = {s.player_id: s.pseudo for s in room.slots}
                players_info = [
                    {"id": s.player_id, "pseudo": s.pseudo, "is_bot": s.is_bot, "bot_type": s.bot_type}
                    for s in room.slots
                ]
                server_logger.info(
                    "[%s] Game started — %d players: %s",
                    room.code,
                    len(room.slots),
                    ", ".join(s.pseudo for s in room.slots),
                )
                log_game_event(room.code, {
                    "type": "game_start",
                    "n_players": len(room.slots),
                    "players": players_info,
                })
                # Send personalised state to each human
                for s in room.human_slots():
                    if s.ws and s.connected:
                        await _send(s.ws, {
                            "type": "game_start",
                            "state": _filter_state(room.game_state, s.player_id),
                            "your_player_id": s.player_id,
                            "pseudos": pseudos,
                        })
                # Run bots if first player is a bot, then schedule AFK
                await _run_bots(room, cfr_policies)
                _reschedule_afk(room, cfr_policies)

        elif room.phase == "playing":
            if t == "action":
                current = room.game_state.get("current_player", -1)
                if current != slot.player_id:
                    return  # not your turn
                # Cancel AFK task — player acted
                _cancel_afk(room)
                action = msg.get("action", {})
                round_num = room.game_state.get("round_num", 1)
                log, round_result, game_over, winner = _apply_action(
                    room.game_state, slot.player_id, action
                )
                _log_action(room.code, round_num, slot.player_id, slot.pseudo, action, round_result)
                if round_result:
                    log_game_event(room.code, {
                        "type": "round_end",
                        "round": round_num,
                        "dice_counts": _log_dice_counts(room.game_state),
                    })
                if game_over:
                    room.phase = "finished"
                    server_logger.info(
                        "[%s] Game over — winner: %s (round %d)",
                        room.code, _pseudo(room, winner), room.game_state.get("round_num", "?"),
                    )
                    log_game_event(room.code, {
                        "type": "game_over",
                        "winner_id": winner,
                        "winner_pseudo": _pseudo(room, winner),
                        "total_rounds": room.game_state.get("round_num", 0),
                    })
                    await _broadcast(room, {
                        "type": "game_over",
                        "winner": winner,
                        "winner_pseudo": _pseudo(room, winner),
                        "log": log,
                        "round_result": round_result,
                    })
                else:
                    await _broadcast_game_update(room, log, round_result)
                    await _run_bots(room, cfr_policies)
                    _reschedule_afk(room, cfr_policies)

        elif room.phase == "finished":
            if t == "rematch" and slot.token == room.creator_token:
                new_room, token_map = await _create_rematch_room(room)
                for s in room.human_slots():
                    if s.ws and s.connected:
                        new_token = token_map.get(s.player_id, "")
                        if new_token:
                            await _send(s.ws, {
                                "type": "rematch",
                                "room_code": new_room.code,
                                "your_token": new_token,
                            })


# ---------------------------------------------------------------------------
# Game logic helpers
# ---------------------------------------------------------------------------


def _human_ids(room: Room) -> set[int]:
    return {s.player_id for s in room.slots if not s.is_bot}


def _init_game(room: Room) -> None:
    """Build initial game state from room slots."""
    n = len(room.slots)
    players = []
    for s in room.slots:
        players.append({
            "id": s.player_id,
            "dice": _roll_dice(5),
            "n_dice": 5,
            "exact_used": False,
        })

    bot_types = [
        s.bot_type if s.is_bot else "human"
        for s in sorted(room.slots, key=lambda x: x.player_id)
    ]

    room.game_state = {
        "players": players,
        "active": list(range(n)),
        "bids": [],
        "current_player": 0,
        "starter": 0,
        "round_num": 1,
        "perco": False,
        "bot_types": bot_types,    # "human" | "threshold" | "cfr"
        "n_players": n,
        "human_ids": list(_human_ids(room)),
    }


def _apply_action(
    state: dict[str, Any],
    player_id: int,
    action: dict[str, Any],
) -> tuple[list[dict], dict | None, bool, int | None]:
    """
    Apply one player action (raise / liar / exact).
    Mutates state in-place.
    Returns (log, round_result, game_over, winner).
    """
    log: list[dict] = []
    round_result: dict | None = None
    game_over = False
    winner: int | None = None

    active: list[int] = state["active"]
    perco: bool = state["perco"]
    bids = [Bid(b["quantity"], b["value"], b["player_id"]) for b in state["bids"]]

    t = action.get("type")

    if t == "raise":
        q = int(action["quantity"])
        v = int(action["value"])
        bids.append(Bid(quantity=q, value=v, player_id=player_id))
        log.append(_log_entry(player_id, f"Mise : {q} × {v}"))
        state["bids"] = _bids_to_list(bids)
        state["current_player"] = _next_active(active, player_id)

    elif t == "liar":
        if not bids:
            return log, None, False, None
        all_dice = [d for pid in active for d in state["players"][pid]["dice"]]
        result = resolve_liar(bids[-1], player_id, all_dice, percolateur=perco)
        loser = result.loser_id
        detail = (
            f"Total {bids[-1].value}s = {result.total_matching} — "
            + (_loser_label(state, loser) if loser is not None else "personne ne perd")
        )
        log.append(_log_entry(player_id, "Menteur !", detail))
        round_result = {
            "loser_id": loser, "gainer_id": None,
            "total": result.total_matching,
            "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
        }
        if loser is not None:
            _apply_die_loss(state, loser)
        state["bids"] = []
        state["active"] = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
        game_over, winner = _check_game_over(state["active"])
        if not game_over:
            starter = _next_starter(state["active"], loser)
            _start_new_round(state, state["active"], starter, log)
            state["current_player"] = starter

    elif t == "exact":
        if not bids:
            return log, None, False, None
        all_dice = [d for pid in active for d in state["players"][pid]["dice"]]
        result = resolve_exact(bids[-1], player_id, all_dice, percolateur=perco)
        state["players"][player_id]["exact_used"] = True
        gainer: int | None = None
        loser_e = result.loser_id
        if result.loser_id is None:
            if state["players"][player_id]["n_dice"] < 5:
                gainer = player_id
                state["players"][player_id]["dice"].append(0)
                state["players"][player_id]["n_dice"] += 1
                detail = (
                    f"Total {bids[-1].value}s = {result.total_matching} — "
                    f"{_player_label(state, player_id)} gagne 1 dé !"
                )
            else:
                detail = (
                    f"Total {bids[-1].value}s = {result.total_matching} — "
                    f"Exact ! (déjà au max, pas de dé gagné)"
                )
        else:
            _apply_die_loss(state, player_id)
            detail = (
                f"Total {bids[-1].value}s = {result.total_matching} — "
                f"{_player_label(state, player_id)} perd 1 dé"
            )
        log.append(_log_entry(player_id, "Exact !", detail))
        round_result = {
            "loser_id": loser_e, "gainer_id": gainer,
            "total": result.total_matching,
            "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
        }
        state["bids"] = []
        state["active"] = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
        game_over, winner = _check_game_over(state["active"])
        if not game_over:
            starter = player_id if player_id in state["active"] else state["active"][0]
            _start_new_round(state, state["active"], starter, log)
            state["current_player"] = starter

    if game_over:
        state["current_player"] = -1

    return log, round_result, game_over, winner


async def _run_bots(room: Room, cfr_policies: dict[int, Policy]) -> None:
    """
    Run bot turns (under room.lock already held by caller).
    Stops when current_player is a human or game is over.
    """
    state = room.game_state
    if state is None:
        return
    human_ids = set(state.get("human_ids", []))
    n_players = state["n_players"]
    cfg = config_for_n_players(n_players)
    policy = cfr_policies.get(n_players)
    rng = np.random.default_rng()

    all_log: list[dict] = []
    round_result: dict | None = None

    while True:
        current = state.get("current_player", -1)
        if current == -1 or current in human_ids:
            break
        if room.phase != "playing":
            break

        active: list[int] = state["active"]
        bids = [Bid(b["quantity"], b["value"], b["player_id"]) for b in state["bids"]]
        bid_q = bids[-1].quantity if bids else 0
        bid_v = bids[-1].value if bids else 0
        all_dice = [d for pid in active for d in state["players"][pid]["dice"]]
        total = len(all_dice)

        p = state["players"][current]
        bot_type = state["bot_types"][current]

        # Get bot policy
        bot_policy: Policy | None = None
        if bot_type == "cfr":
            bot_policy = policy

        bot_act = _bot_action(
            p, bid_q, bid_v, total, state["perco"],
            current, cfg, rng, bot_policy, len(bids),
        )

        bot_pseudo = _pseudo(room, current)
        round_num = state.get("round_num", 1)

        if isinstance(bot_act, RaiseBid):
            b = bot_act.bid
            bids.append(b)
            state["bids"] = _bids_to_list(bids)
            all_log.append(_log_entry(current, f"Mise : {b.quantity} × {b.value}"))
            state["current_player"] = _next_active(active, current)
            _log_action(room.code, round_num, current, bot_pseudo,
                        {"type": "raise", "quantity": b.quantity, "value": b.value})

        elif isinstance(bot_act, Liar):
            all_dice2 = [d for pid in active for d in state["players"][pid]["dice"]]
            result = resolve_liar(bids[-1], current, all_dice2, percolateur=state["perco"])
            loser = result.loser_id
            detail = (
                f"Total {bids[-1].value}s = {result.total_matching} — "
                + (_loser_label(state, loser) if loser is not None else "personne ne perd")
            )
            all_log.append(_log_entry(current, "Menteur !", detail))
            round_result = {
                "loser_id": loser, "gainer_id": None,
                "total": result.total_matching,
                "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
            }
            _log_action(room.code, round_num, current, bot_pseudo, {"type": "liar"}, round_result)
            if loser is not None:
                _apply_die_loss(state, loser)
            state["bids"] = []
            state["active"] = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
            log_game_event(room.code, {"type": "round_end", "round": round_num,
                                       "dice_counts": _log_dice_counts(state)})
            game_over, winner = _check_game_over(state["active"])
            if game_over:
                room.phase = "finished"
                server_logger.info("[%s] Game over — winner: %s (round %d)",
                                   room.code, _pseudo(room, winner), round_num)
                log_game_event(room.code, {"type": "game_over", "winner_id": winner,
                                           "winner_pseudo": _pseudo(room, winner),
                                           "total_rounds": round_num})
                await _broadcast(room, {
                    "type": "game_over",
                    "winner": winner,
                    "winner_pseudo": _pseudo(room, winner),
                    "log": all_log,
                    "round_result": round_result,
                })
                return
            starter = _next_starter(state["active"], loser)
            _start_new_round(state, state["active"], starter, all_log)
            state["current_player"] = starter
            # Broadcast round result mid-bot-sequence
            await _broadcast_game_update(room, all_log, round_result)
            all_log = []
            round_result = None

        else:  # Exact
            all_dice2 = [d for pid in active for d in state["players"][pid]["dice"]]
            result = resolve_exact(bids[-1], current, all_dice2, percolateur=state["perco"])
            state["players"][current]["exact_used"] = True
            gainer: int | None = None
            loser_e = result.loser_id
            if result.loser_id is None:
                if state["players"][current]["n_dice"] < 5:
                    gainer = current
                    state["players"][current]["dice"].append(0)
                    state["players"][current]["n_dice"] += 1
                    detail = (
                        f"Total {bids[-1].value}s = {result.total_matching} — "
                        f"{_player_label(state, current)} gagne 1 dé !"
                    )
                else:
                    detail = (
                        f"Total {bids[-1].value}s = {result.total_matching} — "
                        f"Exact ! (déjà au max, pas de dé gagné)"
                    )
            else:
                _apply_die_loss(state, current)
                detail = (
                    f"Total {bids[-1].value}s = {result.total_matching} — "
                    f"{_player_label(state, current)} perd 1 dé"
                )
            all_log.append(_log_entry(current, "Exact !", detail))
            round_result = {
                "loser_id": loser_e, "gainer_id": gainer,
                "total": result.total_matching,
                "bid_q": bids[-1].quantity, "bid_v": bids[-1].value,
            }
            _log_action(room.code, round_num, current, bot_pseudo, {"type": "exact"}, round_result)
            state["bids"] = []
            state["active"] = [pid for pid in active if state["players"][pid]["n_dice"] > 0]
            log_game_event(room.code, {"type": "round_end", "round": round_num,
                                       "dice_counts": _log_dice_counts(state)})
            game_over, winner = _check_game_over(state["active"])
            if game_over:
                room.phase = "finished"
                server_logger.info("[%s] Game over — winner: %s (round %d)",
                                   room.code, _pseudo(room, winner), round_num)
                log_game_event(room.code, {"type": "game_over", "winner_id": winner,
                                           "winner_pseudo": _pseudo(room, winner),
                                           "total_rounds": round_num})
                await _broadcast(room, {
                    "type": "game_over",
                    "winner": winner,
                    "winner_pseudo": _pseudo(room, winner),
                    "log": all_log,
                    "round_result": round_result,
                })
                return
            starter = current if current in state["active"] else state["active"][0]
            _start_new_round(state, state["active"], starter, all_log)
            state["current_player"] = starter
            await _broadcast_game_update(room, all_log, round_result)
            all_log = []
            round_result = None

    # Broadcast remaining log (bots raised, no challenge yet)
    if all_log or round_result:
        await _broadcast_game_update(room, all_log, round_result)
    elif not all_log:
        # Broadcast state even if no log (human must act)
        await _broadcast_game_update(room, [], None)


# ---------------------------------------------------------------------------
# AFK detection
# ---------------------------------------------------------------------------


def _cancel_afk(room: Room) -> None:
    """Cancel any running AFK task for this room."""
    if room.afk_task and not room.afk_task.done():
        room.afk_task.cancel()
    room.afk_task = None


def _reschedule_afk(room: Room, cfr_policies: dict) -> None:
    """Start a fresh AFK timer for the current human player (if any)."""
    _cancel_afk(room)
    state = room.game_state
    if state is None or room.phase != "playing":
        return
    current = state.get("current_player", -1)
    human_ids = set(state.get("human_ids", []))
    if current == -1 or current not in human_ids:
        return
    room.afk_task = asyncio.create_task(_afk_check(room, current, cfr_policies))


async def _afk_check(room: Room, player_id: int, cfr_policies: dict) -> None:
    """Background task: warn then auto-play an unresponsive player."""
    try:
        await asyncio.sleep(AFK_WARNING_SECS)
    except asyncio.CancelledError:
        return

    # Warning phase
    async with room.lock:
        if room.phase != "playing":
            return
        state = room.game_state
        if state is None or state.get("current_player") != player_id:
            return
        secs_left = AFK_TIMEOUT_SECS - AFK_WARNING_SECS
        server_logger.info("[%s] AFK warning — %s (%ds left)", room.code, _pseudo(room, player_id), secs_left)
        log_game_event(room.code, {"type": "afk_warning", "player_id": player_id,
                                   "pseudo": _pseudo(room, player_id), "seconds_left": secs_left})
        await _broadcast(room, {
            "type": "afk_warning",
            "player_id": player_id,
            "pseudo": _pseudo(room, player_id),
            "seconds_left": secs_left,
        })

    try:
        await asyncio.sleep(AFK_TIMEOUT_SECS - AFK_WARNING_SECS)
    except asyncio.CancelledError:
        return

    # Auto-play phase
    async with room.lock:
        if room.phase != "playing":
            return
        state = room.game_state
        if state is None or state.get("current_player") != player_id:
            return

        bids = state.get("bids", [])
        action = {"type": "liar"} if bids else {"type": "raise", "quantity": 1, "value": 2}

        pseudo = _pseudo(room, player_id)
        server_logger.info("[%s] AFK autoplay — %s forced %s", room.code, pseudo, action.get("type"))
        log_game_event(room.code, {"type": "afk_autoplay", "player_id": player_id,
                                   "pseudo": pseudo, "forced_action": action.get("type")})
        await _broadcast(room, {
            "type": "afk_autoplay",
            "player_id": player_id,
            "pseudo": pseudo,
        })

        round_num = state.get("round_num", 1)
        log, round_result, game_over, winner = _apply_action(state, player_id, action)
        _log_action(room.code, round_num, player_id, pseudo, action, round_result)
        if round_result:
            log_game_event(room.code, {"type": "round_end", "round": round_num,
                                       "dice_counts": _log_dice_counts(state)})
        if game_over:
            room.phase = "finished"
            server_logger.info("[%s] Game over (AFK) — winner: %s", room.code, _pseudo(room, winner))
            log_game_event(room.code, {"type": "game_over", "winner_id": winner,
                                       "winner_pseudo": _pseudo(room, winner),
                                       "total_rounds": state.get("round_num", 0)})
            await _broadcast(room, {
                "type": "game_over",
                "winner": winner,
                "winner_pseudo": _pseudo(room, winner),
                "log": log,
                "round_result": round_result,
            })
        else:
            await _broadcast_game_update(room, log, round_result)
            await _run_bots(room, cfr_policies)
            _reschedule_afk(room, cfr_policies)


# ---------------------------------------------------------------------------
# Rematch
# ---------------------------------------------------------------------------


async def _create_rematch_room(old_room: Room) -> tuple[Room, dict[int, str]]:
    """
    Clone old_room configuration into a fresh lobby.
    Returns (new_room, {old_player_id: new_token}).
    """
    creator_slot = old_room.slot_by_token(old_room.creator_token)
    creator_pseudo = creator_slot.pseudo if creator_slot else "Créateur"

    new_room, new_creator = await room_manager.create_room(creator_pseudo, old_room.n_seats)
    token_map: dict[int, str] = {0: new_creator.token}

    for s in sorted(old_room.slots, key=lambda x: x.player_id):
        if s.token == old_room.creator_token:
            continue
        if s.is_bot:
            async with new_room.lock:
                room_manager.add_bot(new_room, s.bot_type)
        else:
            result = await room_manager.join_room(new_room.code, s.pseudo)
            if not isinstance(result, str):
                _, new_slot = result
                token_map[s.player_id] = new_slot.token

    return new_room, token_map


# ---------------------------------------------------------------------------
# Disconnect / reconnect
# ---------------------------------------------------------------------------


async def _on_disconnect(room: Room, slot: PlayerSlot, cfr_policies: dict[int, Policy]) -> None:
    async with room.lock:
        slot.connected = False
        slot.ws = None
        _cancel_afk(room)

        server_logger.info("[%s] %s disconnected (phase=%s)", room.code, slot.pseudo, room.phase)
        if room.phase == "playing":
            room.phase = "paused"
            room.paused_player_id = slot.player_id
            room.pause_deadline = time.time() + PAUSE_TIMEOUT_SECS
            log_game_event(room.code, {
                "type": "disconnect",
                "player_id": slot.player_id,
                "pseudo": slot.pseudo,
                "resume_before": room.pause_deadline,
            })
            await _broadcast(room, {
                "type": "game_paused",
                "player_id": slot.player_id,
                "pseudo": slot.pseudo,
                "resume_before": room.pause_deadline,
            })
        elif room.phase == "lobby":
            # Remove the slot if not creator; if creator left, close room
            if slot.token == room.creator_token:
                server_logger.info("[%s] Room closed (creator left)", room.code)
                await _broadcast(room, {"type": "room_closed", "reason": "Le créateur a quitté."})
                await room_manager.remove_room(room.code)
            else:
                room.slots = [s for s in room.slots if s.token != slot.token]
                for i, s in enumerate(room.slots):
                    s.player_id = i
                await _broadcast(room, {"type": "lobby_update", **room.lobby_payload()})


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------


async def _broadcast(room: Room, msg: dict[str, Any]) -> None:
    """Send a message to all connected human WebSockets in the room."""
    dead = []
    for s in room.slots:
        if s.ws is not None and s.connected:
            try:
                await s.ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(s)
    for s in dead:
        s.connected = False
        s.ws = None


async def _broadcast_game_update(
    room: Room,
    log: list[dict],
    round_result: dict | None,
) -> None:
    """Send personalised game_update to each human (they see their own dice)."""
    state = room.game_state
    if state is None:
        return
    dead = []
    for s in room.slots:
        if s.is_bot or s.ws is None or not s.connected:
            continue
        try:
            await s.ws.send_text(json.dumps({
                "type": "game_update",
                "state": _filter_state(state, s.player_id),
                "log": log,
                "round_result": round_result,
            }))
        except Exception:
            dead.append(s)
    for s in dead:
        s.connected = False
        s.ws = None


async def _send(ws: WebSocket, msg: dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# State filtering (hide other players' dice)
# ---------------------------------------------------------------------------


def _filter_state(state: dict[str, Any], viewer_id: int) -> dict[str, Any]:
    """Return a copy of state with other players' dice hidden."""
    import copy
    s = copy.deepcopy(state)
    for p in s["players"]:
        if p["id"] != viewer_id:
            p["dice"] = []  # hidden
    return s


# ---------------------------------------------------------------------------
# WS iterator (yields raw text messages)
# ---------------------------------------------------------------------------


async def _ws_iter(ws: WebSocket):
    while True:
        try:
            data = await ws.receive_text()
            yield data
        except WebSocketDisconnect:
            return
        except Exception:
            return


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_action(
    room_code: str,
    round_num: int,
    player_id: int,
    pseudo: str,
    action: dict,
    round_result: dict | None = None,
) -> None:
    """Write one action event to the game's JSONL log."""
    event: dict = {
        "type": "action",
        "round": round_num,
        "player_id": player_id,
        "pseudo": pseudo,
        "action": action.get("type"),
    }
    if action.get("type") == "raise":
        event["quantity"] = action.get("quantity")
        event["value"] = action.get("value")
    if round_result:
        event["result"] = {
            "loser_id": round_result.get("loser_id"),
            "gainer_id": round_result.get("gainer_id"),
            "total_matching": round_result.get("total"),
            "bid_q": round_result.get("bid_q"),
            "bid_v": round_result.get("bid_v"),
        }
    log_game_event(room_code, event)


def _log_dice_counts(state: dict) -> dict[str, int]:
    """Return {player_id: n_dice} snapshot — appended to round_end events."""
    return {str(p["id"]): p["n_dice"] for p in state["players"]}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _player_label(state: dict, pid: int) -> str:
    human_ids = set(state.get("human_ids", []))
    if pid in human_ids:
        return f"Joueur {pid}"
    return f"Bot {pid}"


def _loser_label(state: dict, loser_id: int | None) -> str:
    if loser_id is None:
        return "personne ne perd"
    return f"{_player_label(state, loser_id)} perd 1 dé"


def _pseudo(room: Room, player_id: int | None) -> str:
    if player_id is None:
        return "?"
    s = room.slot_by_id(player_id)
    return s.pseudo if s else f"Joueur {player_id}"
