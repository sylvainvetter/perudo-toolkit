"""
FastAPI web application — Perudo Toolkit.

Routes:
  GET  /               → lobby (online multiplayer landing)
  GET  /solo           → solo game (human vs bots)
  GET  /toolkit        → advisor page (toolkit landing)
  GET  /proba          → probability calculator page
  GET  /sim            → simulation results page
  WS   /ws/{room}/{tok}→ WebSocket multiplayer

  POST /api/room/create → create multiplayer room
  POST /api/room/join   → join existing room
  POST /api/game/start  → start solo game, return initial state
  POST /api/game/action → process one solo action + bot responses
  POST /api/recommend   → recommendation JSON
  POST /api/stats       → bid probability stats JSON
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from perudo.core.types import Bid, GameState, Player, RoundState
from perudo.m1 import bid_stats
from perudo.m2 import recommend
from perudo.m3 import Honest, RandomLegal, ThresholdBot, run_simulation
from perudo.m3.strategies import Strategy
from perudo.m4 import CFRBot, Policy

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Perudo Toolkit", docs_url="/api/docs")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# CFR policy cache  (loaded once at startup from models/cfr_Np.pkl)
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
_cfr_policies: dict[int, Policy] = {}


def _load_cfr_policies() -> None:
    for n in range(2, 7):
        path = _MODELS_DIR / f"cfr_{n}p.pkl"
        if path.exists():
            try:
                _cfr_policies[n] = Policy.load(path)
                n_states = _cfr_policies[n].n_states
                print(f"[CFR] modele charge : {path.name}  ({n_states:,} etats)")
            except Exception as exc:
                print(f"[CFR] echec chargement {path.name}: {exc}")


_load_cfr_policies()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BidIn(BaseModel):
    quantity: Annotated[int, Field(ge=1, le=30)]
    value: Annotated[int, Field(ge=1, le=6)]


class OpponentIn(BaseModel):
    dice_count: Annotated[int, Field(ge=1, le=5)]


class RecommendRequest(BaseModel):
    own_dice: Annotated[list[int], Field(min_length=1, max_length=5)]
    current_bid: BidIn | None = None
    opponents: Annotated[list[OpponentIn], Field(min_length=1, max_length=8)]
    percolateur: bool = False
    exact_used: bool = False


class StatsRequest(BaseModel):
    own_dice: Annotated[list[int], Field(min_length=1, max_length=5)]
    bid: BidIn
    n_unknown: Annotated[int, Field(ge=0, le=40)]
    percolateur: bool = False


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def page_lobby(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "lobby.html")


@app.get("/toolkit", response_class=HTMLResponse)
async def page_advisor(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "advisor.html")


@app.get("/proba", response_class=HTMLResponse)
async def page_proba(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "proba.html")


@app.get("/sim", response_class=HTMLResponse)
async def page_sim(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "sim.html")


@app.get("/solo", response_class=HTMLResponse)
@app.get("/play", response_class=HTMLResponse)   # keep old URL working
async def page_play(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "play.html")


@app.get("/room/{room_code}", response_class=HTMLResponse)
async def page_room(request: Request, room_code: str) -> HTMLResponse:
    return templates.TemplateResponse(request, "room.html", {"room_code": room_code.upper()})


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


def _build_game_state(req: RecommendRequest) -> GameState:
    """Build a GameState from the advisor form payload."""
    own_dice = [int(d) for d in req.own_dice]
    player_id = 0
    players: list[Player] = [
        Player(id=player_id, dice=own_dice, exact_used=req.exact_used)
    ]
    for i, opp in enumerate(req.opponents):
        # Opponent dice values unknown — fill with zeros (count only matters)
        players.append(Player(id=i + 1, dice=[3] * opp.dice_count, exact_used=False))

    current_bid: Bid | None = None
    if req.current_bid is not None:
        # last_bidder = any opponent (doesn't matter for probability)
        current_bid = Bid(
            quantity=req.current_bid.quantity,
            value=req.current_bid.value,
            player_id=1,
        )

    bids = [current_bid] if current_bid is not None else []
    round_state = RoundState(
        bids=bids,
        current_player_id=player_id,
        percolateur=req.percolateur,
        starter_id=player_id,
    )
    return GameState(
        players=players,
        round=round_state,
        turn_order=[p.id for p in players],
    )


@app.post("/api/recommend")
async def api_recommend(req: RecommendRequest) -> dict:  # type: ignore[type-arg]
    state = _build_game_state(req)
    rec = recommend(state)

    result: dict = {"action": rec.best_action}  # type: ignore[type-arg]

    # Bid suggested if raise
    if rec.bid_if_raise is not None:
        result["bid"] = {
            "quantity": rec.bid_if_raise.quantity,
            "value": rec.bid_if_raise.value,
        }
    else:
        result["bid"] = None

    # Probabilities about the current bid on the table
    rat = rec.rationale
    result["p_true"] = round(rat.p_current_bid_true, 4)
    result["p_exact"] = round(rat.p_current_bid_exact, 4)
    result["expected"] = round(rat.expected_total, 2)
    result["score"] = round(rat.score, 4)

    # Top alternatives
    result["alternatives"] = [
        {
            "quantity": c.bid.quantity,
            "value": c.bid.value,
            "p_true": round(c.stats.p_true, 4),
            "score": round(c.score, 4),
        }
        for c in sorted(rat.alternatives_considered, key=lambda c: -c.score)[:4]
    ]

    return result


@app.post("/api/stats")
async def api_stats(req: StatsRequest) -> dict:  # type: ignore[type-arg]
    own_dice = [int(d) for d in req.own_dice]
    b = Bid(quantity=req.bid.quantity, value=req.bid.value, player_id=1)
    stats = bid_stats(b, own_dice, req.n_unknown, percolateur=req.percolateur)
    return {
        "p_true": round(stats.p_true, 4),
        "p_exact": round(stats.p_exact, 4),
        "expected": round(stats.expected, 2),
        "own_count": stats.own_count,
        "p_die": round(stats.p_die, 4),
    }


class RoomCreateRequest(BaseModel):
    pseudo: Annotated[str, Field(min_length=1, max_length=20)]
    n_seats: Annotated[int, Field(ge=2, le=6)] = 4


class RoomJoinRequest(BaseModel):
    room_code: Annotated[str, Field(min_length=4, max_length=8)]
    pseudo: Annotated[str, Field(min_length=1, max_length=20)]


@app.post("/api/room/create")
async def api_room_create(req: RoomCreateRequest) -> dict:  # type: ignore[type-arg]
    from perudo.web.multiplayer.room_manager import room_manager as rm

    room, slot = await rm.create_room(req.pseudo, req.n_seats)
    return {
        "room_code": room.code,
        "player_token": slot.token,
        "player_id": slot.player_id,
    }


@app.post("/api/room/join")
async def api_room_join(req: RoomJoinRequest) -> dict:  # type: ignore[type-arg]
    from fastapi import HTTPException
    from perudo.web.multiplayer.room_manager import room_manager as rm

    result = await rm.join_room(req.room_code, req.pseudo)
    if isinstance(result, str):
        raise HTTPException(status_code=400, detail=result)
    room, slot = result
    return {
        "room_code": room.code,
        "player_token": slot.token,
        "player_id": slot.player_id,
    }


@app.websocket("/ws/{room_code}/{token}")
async def ws_multiplayer(websocket: WebSocket, room_code: str, token: str) -> None:
    from perudo.web.multiplayer.ws_handler import handle_ws

    await handle_ws(websocket, room_code, token, _cfr_policies)


class GameStartRequest(BaseModel):
    n_bots: Annotated[int, Field(ge=1, le=5)]
    bot_types: Annotated[list[str], Field(min_length=1, max_length=5)]
    rng_seed: int | None = None


class GameActionRequest(BaseModel):
    state: dict
    action: dict


@app.post("/api/game/start")
async def api_game_start(req: GameStartRequest) -> dict:  # type: ignore[type-arg]
    from perudo.web.game_engine import start_game

    if len(req.bot_types) != req.n_bots:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="bot_types length must match n_bots")

    state = start_game(req.n_bots, req.bot_types, req.rng_seed)
    return {"state": state, "log": []}


@app.post("/api/game/action")
async def api_game_action(req: GameActionRequest) -> dict:  # type: ignore[type-arg]
    from perudo.web.game_engine import process_action

    return process_action(req.state, req.action, _cfr_policies)


class SimulateRequest(BaseModel):
    strategies: Annotated[list[str], Field(min_length=2, max_length=6)]
    n_games: Annotated[int, Field(ge=10, le=2000)] = 500


_STRATEGY_MAP: dict[str, type[Strategy]] = {
    "random": RandomLegal,
    "honest": Honest,
    "threshold": ThresholdBot,
}

_STRATEGY_NAMES: dict[str, str] = {
    "random": "RandomLegal",
    "honest": "Honest",
    "threshold": "ThresholdBot",
}


@app.get("/api/cfr_models")
async def api_cfr_models() -> dict:  # type: ignore[type-arg]
    """Return which player counts have a loaded CFR model."""
    return {
        "available": sorted(_cfr_policies.keys()),
        "models": {
            str(n): {"n_states": p.n_states, "n_iters": p.n_iters}
            for n, p in _cfr_policies.items()
        },
    }


@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest) -> dict:  # type: ignore[type-arg]
    from fastapi import HTTPException

    n_players = len(req.strategies)
    strategies: list[Strategy] = []
    for key in req.strategies:
        if key == "cfr":
            policy = _cfr_policies.get(n_players)
            if policy is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Pas de modele CFR pour {n_players} joueurs. "
                        f"Lancez : python scripts/train_cfr.py --players {n_players}"
                    ),
                )
            strategies.append(CFRBot(policy))
        else:
            cls = _STRATEGY_MAP.get(key)
            if cls is None:
                raise HTTPException(
                    status_code=422, detail=f"Strategie inconnue: {key}"
                )
            strategies.append(cls())

    names = " vs ".join(s.name for s in strategies)
    print(f"\n[Simulation] {req.n_games} parties — {names}")
    results = run_simulation(req.n_games, strategies, seed=None, verbose=True)

    stats = []
    for s in results.strategy_stats:
        lo, hi = s.wilson_ci()
        stats.append(
            {
                "name": s.name,
                "wins": s.wins,
                "win_rate": round(s.win_rate, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
        )

    return {
        "n_games": results.n_games,
        "avg_rounds": round(results.avg_rounds, 1),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    uvicorn.run("perudo.web.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
