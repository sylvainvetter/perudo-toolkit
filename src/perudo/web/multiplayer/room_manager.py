"""
Room manager for online multiplayer.

Holds all active rooms in memory (no persistence across server restarts).
Each Room has an asyncio.Lock to guard concurrent WebSocket access.
"""

from __future__ import annotations

import random
import string
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOM_CODE_LENGTH = 6
ROOM_CODE_CHARS = string.ascii_uppercase + string.digits
MAX_PLAYERS = 6
MIN_PLAYERS = 2
PAUSE_TIMEOUT_SECS = 300   # 5 minutes to reconnect
FINISHED_TTL_SECS = 600    # clean up finished rooms after 10 min


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlayerSlot:
    player_id: int           # 0-based index in the game
    pseudo: str
    is_bot: bool
    bot_type: str | None     # "threshold" | "cfr" | None (for humans)
    token: str               # secret UUID, stored in client localStorage
    ws: "WebSocket | None"   # None for bots or disconnected humans
    connected: bool


@dataclass
class Room:
    code: str
    slots: list[PlayerSlot]
    n_seats: int             # total seats (humans + bots), set when room created
    game_state: dict[str, Any] | None
    phase: str               # "lobby" | "playing" | "paused" | "finished"
    creator_token: str       # token of the player who created the room
    created_at: float
    pause_deadline: float | None  # epoch timestamp; None when not paused
    paused_player_id: int | None  # which human disconnected
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    afk_task: "asyncio.Task | None" = field(default=None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def slot_by_token(self, token: str) -> PlayerSlot | None:
        for s in self.slots:
            if s.token == token:
                return s
        return None

    def slot_by_id(self, player_id: int) -> PlayerSlot | None:
        for s in self.slots:
            if s.player_id == player_id:
                return s
        return None

    def human_slots(self) -> list[PlayerSlot]:
        return [s for s in self.slots if not s.is_bot]

    def connected_ws(self) -> list["WebSocket"]:
        return [s.ws for s in self.slots if s.ws is not None and s.connected]

    def is_full(self) -> bool:
        """True when all n_seats are occupied (human or bot)."""
        return len(self.slots) >= self.n_seats

    def lobby_payload(self) -> dict[str, Any]:
        """JSON-serialisable snapshot for lobby_update broadcasts."""
        return {
            "code": self.code,
            "n_seats": self.n_seats,
            "phase": self.phase,
            "slots": [
                {
                    "player_id": s.player_id,
                    "pseudo": s.pseudo,
                    "is_bot": s.is_bot,
                    "bot_type": s.bot_type,
                    "connected": s.connected,
                }
                for s in self.slots
            ],
        }


# ---------------------------------------------------------------------------
# RoomManager singleton
# ---------------------------------------------------------------------------


class RoomManager:
    """Thread-safe (asyncio) in-memory store for all active rooms."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._global_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Room lifecycle
    # ------------------------------------------------------------------

    async def create_room(self, pseudo: str, n_seats: int) -> tuple[Room, PlayerSlot]:
        """
        Create a new room and add the creator as player 0.

        Returns (room, creator_slot).
        """
        async with self._global_lock:
            code = self._unique_code()
            token = _new_token()
            creator = PlayerSlot(
                player_id=0,
                pseudo=pseudo,
                is_bot=False,
                bot_type=None,
                token=token,
                ws=None,
                connected=False,
            )
            room = Room(
                code=code,
                slots=[creator],
                n_seats=n_seats,
                game_state=None,
                phase="lobby",
                creator_token=token,
                created_at=time.time(),
                pause_deadline=None,
                paused_player_id=None,
            )
            self._rooms[code] = room
            return room, creator

    async def join_room(
        self, code: str, pseudo: str
    ) -> tuple[Room, PlayerSlot] | str:
        """
        Add a human player to an existing lobby.

        Returns (room, slot) on success, or an error string.
        """
        code = code.upper().strip()
        room = self._rooms.get(code)
        if room is None:
            return "Room introuvable."
        async with room.lock:
            if room.phase != "lobby":
                return "La partie a déjà commencé."
            if room.is_full():
                return "La room est complète."
            # Reject duplicate pseudo
            existing = [s.pseudo.lower() for s in room.slots]
            if pseudo.lower() in existing:
                return f'Le pseudo "{pseudo}" est déjà pris.'
            token = _new_token()
            slot = PlayerSlot(
                player_id=len(room.slots),
                pseudo=pseudo,
                is_bot=False,
                bot_type=None,
                token=token,
                ws=None,
                connected=False,
            )
            room.slots.append(slot)
            return room, slot

    def get_room(self, code: str) -> Room | None:
        return self._rooms.get(code.upper())

    async def remove_room(self, code: str) -> None:
        async with self._global_lock:
            self._rooms.pop(code.upper(), None)

    # ------------------------------------------------------------------
    # Slot management (called under room.lock by ws_handler)
    # ------------------------------------------------------------------

    def add_bot(self, room: Room, bot_type: str) -> PlayerSlot | None:
        """Add a bot to the next empty seat. Returns slot or None if full."""
        if room.is_full():
            return None
        slot = PlayerSlot(
            player_id=len(room.slots),
            pseudo=f"Bot {len(room.slots)}",
            is_bot=True,
            bot_type=bot_type,
            token=_new_token(),
            ws=None,
            connected=True,   # bots are always "connected"
        )
        room.slots.append(slot)
        return slot

    def remove_bot(self, room: Room, player_id: int) -> bool:
        """Remove a bot slot. Returns True if removed."""
        for i, s in enumerate(room.slots):
            if s.player_id == player_id and s.is_bot:
                room.slots.pop(i)
                # Renumber remaining slots
                for j, remaining in enumerate(room.slots):
                    remaining.player_id = j
                return True
        return False

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    async def sweep_stale(self) -> None:
        """Remove finished / expired rooms. Call from a background task."""
        now = time.time()
        async with self._global_lock:
            to_delete = []
            for code, room in self._rooms.items():
                if room.phase == "finished" and (now - room.created_at) > FINISHED_TTL_SECS:
                    to_delete.append(code)
                # Also kill lobbies older than 2 hours with no activity
                elif room.phase == "lobby" and (now - room.created_at) > 7200:
                    to_delete.append(code)
                # Kill paused rooms past their deadline
                elif (
                    room.phase == "paused"
                    and room.pause_deadline is not None
                    and now > room.pause_deadline
                ):
                    room.phase = "finished"
            for code in to_delete:
                del self._rooms[code]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _unique_code(self) -> str:
        while True:
            code = "".join(random.choices(ROOM_CODE_CHARS, k=ROOM_CODE_LENGTH))
            if code not in self._rooms:
                return code


def _new_token() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Module-level singleton (imported by app.py and ws_handler.py)
# ---------------------------------------------------------------------------

room_manager = RoomManager()
