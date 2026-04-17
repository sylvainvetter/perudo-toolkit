"""
Centralised logging for the Perudo server.

Two outputs:
  server_logger      — human-readable lines in logs/server.log
                       (RotatingFileHandler: 5 MB × 3 backups + stdout mirror)
  log_game_event()   — structured JSONL in logs/games/{ROOM_CODE}.jsonl
                       (one file per game, naturally bounded in size)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent.parent   # project root
LOGS_DIR = _ROOT / "logs"
GAMES_DIR = LOGS_DIR / "games"
LOGS_DIR.mkdir(exist_ok=True)
GAMES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Server logger  (logs/server.log + stdout)
# ---------------------------------------------------------------------------

server_logger = logging.getLogger("perudo.server")
server_logger.setLevel(logging.INFO)
server_logger.propagate = False   # don't bubble up to uvicorn root

if not server_logger.handlers:
    _fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fmt_console = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Rotating file: 5 MB, keep 3 backups → max ~20 MB on disk
    _fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "server.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _fh.setFormatter(_fmt_file)
    server_logger.addHandler(_fh)

    # Console mirror (appears in `fly logs` and local terminal)
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt_console)
    server_logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Per-game JSONL logger
# ---------------------------------------------------------------------------

def log_game_event(room_code: str, event: dict[str, Any]) -> None:
    """
    Append one JSON line to logs/games/{room_code}.jsonl.

    Each line is a self-contained event dict — always includes:
      ts   — Unix timestamp (float, ms precision)
      room — room code
      type — event type string

    Never raises: errors are swallowed and logged to server_logger.
    """
    event.setdefault("ts", round(time.time(), 3))
    event["room"] = room_code
    path = GAMES_DIR / f"{room_code}.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        server_logger.warning("Game log write failed [%s]: %s", room_code, exc)
