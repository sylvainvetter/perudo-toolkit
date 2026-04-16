"""
M3 — Monte Carlo game simulator for Perudo.

Public API:
    run_simulation(n_games, strategies, *, seed, output_dir) → SimulationResults
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from perudo.core.rules import resolve_exact, resolve_liar
from perudo.core.types import (
    Bid,
    GameState,
    Liar,
    Player,
    RaiseBid,
    RoundState,
)
from perudo.m3.reporter import (
    BidRecord,
    GameRecord,
    SimulationResults,
    write_csv,
    write_markdown,
)
from perudo.m3.strategies import Strategy

# ---------------------------------------------------------------------------
# Internal mutable player state
# ---------------------------------------------------------------------------


@dataclass
class _SimPlayer:
    id: int
    strategy: Strategy
    dice: list[int] = field(default_factory=list)
    exact_used: bool = False

    @property
    def dice_count(self) -> int:
        return len(self.dice)

    @property
    def eliminated(self) -> bool:
        return len(self.dice) == 0

    def roll(self, rng: np.random.Generator) -> None:
        self.dice = list(map(int, rng.integers(1, 7, size=len(self.dice))))

    def to_player(self) -> Player:
        return Player(id=self.id, dice=list(self.dice), exact_used=self.exact_used)


# ---------------------------------------------------------------------------
# Game state builder
# ---------------------------------------------------------------------------


def _build_state(
    sim_players: dict[int, _SimPlayer],
    active_ids: list[int],
    bids: list[Bid],
    current_player_id: int,
    percolateur: bool,
    starter_id: int,
) -> GameState:
    players = [sim_players[pid].to_player() for pid in active_ids]
    return GameState(
        players=players,
        round=RoundState(
            bids=list(bids),
            current_player_id=current_player_id,
            percolateur=percolateur,
            starter_id=starter_id,
        ),
        turn_order=list(active_ids),
    )


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------


def _next_active(active_ids: list[int], from_id: int) -> int:
    """Return the next active player after *from_id* in the circular order."""
    if not active_ids:
        return from_id
    try:
        idx = active_ids.index(from_id)
    except ValueError:
        # from_id was just eliminated — find the closest successor
        all_ids = sorted(active_ids)
        for candidate in all_ids:
            if candidate > from_id:
                return candidate
        return all_ids[0]
    return active_ids[(idx + 1) % len(active_ids)]


def _run_single_game(
    strategies: list[Strategy],
    rng: np.random.Generator,
    game_id: int,
) -> tuple[GameRecord, list[BidRecord]]:
    sim_players = {
        i: _SimPlayer(id=i, strategy=strategies[i], dice=[0] * 5)
        for i in range(len(strategies))
    }
    active_ids: list[int] = list(range(len(strategies)))
    bid_records: list[BidRecord] = []
    n_rounds = 0
    starter_id = 0

    while len(active_ids) > 1:
        n_rounds += 1

        # Roll dice for all active players
        for pid in active_ids:
            sim_players[pid].roll(rng)

        # Determine Percolateur: starter has 1 die and opts in
        starter = sim_players[starter_id]
        percolateur = False
        if starter.dice_count == 1:
            temp_state = _build_state(
                sim_players, active_ids, [], starter_id, False, starter_id
            )
            percolateur = starter.strategy.wants_percolateur(temp_state)

        # --- Bidding round ---
        bids: list[Bid] = []
        current_pid = starter_id
        turn_in_round = 0
        next_starter = starter_id  # default; overwritten on resolution

        while True:
            turn_in_round += 1
            state = _build_state(
                sim_players, active_ids, bids, current_pid, percolateur, starter_id
            )
            action = sim_players[current_pid].strategy.choose_action(state, rng)

            if isinstance(action, RaiseBid):
                bids.append(action.bid)
                bid_records.append(
                    BidRecord(
                        game_id=game_id,
                        round_id=n_rounds,
                        turn_id=turn_in_round,
                        player_id=current_pid,
                        action_type="raise",
                        quantity=action.bid.quantity,
                        value=action.bid.value,
                    )
                )
                current_pid = _next_active(active_ids, current_pid)

            elif isinstance(action, Liar):
                bid_records.append(
                    BidRecord(
                        game_id=game_id,
                        round_id=n_rounds,
                        turn_id=turn_in_round,
                        player_id=current_pid,
                        action_type="liar",
                        quantity=None,
                        value=None,
                    )
                )
                all_dice = [d for pid in active_ids for d in sim_players[pid].dice]
                result = resolve_liar(
                    bids[-1], current_pid, all_dice, percolateur=percolateur
                )
                loser_id = result.loser_id
                assert loser_id is not None
                loser = sim_players[loser_id]
                loser.dice = loser.dice[:-1]
                next_starter = loser_id
                break

            else:  # Exact
                bid_records.append(
                    BidRecord(
                        game_id=game_id,
                        round_id=n_rounds,
                        turn_id=turn_in_round,
                        player_id=current_pid,
                        action_type="exact",
                        quantity=None,
                        value=None,
                    )
                )
                sim_players[current_pid].exact_used = True
                all_dice = [d for pid in active_ids for d in sim_players[pid].dice]
                result = resolve_exact(
                    bids[-1], current_pid, all_dice, percolateur=percolateur
                )
                caller = sim_players[current_pid]
                if result.loser_id is None:
                    # Exact succeeded: caller gains 1 die (max 5)
                    if caller.dice_count < 5:
                        caller.dice.append(0)  # placeholder; re-rolled next round
                else:
                    caller.dice = caller.dice[:-1]
                # Per Q-F: Exact caller always opens the next round
                next_starter = current_pid
                break

        # Remove eliminated players
        active_ids = [pid for pid in active_ids if not sim_players[pid].eliminated]

        # Resolve starter: must be an active player
        if next_starter not in active_ids and active_ids:
            # Fall back to the next active player in original order
            for offset in range(1, len(strategies) + 1):
                candidate = (next_starter + offset) % len(strategies)
                if candidate in active_ids:
                    next_starter = candidate
                    break

        if active_ids:
            starter_id = next_starter

    winner_id = active_ids[0] if active_ids else -1
    return (
        GameRecord(
            game_id=game_id,
            winner_id=winner_id,
            n_rounds=n_rounds,
            strategy_names=[s.name for s in strategies],
        ),
        bid_records,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({100 * done // total}%)"


def run_simulation(
    n_games: int,
    strategies: list[Strategy],
    *,
    seed: int | None = None,
    output_dir: Path | None = None,
    verbose: bool = False,
    label: str = "",
) -> SimulationResults:
    """
    Run *n_games* full Perudo games and return aggregated statistics.

    Args:
        n_games:    Number of games to simulate.
        strategies: One Strategy per player; len(strategies) == n_players.
        seed:       RNG seed for reproducibility (None = random).
        output_dir: If provided, write game_records.csv, bid_records.csv,
                    and summary.md to this directory.
        verbose:    Print a progress bar to stdout (overwrites same line).
        label:      Optional prefix shown in the progress bar.

    Returns:
        SimulationResults with per-strategy win rates and Wilson CIs.
    """
    import sys
    import time

    rng = np.random.default_rng(seed)

    all_games: list[GameRecord] = []
    all_bids: list[BidRecord] = []
    t_start = time.perf_counter()

    for game_id in range(n_games):
        game_record, bid_records = _run_single_game(strategies, rng, game_id)
        all_games.append(game_record)
        all_bids.extend(bid_records)

        step = max(1, n_games // 100)
        if verbose and (game_id % step == 0 or game_id == n_games - 1):
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / (game_id + 1)) * (n_games - game_id - 1)
            wins = dict.fromkeys(range(len(strategies)), 0)
            for g in all_games:
                wins[g.winner_id] += 1
            rates = "  ".join(
                f"{strategies[i].name}: {wins[i] / (game_id + 1):.1%}"
                for i in range(len(strategies))
            )
            prefix = f"{label}  " if label else ""
            bar = _progress_bar(game_id + 1, n_games)
            line = f"\r  {prefix}{bar}  ETA: {eta:.0f}s  |  {rates}   "
            sys.stdout.write(line)
            sys.stdout.flush()

    if verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    results = SimulationResults(
        n_games=n_games,
        n_players=len(strategies),
        strategy_names=[s.name for s in strategies],
        game_records=all_games,
        bid_records=all_bids,
    )

    if output_dir is not None:
        write_csv(results, output_dir)
        write_markdown(results, output_dir / "summary.md")

    return results
