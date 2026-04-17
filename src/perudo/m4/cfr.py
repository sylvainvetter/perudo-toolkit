"""
M4 — Single-agent CFR+ trainer vs fixed ThresholdBot opponents.

Algorithm: CFR+ (Brown & Sandholm 2019) — game-level reward, one learning agent.

CFR+ improvements over vanilla CFR:
  1. Regret clamping: negative regrets are zeroed after each update
     (regrets = max(regrets + delta, 0)), preventing over-correction.
  2. Linear strategy weighting: strategy_sum += strategy * t, giving
     recent strategies more influence in the final policy average.
  3. Per-round dense reward: -1/0/+1/(n-1)/+1 each round, providing
     ~14× more gradient signal per game than a single terminal reward.

These changes yield 5-20x faster convergence on small tabular games
(Tammelin 2014; Brown & Sandholm 2019).

Only player 0 learns. Players 1..n-1 are ThresholdBot (calibrated per
player count).

Performance: _fast_action inlines the ThresholdBot decision logic,
eliminating ~63 Python object allocations and up to 63 scipy calls
per episode that the previous GameState-based approach incurred.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from perudo.core.rules import resolve_exact, resolve_liar
from perudo.core.types import Bid, Exact, Liar, RaiseBid
from perudo.m2 import RecommenderConfig, config_for_n_players
from perudo.m4._tables import (
    binom_pmf,
    binom_sf,
    min_q_table,
    own_counts_from_faces,
)
from perudo.m4.infostate import (
    N_ACTIONS,
    decode_action,
    legal_mask,
    make_info_key,
    make_opening_key,
)

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _Decision:
    info_key: tuple[Any, ...]
    action_idx: int
    strategy: np.ndarray


@dataclass
class _Player:
    pid: int
    dice: list[int] = field(default_factory=list)
    exact_used: bool = False
    face_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )

    @property
    def n_dice(self) -> int:
        return len(self.dice)

    def roll(self, rng: np.random.Generator) -> None:
        arr = rng.integers(1, 7, size=len(self.dice))
        self.dice = list(map(int, arr))
        self.face_counts = np.bincount(arr, minlength=7)[1:].astype(np.int32)


# ---------------------------------------------------------------------------
# Fast inlined ThresholdBot decision — zero Python object allocation
# ---------------------------------------------------------------------------

def _fast_action(
    face_counts: np.ndarray,
    n_own: int,
    bid_q: int,
    bid_v: int,
    total: int,
    exact_used: bool,
    perco: bool,
    pid: int,
    cfg: RecommenderConfig,
) -> Liar | Exact | RaiseBid:
    """
    Inlined ThresholdBot decision using pre-computed lookup tables.

    Args:
        face_counts : int array (6,) — count per face value for this player
        n_own       : number of own dice (= sum(face_counts))
    """
    n_unk = total - n_own
    joker = not perco

    own_counts = own_counts_from_faces(face_counts, joker)  # shape (6,)

    # --- Opening bid ---
    if bid_q == 0:
        best_v = int(np.argmax(own_counts)) + 1
        return RaiseBid(Bid(quantity=1, value=best_v, player_id=pid))

    # --- Evaluate current bid ---
    cur_p_idx = 0 if (perco or bid_v == 1) else 1
    own_cur = int(own_counts[bid_v - 1])
    needed_cur = bid_q - own_cur
    p_true = binom_sf(needed_cur - 1, n_unk, cur_p_idx)

    if p_true < cfg.threshold_liar:
        return Liar()

    if not exact_used:
        p_exact = binom_pmf(needed_cur, n_unk, cur_p_idx)
        if p_exact > cfg.threshold_exact:
            return Exact()

    # --- Find best raise: table lookups for all 6 face values ---
    min_qs = np.array([min_q_table(bid_q, bid_v, v) for v in range(1, 7)],
                      dtype=np.int32)
    valid = min_qs <= total

    if not valid.any():
        return Liar()

    # p_idx per value: 0 if perco or v==1, else 1
    p_idxs = np.zeros(6, dtype=np.int32) if perco else np.array(
        [0, 1, 1, 1, 1, 1], dtype=np.int32
    )
    neededs = min_qs - own_counts.astype(np.int32)
    best_v_idx = -1
    best_pt = -1.0
    for vi in range(6):
        if not valid[vi]:
            continue
        pt = binom_sf(int(neededs[vi]) - 1, n_unk, int(p_idxs[vi]))
        if pt > best_pt:
            best_pt = pt
            best_v_idx = vi

    best_v = best_v_idx + 1
    best_q = int(min_qs[best_v_idx])
    return RaiseBid(Bid(quantity=best_q, value=best_v, player_id=pid))


# ---------------------------------------------------------------------------
# Self-play opponent action
# ---------------------------------------------------------------------------


def _opponent_action(
    opponent_policy: Policy | None,
    player: _Player,
    bid_q: int,
    bid_v: int,
    total: int,
    perco: bool,
    pid: int,
    cfg: RecommenderConfig,
    rng: np.random.Generator,
    n_bids: int = 0,
) -> Liar | Exact | RaiseBid:
    """
    Decision for a self-play opponent.

    Uses the frozen opponent_policy when the state is known, falls back to
    _fast_action (ThresholdBot) for unseen states or when no policy exists yet.

    This means early in training opponents behave like ThresholdBot, and
    progressively adopt the learned CFR strategy as coverage grows.
    """
    if opponent_policy is not None:
        if bid_q == 0:
            info_key: tuple = make_opening_key(player.face_counts, total, perco)
            mask = legal_mask(0, 0, total, exact_avail=False)
        else:
            exact_avail = not player.exact_used
            info_key = make_info_key(
                player.dice, bid_q, bid_v, total, exact_avail, perco, n_bids
            )
            mask = legal_mask(bid_q, bid_v, total, exact_avail)

        if opponent_policy.knows(info_key):
            probs = opponent_policy.get_probs(info_key, mask)
            legal = np.where(mask)[0]
            lp = probs[legal]
            s = lp.sum()
            lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
            idx = int(rng.choice(legal, p=lp))
            return decode_action(idx, bid_q, bid_v, pid)

    # Fallback: fast ThresholdBot (no policy yet, or unseen state)
    return _fast_action(
        player.face_counts, player.n_dice,
        bid_q, bid_v, total, player.exact_used, perco, pid, cfg,
    )


# ---------------------------------------------------------------------------
# Turn order helper
# ---------------------------------------------------------------------------


def _next_active(active: list[int], from_id: int) -> int:
    try:
        idx = active.index(from_id)
    except ValueError:
        candidates = [x for x in sorted(active) if x > from_id]
        return candidates[0] if candidates else active[0]
    return active[(idx + 1) % len(active)]


# ---------------------------------------------------------------------------
# Fast evaluation (no GameState objects — same speed as training)
# ---------------------------------------------------------------------------


def fast_eval(
    policy: Policy,
    n_players: int,
    n_games: int,
    seed: int = 999,
) -> tuple[float, float, float]:
    """
    Evaluate a policy against fast ThresholdBot opponents.

    Uses _fast_action for all players — no GameState/recommend() overhead.
    Returns (win_rate, ci_lo, ci_hi) with Wilson 95% confidence interval.
    """
    import math

    cfg = config_for_n_players(n_players)
    rng = np.random.default_rng(seed)
    wins = 0

    for _ in range(n_games):
        players = {i: _Player(pid=i, dice=[0] * 5) for i in range(n_players)}
        active: list[int] = list(range(n_players))
        starter: int = 0
        perco: bool = False

        while len(active) > 1:
            for pid in active:
                players[pid].roll(rng)

            bids: list[Bid] = []
            current: int = starter

            while True:
                all_dice = [d for pid in active for d in players[pid].dice]
                total = len(all_dice)
                bid_q = bids[-1].quantity if bids else 0
                bid_v = bids[-1].value if bids else 0

                if current == 0 and bid_q == 0:
                    # CFR opening bid (fallback to _fast_action for old models)
                    info_key: tuple = make_opening_key(
                        players[0].face_counts, total, perco
                    )
                    if policy.knows(info_key):
                        mask = legal_mask(0, 0, total, exact_avail=False)
                        probs = policy.get_probs(info_key, mask)
                        legal = np.where(mask)[0]
                        lp = probs[legal]
                        s = lp.sum()
                        lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
                        idx = int(rng.choice(legal, p=lp))
                        action: Liar | Exact | RaiseBid = decode_action(idx, 0, 0, 0)
                    else:
                        action = _fast_action(
                            players[0].face_counts, players[0].n_dice,
                            0, 0, total, players[0].exact_used, perco, 0, cfg,
                        )
                elif current == 0 and bid_q > 0:
                    # CFR non-opening decision
                    exact_avail = not players[0].exact_used
                    info_key = make_info_key(
                        players[0].dice, bid_q, bid_v, total, exact_avail, perco,
                        n_bids=len(bids),
                    )
                    mask = legal_mask(bid_q, bid_v, total, exact_avail)
                    probs = policy.get_probs(info_key, mask)
                    legal = np.where(mask)[0]
                    lp = probs[legal]
                    s = lp.sum()
                    lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
                    idx = int(rng.choice(legal, p=lp))
                    action = decode_action(idx, bid_q, bid_v, 0)
                else:
                    action = _fast_action(
                        players[current].face_counts, players[current].n_dice,
                        bid_q, bid_v, total,
                        players[current].exact_used, perco, current, cfg,
                    )

                if isinstance(action, RaiseBid):
                    bids.append(action.bid)
                    current = _next_active(active, current)
                elif isinstance(action, Liar):
                    result = resolve_liar(
                        bids[-1], current, all_dice, percolateur=perco
                    )
                    loser_id = result.loser_id
                    starter = loser_id if loser_id is not None else starter
                    if loser_id is not None:
                        players[loser_id].dice = players[loser_id].dice[:-1]
                    break
                else:  # Exact
                    players[current].exact_used = True
                    result = resolve_exact(
                        bids[-1], current, all_dice, percolateur=perco
                    )
                    if result.loser_id is None:
                        if players[current].n_dice < 5:
                            players[current].dice.append(0)
                    else:
                        players[result.loser_id].dice = (
                            players[result.loser_id].dice[:-1]
                        )
                    starter = current
                    break

            active = [pid for pid in active if players[pid].n_dice > 0]
            if active and starter not in active:
                starter = active[0]

        if active and active[0] == 0:
            wins += 1

    # Wilson 95% CI
    n = n_games
    p = wins / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Single training episode
# ---------------------------------------------------------------------------


def _run_episode(
    trainer: CFRTrainer,
    n_players: int,
    cfg: RecommenderConfig,
    rng: np.random.Generator,
    epsilon: float,
    rng_explore: np.random.Generator,
) -> int:
    """
    Play one game: player 0 uses CFR, players 1..n-1 use _fast_action.

    Reward is per-round, assigned immediately after each round:
        -1                 player 0 lost a die this round
        +1                 player 0 gained a die (Exact success)
        +1/(n_active-1)    player 0 survived (another player lost a die)
        0                  nobody lost a die

    Per-round reward provides a dense, zero-expected-value signal that
    correlates strongly with the game objective. Game-level (+1/-1) is
    too sparse for a ~14-round horizon: the same reward applied to all
    decisions in a game cannot distinguish good decisions from bad ones
    within that game, resulting in near-zero win rates in practice.

    Returns the winner's player id.
    """
    players = {i: _Player(pid=i, dice=[0] * 5) for i in range(n_players)}
    active: list[int] = list(range(n_players))
    starter: int = 0
    perco: bool = False

    while len(active) > 1:
        # Roll dice
        for pid in active:
            players[pid].roll(rng)

        # Bidding round
        bids: list[Bid] = []
        current: int = starter
        round_dec: list[_Decision] = []
        loser_id: int | None = None
        gainer_id: int | None = None

        while True:
            all_dice = [d for pid in active for d in players[pid].dice]
            total = len(all_dice)
            bid_q = bids[-1].quantity if bids else 0
            bid_v = bids[-1].value if bids else 0

            if current == 0:
                if bid_q == 0:
                    # Opening bid: learned by CFR+ (face_counts → which value to open)
                    info_key: tuple = make_opening_key(
                        players[0].face_counts, total, perco
                    )
                    mask = legal_mask(0, 0, total, exact_avail=False)
                else:
                    # Non-opening: CFR+ strategy
                    exact_avail = not players[0].exact_used
                    info_key = make_info_key(
                        players[0].dice, bid_q, bid_v, total, exact_avail, perco,
                        n_bids=len(bids),
                    )
                    mask = legal_mask(bid_q, bid_v, total, exact_avail)

                # CFR+: linear weight t = iteration number (1-indexed)
                cfr_weight = float(trainer.n_iters + 1)
                strategy = trainer.get_strategy(info_key, mask, weight=cfr_weight)
                legal = np.where(mask)[0]
                if epsilon > 0 and rng_explore.random() < epsilon:
                    idx = int(rng_explore.choice(legal))
                else:
                    probs = strategy[legal]
                    probs = probs / probs.sum()
                    idx = int(rng.choice(legal, p=probs))

                round_dec.append(_Decision(info_key, idx, strategy.copy()))
                action: Liar | Exact | RaiseBid = decode_action(idx, bid_q, bid_v, 0)

            else:
                # Opponent: self-play policy when available, else fast ThresholdBot
                if trainer.opponent_policy is not None:
                    action = _opponent_action(
                        trainer.opponent_policy,
                        players[current],
                        bid_q, bid_v, total, perco, current, cfg, rng,
                        n_bids=len(bids),
                    )
                else:
                    action = _fast_action(
                        players[current].face_counts, players[current].n_dice,
                        bid_q, bid_v, total,
                        players[current].exact_used, perco, current, cfg,
                    )

            # Execute action
            if isinstance(action, RaiseBid):
                bids.append(action.bid)
                current = _next_active(active, current)

            elif isinstance(action, Liar):
                result = resolve_liar(bids[-1], current, all_dice, percolateur=perco)
                loser_id = result.loser_id
                starter = loser_id if loser_id is not None else starter
                break

            else:  # Exact
                players[current].exact_used = True
                result = resolve_exact(bids[-1], current, all_dice, percolateur=perco)
                if result.loser_id is None:
                    gainer_id = current
                    if players[current].n_dice < 5:
                        players[current].dice.append(0)
                else:
                    loser_id = result.loser_id
                starter = current
                break

        # Per-round reward: dense, zero-expected-value signal
        if round_dec and 0 in active:
            n_active = len(active)
            if loser_id == 0:
                reward = -1.0
            elif gainer_id == 0 and loser_id is None:
                reward = +1.0
            elif loser_id is not None:
                reward = 1.0 / max(n_active - 1, 1)
            else:
                reward = 0.0

            for dec in round_dec:
                regrets = trainer.regret_sum.setdefault(
                    dec.info_key, np.zeros(N_ACTIONS)
                )
                indicator = np.zeros(N_ACTIONS)
                indicator[dec.action_idx] = 1.0
                regrets += (indicator - dec.strategy) * reward
                # CFR+: clamp negative regrets to 0 (prevents over-correction)
                np.maximum(regrets, 0.0, out=regrets)

        # Apply die changes and eliminate
        if loser_id is not None:
            players[loser_id].dice = players[loser_id].dice[:-1]

        active = [pid for pid in active if players[pid].n_dice > 0]
        if active and starter not in active:
            starter = active[0]

    return active[0] if active else -1


# ---------------------------------------------------------------------------
# CFR Trainer
# ---------------------------------------------------------------------------


class CFRTrainer:
    """
    Trains player 0's strategy via CFR+ against fast-inlined ThresholdBot
    opponents (_fast_action — no GameState object creation).

    Implements CFR+ (Tammelin 2014 / Brown & Sandholm 2019):
    - Negative regrets clamped to 0 after each update
    - Strategy sum weighted linearly by iteration t (recent strategies
      get higher weight → faster policy convergence)

    Attributes:
        regret_sum   : info_key -> cumulative regret vector (N_ACTIONS)
        strategy_sum : info_key -> t-weighted strategy vector (for averaging)
        n_iters      : episodes completed
    """

    def __init__(self) -> None:
        self.regret_sum: dict[tuple[Any, ...], np.ndarray] = {}
        self.strategy_sum: dict[tuple[Any, ...], np.ndarray] = {}
        self.n_iters: int = 0
        # Frozen policy used as opponent in self-play (None = ThresholdBot)
        self.opponent_policy: Policy | None = None

    def get_strategy(
        self,
        info_key: tuple[Any, ...],
        mask: np.ndarray,
        weight: float = 1.0,
    ) -> np.ndarray:
        """Compute current strategy via regret matching and accumulate for averaging.

        Args:
            weight: linear weight for strategy_sum accumulation.
                    Pass t (current iteration, 1-indexed) for CFR+ weighting.
                    Pass 1.0 for vanilla uniform weighting.
        """
        regrets = self.regret_sum.get(info_key, np.zeros(N_ACTIONS))
        pos: np.ndarray = np.where(mask, np.maximum(regrets, 0.0), 0.0)
        total = float(pos.sum())
        if total > 0:
            strategy: np.ndarray = pos / total
        else:
            n_legal = int(mask.sum())
            strategy = np.where(mask, 1.0 / n_legal, 0.0)

        acc = self.strategy_sum.setdefault(info_key, np.zeros(N_ACTIONS))
        acc += strategy * weight  # CFR+: linear weighting by iteration t
        return strategy

    def get_average_strategy(
        self, info_key: tuple[Any, ...], mask: np.ndarray
    ) -> np.ndarray:
        s = self.strategy_sum.get(info_key)
        if s is None:
            n_legal = int(mask.sum())
            result: np.ndarray = np.where(mask, 1.0 / n_legal, 0.0)
            return result
        total = float(s.sum())
        if total > 0:
            return s / total
        n_legal = int(mask.sum())
        result = np.where(mask, 1.0 / n_legal, 0.0)
        return result

    def train(
        self,
        n_iters: int,
        n_players: int,
        *,
        seed: int = 42,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.02,
        selfplay_every: int = 0,
        verbose: bool = True,
    ) -> None:
        """
        Train for n_iters episodes.

        Args:
            selfplay_every: freeze current policy as opponent every N *total*
                            iterations (0 = disabled, opponents stay ThresholdBot).
                            At each update, opponents switch from ThresholdBot to the
                            current CFR average strategy for known states.
        """
        cfg = config_for_n_players(n_players)
        rng = np.random.default_rng(seed)
        rng_explore = np.random.default_rng(seed + 1)
        t0 = time.perf_counter()

        for i in range(n_iters):
            # Self-play: periodically freeze current policy as opponent
            if (
                selfplay_every > 0
                and self.n_iters > 0
                and self.n_iters % selfplay_every == 0
            ):
                self.opponent_policy = self.to_policy()

            progress = i / max(n_iters - 1, 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

            _run_episode(self, n_players, cfg, rng, epsilon, rng_explore)
            self.n_iters += 1

            if verbose and (i % max(1, n_iters // 200) == 0 or i == n_iters - 1):
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (n_iters - i - 1)
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                filled = int(30 * (i + 1) / n_iters)
                bar = "#" * filled + "-" * (30 - filled)
                sp = "*" if self.opponent_policy is not None else " "
                sys.stdout.write(
                    f"\r  [{bar}] {i + 1:>7,}/{n_iters:,}  "
                    f"states={len(self.regret_sum):>5,}  "
                    f"{rate:>6,.0f} ep/s  ETA={eta:.0f}s {sp}  "
                )
                sys.stdout.flush()

        if verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def to_policy(self) -> Policy:
        from perudo.m4.policy import Policy

        return Policy(
            strategy_sum={k: v.copy() for k, v in self.strategy_sum.items()},
            n_iters=self.n_iters,
        )


from perudo.m4.policy import Policy  # noqa: E402
