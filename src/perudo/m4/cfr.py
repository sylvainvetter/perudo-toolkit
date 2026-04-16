"""
M4 — Monte Carlo CFR trainer via self-play.

Algorithm: outcome-sampling regret update (simplified MCCFR).

Each training iteration:
  1. Play one complete game with all players using the current mixed strategy.
  2. Record every decision: (info_key, action_idx, strategy_vector).
  3. After the game, update regrets for every decision:
       Δregret[a] = (indicator[a == a*] - strategy[a]) × outcome
     where outcome ∈ {+1, −1/(n−1)} and a* is the action taken.
  4. Recompute the mixed strategy via regret matching (positive regrets only).

The time-averaged strategy (strategy_sum / n_iters) converges to a Nash
equilibrium approximation.  For n > 2 players the convergence guarantee
weakens, but empirically the policy improves steadily.

Typical usage:
    trainer = CFRTrainer()
    trainer.train(n_iters=100_000, n_players=4, verbose=True)
    policy = trainer.to_policy()
    policy.save(Path("models/cfr_4p.pkl"))
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from perudo.core.types import Action, GameState
from perudo.m3.simulator import run_simulation
from perudo.m3.strategies import Strategy
from perudo.m4.infostate import (
    N_ACTIONS,
    decode_action,
    legal_mask,
    make_info_key,
)

# ---------------------------------------------------------------------------
# Internal decision record
# ---------------------------------------------------------------------------


@dataclass
class _Decision:
    player_id: int
    info_key: tuple[Any, ...]
    action_idx: int
    strategy: np.ndarray  # full N_ACTIONS strategy vector at this state


# ---------------------------------------------------------------------------
# Recording strategy — wraps the trainer during self-play
# ---------------------------------------------------------------------------


class _RecordingStrategy(Strategy):
    """
    Samples actions using the current CFR mixed strategy and logs each
    decision into a shared buffer for the regret-update step.
    """

    def __init__(
        self,
        player_id: int,
        trainer: CFRTrainer,
        buffer: list[_Decision],
        epsilon: float,
        rng_explore: np.random.Generator,
    ) -> None:
        self._pid = player_id
        self._trainer = trainer
        self._buffer = buffer
        self._epsilon = epsilon
        self._rng_explore = rng_explore

    def choose_action(self, game_state: GameState, rng: np.random.Generator) -> Action:
        pid = self._pid
        player = next(p for p in game_state.players if p.id == pid)
        prev = game_state.round.current_bid
        total = sum(p.dice_count for p in game_state.players)

        bid_q = prev.quantity if prev else 0
        bid_v = prev.value if prev else 0

        info_key = make_info_key(
            player.dice,
            bid_q,
            bid_v,
            total,
            not player.exact_used,
            game_state.round.percolateur,
        )
        mask = legal_mask(bid_q, bid_v, total, not player.exact_used)
        strategy = self._trainer.get_strategy(info_key, mask)

        # ε-greedy exploration: with prob ε choose uniformly over legal actions
        if self._epsilon > 0 and self._rng_explore.random() < self._epsilon:
            legal = np.where(mask)[0]
            action_idx = int(self._rng_explore.choice(legal))
        else:
            legal = np.where(mask)[0]
            probs = strategy[legal]
            probs = probs / probs.sum()
            action_idx = int(rng.choice(legal, p=probs))

        self._buffer.append(_Decision(
            player_id=pid,
            info_key=info_key,
            action_idx=action_idx,
            strategy=strategy.copy(),
        ))

        return decode_action(action_idx, bid_q, bid_v, pid)

    @property
    def name(self) -> str:
        return f"CFR(p{self._pid})"


# ---------------------------------------------------------------------------
# CFR Trainer
# ---------------------------------------------------------------------------


class CFRTrainer:
    """
    Maintains cumulative regret tables and drives self-play training.

    Attributes:
        regret_sum    : info_key → cumulative regret array (length N_ACTIONS)
        strategy_sum  : info_key → cumulative strategy array (for averaging)
        n_iters       : number of training iterations completed
    """

    def __init__(self) -> None:
        self.regret_sum: dict[tuple[Any, ...], np.ndarray] = {}
        self.strategy_sum: dict[tuple[Any, ...], np.ndarray] = {}
        self.n_iters: int = 0

    # ------------------------------------------------------------------ strategy

    def get_strategy(self, info_key: tuple[Any, ...], mask: np.ndarray) -> np.ndarray:
        """
        Derive the current mixed strategy via regret matching and accumulate
        it into strategy_sum for computing the average strategy after training.
        """
        regrets = self.regret_sum.get(info_key, np.zeros(N_ACTIONS))
        pos: np.ndarray = np.where(mask, np.maximum(regrets, 0.0), 0.0)
        total = float(pos.sum())
        if total > 0:
            strategy: np.ndarray = pos / total
        else:
            n_legal = int(mask.sum())
            strategy = np.where(mask, 1.0 / n_legal, 0.0)

        # Accumulate for average strategy
        acc = self.strategy_sum.setdefault(info_key, np.zeros(N_ACTIONS))
        acc += strategy
        return strategy

    def get_average_strategy(
        self, info_key: tuple[Any, ...], mask: np.ndarray
    ) -> np.ndarray:
        """Return the time-averaged strategy (for deployment, not training)."""
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

    # ------------------------------------------------------------------ update

    def _update_regrets(
        self,
        decisions: list[_Decision],
        winner_id: int,
        n_players: int,
    ) -> None:
        """
        Apply regret updates from one game.

        For each decision (info_key, a*, strategy σ) by player p:
            Δregret[a] = (𝟙[a = a*] − σ[a]) × outcome
        where outcome = +1 if p won, −1/(n−1) otherwise.

        This is the policy-gradient / REINFORCE update with σ as baseline.
        It is an unbiased estimator of the counterfactual regret under the
        assumption that unchosen actions would have yielded the same outcome.
        """
        for dec in decisions:
            outcome = 1.0 if dec.player_id == winner_id else -1.0 / (n_players - 1)
            regrets = self.regret_sum.setdefault(dec.info_key, np.zeros(N_ACTIONS))
            # Δr[a] = (indicator - strategy) * outcome
            indicator = np.zeros(N_ACTIONS)
            indicator[dec.action_idx] = 1.0
            regrets += (indicator - dec.strategy) * outcome

    # ------------------------------------------------------------------ training loop

    def train(
        self,
        n_iters: int,
        n_players: int,
        *,
        seed: int = 42,
        epsilon_start: float = 0.10,
        epsilon_end: float = 0.01,
        verbose: bool = True,
    ) -> None:
        """
        Run n_iters self-play games and update regret tables.

        Args:
            n_iters       : number of games to simulate.
            n_players     : number of players per game (2–6).
            seed          : base RNG seed.
            epsilon_start : initial exploration rate (decays to epsilon_end).
            epsilon_end   : final exploration rate.
            verbose       : print progress bar to stdout.
        """
        rng = np.random.default_rng(seed)
        rng_explore = np.random.default_rng(seed + 1)
        t0 = time.perf_counter()

        for i in range(n_iters):
            # Linearly decay exploration
            progress = i / max(n_iters - 1, 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

            decisions: list[_Decision] = []
            strategies: list[Strategy] = [
                _RecordingStrategy(pid, self, decisions, epsilon, rng_explore)
                for pid in range(n_players)
            ]
            game_seed = int(rng.integers(0, 2**31))
            results = run_simulation(1, strategies, seed=game_seed)
            winner_id = results.game_records[0].winner_id
            self._update_regrets(decisions, winner_id, n_players)
            self.n_iters += 1

            if verbose and (i % max(1, n_iters // 100) == 0 or i == n_iters - 1):
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (n_iters - i - 1)
                filled = int(30 * (i + 1) / n_iters)
                bar = "#" * filled + "-" * (30 - filled)
                n_states = len(self.regret_sum)
                sys.stdout.write(
                    f"\r  [{bar}] {i + 1:>7,}/{n_iters:,}  "
                    f"states={n_states:>7,}  ETA={eta:.0f}s   "
                )
                sys.stdout.flush()

        if verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()

    # ------------------------------------------------------------------ export

    def to_policy(self) -> Policy:
        """Extract the time-averaged strategy as a deployable Policy."""
        from perudo.m4.policy import Policy

        return Policy(
            strategy_sum={k: v.copy() for k, v in self.strategy_sum.items()},
            n_iters=self.n_iters,
        )


# Import here to avoid circular reference at module level
from perudo.m4.policy import Policy  # noqa: E402
