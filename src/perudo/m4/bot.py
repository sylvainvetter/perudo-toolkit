"""
M4 — CFRBot: a Strategy backed by a trained CFR policy.

For info states visited during training, samples from the learned
time-averaged strategy.  For unseen states (new dice combinations /
game situations outside the training distribution) it falls back to
ThresholdBot with calibrated thresholds.

Usage:
    from perudo.m4 import CFRBot
    from perudo.m4.policy import Policy
    from pathlib import Path

    policy = Policy.load(Path("models/cfr_4p.pkl"))
    bot = CFRBot(policy)
"""

from __future__ import annotations

import numpy as np

from perudo.core.types import Action, GameState
from perudo.m3.strategies import Strategy, ThresholdBot
from perudo.m4.infostate import decode_action, legal_mask, make_info_key
from perudo.m4.policy import Policy


class CFRBot(Strategy):
    """
    Play using a trained CFR policy.

    Falls back to ThresholdBot for info states not covered by the policy.
    Tracks the fallback rate to help diagnose under-trained policies.
    """

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._fallback = ThresholdBot()
        self._n_total = 0
        self._n_fallback = 0

    def choose_action(self, game_state: GameState, rng: np.random.Generator) -> Action:
        self._n_total += 1
        pid = game_state.round.current_player_id
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

        if self._policy.knows(info_key):
            probs = self._policy.get_probs(info_key, mask)
            legal = np.where(mask)[0]
            legal_probs = probs[legal]
            s = legal_probs.sum()
            if s > 0:
                legal_probs /= s
                action_idx = int(rng.choice(legal, p=legal_probs))
                return decode_action(action_idx, bid_q, bid_v, pid)

        # Unseen state — delegate to ThresholdBot
        self._n_fallback += 1
        return self._fallback.choose_action(game_state, rng)

    def wants_percolateur(self, game_state: GameState) -> bool:
        return self._fallback.wants_percolateur(game_state)

    @property
    def fallback_rate(self) -> float:
        """Fraction of decisions delegated to ThresholdBot (lower = better coverage)."""
        return self._n_fallback / self._n_total if self._n_total > 0 else 0.0

    @property
    def name(self) -> str:
        return "CFRBot"
