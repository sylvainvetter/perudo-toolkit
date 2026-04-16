"""M4 — CFR self-play trainer and deployable bot."""

from perudo.m4.bot import CFRBot
from perudo.m4.cfr import CFRTrainer
from perudo.m4.policy import Policy

__all__ = ["CFRBot", "CFRTrainer", "Policy"]
