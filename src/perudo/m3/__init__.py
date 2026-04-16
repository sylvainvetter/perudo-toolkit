"""M3 — Monte Carlo simulator, strategies, and reporter."""

from perudo.m3.reporter import (
    BidRecord,
    GameRecord,
    SimulationResults,
    StrategyStats,
    write_csv,
    write_markdown,
)
from perudo.m3.simulator import run_simulation
from perudo.m3.strategies import Honest, RandomLegal, Strategy, ThresholdBot

__all__ = [
    "BidRecord",
    "GameRecord",
    "Honest",
    "RandomLegal",
    "SimulationResults",
    "Strategy",
    "StrategyStats",
    "ThresholdBot",
    "run_simulation",
    "write_csv",
    "write_markdown",
]
