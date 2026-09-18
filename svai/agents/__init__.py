"""Agent implementations and a name -> constructor registry (used by the arena)."""

import random
from typing import Callable, Dict, Optional

from svai.agents.base import Agent

_REGISTRY: Dict[str, Callable[[Optional[random.Random]], Agent]] = {}


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def make_agent(name: str, rng: Optional[random.Random] = None) -> Agent:
    """Build the agent registered under ``name`` with its own ``rng``."""
    if name not in _REGISTRY:
        # Lazy imports so that importing the package stays cheap.
        if name == "random":
            from svai.agents.random_agent import RandomAgent
            _REGISTRY["random"] = lambda r: RandomAgent(r)
        elif name == "greedy":
            from svai.agents.greedy_agent import GreedyAgent
            _REGISTRY["greedy"] = lambda r: GreedyAgent(r)
        elif name == "hang":  # test-only: never returns, exercises the arena timeout
            from svai.agents.testing import HangAgent
            _REGISTRY["hang"] = lambda r: HangAgent()
        else:
            raise KeyError(f"unknown agent {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](rng)


def agent_names():
    return sorted(set(_REGISTRY) | {"random", "greedy"})
