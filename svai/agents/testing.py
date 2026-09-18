"""Agents that exist only to test the tooling."""

import time
from typing import Any, List

from svai.agents.base import Agent
from svai.interfaces import Action


class HangAgent(Agent):
    """Never decides. Used to verify that the arena's per-game timeout fires."""

    def act(self, game: Any, player_id: str, legal_actions: List[Action]) -> Action:
        while True:
            time.sleep(1)
