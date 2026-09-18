"""Uniformly random baseline agent."""

import random
from typing import Any, Dict, List, Optional

from svai.agents.base import Agent
from svai.interfaces import Action


class RandomAgent(Agent):
    """Picks uniformly among legal actions and among effect options.

    Uses its own ``random.Random`` so that an agent's randomness is separable
    from the engine's; pass a seeded instance for reproducible games.
    """

    name = "random"

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()

    def act(self, game: Any, player_id: str, legal_actions: List[Action]) -> Action:
        return self.rng.choice(legal_actions)

    def choose_option(self, prompt: str, choices: Dict[str, Any]) -> Any:
        if not choices:
            return None
        return choices[self.rng.choice(list(choices.keys()))]

    def choose_mulligan(self, player_id: str, hand: List[Any]) -> List[str]:
        if not hand:
            return []
        k = self.rng.randint(0, len(hand))
        return [c.card_id for c in self.rng.sample(hand, k)]

    def choose_discard(self, player_id: str, hand: List[Any], count: int) -> List[str]:
        ids = [c.card_id for c in hand]
        return self.rng.sample(ids, min(count, len(ids)))

    def choose_fuse(self, player_id: str, base_card: Any, candidates: List[Any]) -> List[str]:
        k = self.rng.randint(0, len(candidates))
        return [c.card_id for c in self.rng.sample(candidates, k)]
