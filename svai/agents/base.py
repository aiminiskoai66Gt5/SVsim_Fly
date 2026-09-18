"""Base class for AI agents. An agent never sees GUI objects."""

from typing import Any, Dict, List

from svai.interfaces import Action


class Agent:
    """Minimal agent contract. Subclasses must implement ``act``.

    The other hooks have deliberately dumb, deterministic defaults so that a
    new agent only has to think about main-phase actions to be playable.
    """

    def act(self, game: Any, player_id: str, legal_actions: List[Action]) -> Action:
        raise NotImplementedError

    def choose_option(self, prompt: str, choices: Dict[str, Any]) -> Any:
        return next(iter(choices.values())) if choices else None

    def choose_mulligan(self, player_id: str, hand: List[Any]) -> List[str]:
        return []

    def choose_discard(self, player_id: str, hand: List[Any], count: int) -> List[str]:
        return [card.card_id for card in hand[:count]]

    def choose_fuse(self, player_id: str, base_card: Any, candidates: List[Any]) -> List[str]:
        return []
