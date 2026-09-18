"""Presentation and decision interfaces used by the game engine.

The engine talks to exactly two abstractions:

* ``View``    - something that can redraw itself from the current game state.
* ``Decider`` - something that answers every question the rules ask a player.

Neither interface knows about tkinter. A GUI, a headless mock, or an AI agent
are all just implementations of these.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# An action is a plain dict, e.g. {"type": "PLAY_CARD", "card_id": "12", ...}.
# The authoritative list of action types lives in svai.actions.
Action = Dict[str, Any]


class View(ABC):
    """Anything that presents the game state. It never makes decisions."""

    @abstractmethod
    def update(self) -> None:
        """Redraw from the current game state."""


class NullView(View):
    """A view that shows nothing. Used for headless play."""

    def update(self) -> None:
        pass


class Decider(ABC):
    """Answers every choice the engine (or the turn loop) asks of a player."""

    @abstractmethod
    def choose_action(self, game: Any, player_id: str, legal_actions: Optional[List[Action]]) -> Action:
        """Pick the next main-phase action for ``player_id``."""

    @abstractmethod
    def choose_option(self, prompt: str, choices: Dict[str, Any]) -> Any:
        """Pick one value out of ``choices`` (label -> value) for a card effect."""

    @abstractmethod
    def choose_mulligan(self, player_id: str, hand: List[Any]) -> List[str]:
        """Return the card ids to send back during the mulligan."""

    @abstractmethod
    def choose_discard(self, player_id: str, hand: List[Any], count: int) -> List[str]:
        """Return ``count`` card ids to discard from ``hand``."""

    def choose_fuse(self, player_id: str, base_card: Any, candidates: List[Any]) -> List[str]:
        """Return the ids of hand cards to fuse into ``base_card``.

        Not abstract: declining to fuse is always legal, so that is the default.
        """
        return []

    def notify_action_applied(self, game: Any, player_id: str, action: Action) -> None:
        """Hook called by the turn loop after an action was executed."""
