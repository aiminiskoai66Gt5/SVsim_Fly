"""Action representation shared by every driver of the engine.

M0 only needs ``apply_action``. Enumeration of legal actions moves here in M1.
"""

from typing import Any

from src.common.enums import CardType
from svai.interfaces import Action

PLAY_CARD = "PLAY_CARD"
ATTACK = "ATTACK"
EVOLVE = "EVOLVE"
SUPER_EVOLVE = "SUPER_EVOLVE"
ENGAGE = "ENGAGE"
END_TURN = "END_TURN"


def apply_action(game: Any, player_id: str, action: Action) -> Any:
    """Execute ``action`` for ``player_id`` through the public Game API.

    Returns whatever the engine call returns. Unknown action types raise, they
    are never skipped silently.
    """
    kind = action["type"]
    if kind == PLAY_CARD:
        return game.play_card(player_id, action["card_id"], action.get("enhanced_cost", 0),
                              action.get("use_extra_pp", False))
    if kind == ATTACK:
        target_type = game.game_state_manager.get_type(action["target_id"])
        if target_type == CardType.LEADER:
            return game.attack_leader(action["attacker_id"])
        return game.attack_follower(action["attacker_id"], action["target_id"])
    if kind == EVOLVE:
        return game.evolve_follower(action["card_id"], player_id)
    if kind == SUPER_EVOLVE:
        return game.super_evolve_follower(action["card_id"], player_id)
    if kind == ENGAGE:
        return game.engage_card(action["card_id"], player_id)
    if kind == END_TURN:
        return game.end_turn(player_id)
    raise ValueError(f"Unknown action type: {kind!r}")
