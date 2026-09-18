"""Action representation shared by every driver of the engine.

An action is a plain dict::

    {"type": "PLAY_CARD", "card_id": "12", "enhanced_cost": 0, "use_extra_pp": False}
    {"type": "ATTACK", "attacker_id": "7", "target_id": "player2"}   # target may be a leader id
    {"type": "EVOLVE" | "SUPER_EVOLVE" | "ENGAGE", "card_id": "7"}
    {"type": "END_TURN"}

``legal_actions`` enumerates them (moved verbatim from fuzz_runner.py, M1),
``apply_action`` executes one, and ``encode`` / ``decode`` map them onto a
fixed-size discrete index space for the Gymnasium env:

    index                              meaning
    0  .. 17    hand_slot * 2 + use_extra_pp   PLAY_CARD (hand slot 0..8)
    18 .. 47    18 + attacker_slot * 6 + t     ATTACK, t = enemy field slot 0..4, 5 = enemy leader
    48 .. 52    48 + slot                      EVOLVE
    53 .. 57    53 + slot                      SUPER_EVOLVE
    58 .. 62    58 + slot                      ENGAGE
    63                                         END_TURN

Slots are positions in the engine's zone lists (hand / field order). A card's
Enhance cost is not a separate choice: like the original fuzzer, the largest
affordable Enhance is used (documented compromise, see OBSERVATION.md).
"""

from typing import Any, Dict, List, Optional

from src.common.enums import CardType, EffectType, Zone
from src.common import text as T
from svai.interfaces import Action

PLAY_CARD = "PLAY_CARD"
ATTACK = "ATTACK"
EVOLVE = "EVOLVE"
SUPER_EVOLVE = "SUPER_EVOLVE"
ENGAGE = "ENGAGE"
END_TURN = "END_TURN"

HAND_MAX = 9
FIELD_MAX = 5
_PLAY_BASE = 0
_ATTACK_BASE = HAND_MAX * 2                     # 18
_EVOLVE_BASE = _ATTACK_BASE + FIELD_MAX * 6     # 48
_SUPER_BASE = _EVOLVE_BASE + FIELD_MAX          # 53
_ENGAGE_BASE = _SUPER_BASE + FIELD_MAX          # 58
END_TURN_INDEX = _ENGAGE_BASE + FIELD_MAX       # 63
ACTION_SPACE_SIZE = END_TURN_INDEX + 1          # 64


class IllegalActionError(ValueError):
    """Raised when a caller tries to execute an action that is not legal now."""


def legal_actions(game: Any, current_player: str) -> List[Action]:
    """Collect every action ``current_player`` may take right now.

    Body moved unchanged from ``fuzz_runner.get_all_possible_actions``.
    """
    possible_actions = []
    opponent_id = game.opponent_id[current_player]

    # 1. Cards playable from hand.
    use_extra_pp_options = [False]
    if game.has_extra_pp(current_player):
        use_extra_pp_options.append(True)

    for use_extra_pp in use_extra_pp_options:
        hand_cards_id, is_validate = game.get_playable_cards_id(current_player, use_extra_pp)
        if hand_cards_id:
            for i, card_id in enumerate(hand_cards_id):
                if is_validate[i]:
                    current_pp, _ = game.game_state_manager.get_pp_info(current_player)
                    enhance_effects = [effect for effect in game.game_state_manager.get_card_effects(card_id, EffectType.ENHANCE)]
                    enhance_costs_for_card = [effect.enhance_cost for effect in enhance_effects if effect.enhance_cost <= current_pp + (1 if use_extra_pp else 0)]
                    enhanced_cost = max(enhance_costs_for_card) if enhance_costs_for_card else 0

                    possible_actions.append({
                        "type": PLAY_CARD,
                        "card_id": card_id,
                        "enhanced_cost": enhanced_cost,
                        "use_extra_pp": use_extra_pp
                    })

    # 2. Attack / evolve / super evolve / engage with cards on the field.
    player_field_card_ids = game.game_state_manager.get_card_ids_in_zone(current_player, Zone.FIELD)
    opponent_field_card_ids = game.game_state_manager.get_card_ids_in_zone(opponent_id, Zone.FIELD)

    for card_id in player_field_card_ids:
        available_actions, _ = game.get_available_actions(card_id, current_player)

        if T.ACTION_ATTACK in available_actions:
            opponent_targets_id = [opp_card_id for opp_card_id in opponent_field_card_ids if game.game_state_manager.get_type(opp_card_id) == CardType.FOLLOWER] + [opponent_id]
            for target_id in opponent_targets_id:
                if game.rule_engine.validate_attack(card_id, target_id):
                    possible_actions.append({
                        "type": ATTACK,
                        "attacker_id": card_id,
                        "target_id": target_id
                    })

        if T.ACTION_EVOLVE in available_actions:
            possible_actions.append({"type": EVOLVE, "card_id": card_id})

        if T.ACTION_SUPER_EVOLVE in available_actions:
            possible_actions.append({"type": SUPER_EVOLVE, "card_id": card_id})

        if T.ACTION_ENGAGE in available_actions:
            possible_actions.append({"type": ENGAGE, "card_id": card_id})

    # 3. Ending the turn is always possible.
    possible_actions.append({"type": END_TURN})

    return possible_actions


def apply_action(game: Any, player_id: str, action: Action) -> Any:
    """Execute ``action`` for ``player_id`` through the public Game API.

    Returns whatever the engine call returns. Unknown action types raise; they
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


# --- fixed-size index space ------------------------------------------------

def _slot(ids: List[str], card_id: str) -> int:
    try:
        return ids.index(card_id)
    except ValueError:
        raise IllegalActionError(f"card {card_id!r} is not in the expected zone") from None


def encode(game: Any, player_id: str, action: Action) -> int:
    """Map an action dict onto its index in the 64-way action space."""
    gsm = game.game_state_manager
    kind = action["type"]
    if kind == END_TURN:
        return END_TURN_INDEX
    if kind == PLAY_CARD:
        slot = _slot(gsm.get_card_ids_in_zone(player_id, Zone.HAND), action["card_id"])
        if slot >= HAND_MAX:
            raise IllegalActionError(f"hand slot {slot} exceeds HAND_MAX={HAND_MAX}")
        return _PLAY_BASE + slot * 2 + (1 if action.get("use_extra_pp") else 0)
    my_field = gsm.get_card_ids_in_zone(player_id, Zone.FIELD)
    if kind == ATTACK:
        slot = _slot(my_field, action["attacker_id"])
        opponent_id = game.opponent_id[player_id]
        if action["target_id"] == opponent_id:
            t = FIELD_MAX
        else:
            t = _slot(gsm.get_card_ids_in_zone(opponent_id, Zone.FIELD), action["target_id"])
        return _ATTACK_BASE + slot * 6 + t
    slot = _slot(my_field, action["card_id"])
    if kind == EVOLVE:
        return _EVOLVE_BASE + slot
    if kind == SUPER_EVOLVE:
        return _SUPER_BASE + slot
    if kind == ENGAGE:
        return _ENGAGE_BASE + slot
    raise ValueError(f"Unknown action type: {kind!r}")


def index_table(game: Any, player_id: str, actions: Optional[List[Action]] = None) -> Dict[int, Action]:
    """Return {index: action} for the currently legal actions (or for ``actions``)."""
    if actions is None:
        actions = legal_actions(game, player_id)
    table: Dict[int, Action] = {}
    for action in actions:
        table[encode(game, player_id, action)] = action
    return table


def legal_action_mask(game: Any, player_id: str, actions: Optional[List[Action]] = None) -> List[bool]:
    """Boolean mask of length ACTION_SPACE_SIZE, True where the index is legal now."""
    mask = [False] * ACTION_SPACE_SIZE
    for index in index_table(game, player_id, actions):
        mask[index] = True
    return mask


def decode(game: Any, player_id: str, index: int, actions: Optional[List[Action]] = None) -> Action:
    """Map an index back to the legal action dict, or raise IllegalActionError."""
    table = index_table(game, player_id, actions)
    if index not in table:
        raise IllegalActionError(f"action index {index} is not legal for {player_id} now "
                                 f"(legal: {sorted(table)})")
    return table[index]
