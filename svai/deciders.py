"""Concrete deciders: a human behind the existing GUI dialogs, or an Agent."""

from typing import Any, Dict, List, Optional

from src.common.enums import CardType, EffectType
from svai import actions as A
from src.common import text as T
from svai.interfaces import Action, Decider


class AgentDecider(Decider):
    """Delegates every decision to an ``svai.agents.base.Agent``."""

    def __init__(self, agent: Any):
        self.agent = agent

    def choose_action(self, game: Any, player_id: str, legal_actions: Optional[List[Action]]) -> Action:
        if not legal_actions:
            raise ValueError("AgentDecider.choose_action needs a non-empty legal_actions list.")
        action = self.agent.act(game, player_id, legal_actions)
        if action not in legal_actions:
            raise ValueError(f"Agent returned an action that is not legal: {action!r}")
        return action

    def choose_option(self, prompt: str, choices: Dict[str, Any]) -> Any:
        return self.agent.choose_option(prompt, choices)

    def choose_mulligan(self, player_id: str, hand: List[Any]) -> List[str]:
        return self.agent.choose_mulligan(player_id, hand)

    def choose_discard(self, player_id: str, hand: List[Any], count: int) -> List[str]:
        return self.agent.choose_discard(player_id, hand, count)

    def choose_fuse(self, player_id: str, base_card: Any, candidates: List[Any]) -> List[str]:
        return self.agent.choose_fuse(player_id, base_card, candidates)


class HumanDecider(Decider):
    """A human answering through the existing GUI dialogs.

    ``dialogs`` is any object exposing ``get_user_choice``,
    ``get_mulligan_choices`` and ``get_discard_choices`` (i.e. ``GameGUI``).
    The menu flow in ``choose_action`` was moved verbatim from the original
    ``main.py`` loop; the user-facing strings come from ``src.common.text``.
    """

    def __init__(self, dialogs: Any):
        self.dialogs = dialogs

    # --- engine-raised choices -------------------------------------------
    def choose_option(self, prompt: str, choices: Dict[str, Any]) -> Any:
        return self.dialogs.get_user_choice(prompt, choices)

    def choose_mulligan(self, player_id: str, hand: List[Any]) -> List[str]:
        return self.dialogs.get_mulligan_choices(player_id, hand)

    def choose_discard(self, player_id: str, hand: List[Any], count: int) -> List[str]:
        """Use the GUI's discard dialog if it has one, else the generic dialog.

        The original GameGUI has no ``get_discard_choices`` although the engine
        called it, so the fallback below is new behaviour, not a moved one.
        """
        if hasattr(self.dialogs, "get_discard_choices"):
            return self.dialogs.get_discard_choices(player_id, hand, count)
        need = min(count, len(hand))
        return self._pick_cards(T.DISCARD_PROMPT.format(player_id=player_id), hand, need, optional=False)

    def choose_fuse(self, player_id: str, base_card: Any, candidates: List[Any]) -> List[str]:
        """Pick fuse materials one at a time with the generic choice dialog.

        The original GUI never had a fuse dialog (the engine called a method
        that did not exist), so this is new behaviour, not a moved one.
        """
        if hasattr(self.dialogs, "get_fuse_choices"):
            return self.dialogs.get_fuse_choices(player_id, base_card, candidates)
        prompt = T.FUSE_PROMPT.format(player_id=player_id, name=base_card.get_display_name())
        return self._pick_cards(prompt, candidates, len(candidates), optional=True)

    def _pick_cards(self, prompt: str, cards: List[Any], limit: int, optional: bool) -> List[str]:
        """Ask for up to ``limit`` distinct cards, one generic dialog at a time."""
        picked: List[str] = []
        remaining = list(cards)
        while remaining and len(picked) < limit:
            options = {f"{c.get_display_name()} (ID - {c.card_id})": c.card_id for c in remaining}
            if optional:
                options[T.MENU_DONE] = None
            chosen = str(self.dialogs.get_user_choice(f"{prompt} ({len(picked) + 1}/{limit})", options))
            if chosen in ("None", ""):
                break
            picked.append(chosen)
            remaining = [c for c in remaining if c.card_id != chosen]
        return picked

    # --- main-phase action menu (moved from main.py) ----------------------
    def choose_action(self, game: Any, player_id: str, legal_actions: Optional[List[Action]] = None) -> Action:
        """Run the nested menus until the human commits to one action.

        ``legal_actions`` is ignored: the menus validate against the engine
        themselves, exactly as the original loop did.
        """
        ask = self.dialogs.get_user_choice
        gsm = game.game_state_manager
        opponent_id = game.opponent_id[player_id]

        while True:
            # Resolve a pending "choose" effect first, as the original loop did.
            game.process_player_choice()

            current_pp, max_pp, player_field_card_ids, opponent_field_card_ids = game.get_start_turn_ifo(player_id)

            choices = {T.MENU_PLAY_CARD: 0, T.MENU_FIELD: 1, T.MENU_END_TURN: 2}
            choice = int(ask(T.MENU_TITLE, choices))

            if choice == 0:
                use_extra_pp = False
                if game.has_extra_pp(player_id):
                    extra_pp_choices = {T.USE_EXTRA_PP_YES: 0, T.USE_EXTRA_PP_NO: 1}
                    use_extra_pp = int(ask(T.USE_EXTRA_PP_PROMPT, extra_pp_choices)) == 0

                hand_cards_id, is_validate = game.get_playable_cards_id(player_id, use_extra_pp)
                if not hand_cards_id or not any(is_validate):
                    ask(T.NO_PLAYABLE_CARD, {T.MENU_OK: None})
                    continue
                playable_cards_id = [card_id for i, card_id in enumerate(hand_cards_id) if is_validate[i]]

                enhanced_costs = []
                for i, card_id in enumerate(hand_cards_id):
                    if is_validate[i]:
                        enhance_effects = [effect for effect in gsm.get_card_effects(card_id, EffectType.ENHANCE)]
                        enhance_costs_for_card = [effect.enhance_cost for effect in enhance_effects
                                                  if effect.enhance_cost <= current_pp + (1 if use_extra_pp else 0)]
                        enhanced_costs.append(max(enhance_costs_for_card) if enhance_costs_for_card else 0)

                card_choices = {f"{gsm.get_card_name(card_id)} (ID - {card_id})": card_id
                                for card_id in playable_cards_id}
                card_choices[T.MENU_BACK] = None
                selected_card_id = str(ask(T.HAND_TITLE, card_choices))

                if selected_card_id == "None":
                    print(T.CHOOSE_AGAIN)
                    continue

                enhanced_cost = enhanced_costs[playable_cards_id.index(selected_card_id)]
                return {"type": A.PLAY_CARD, "card_id": selected_card_id,
                        "enhanced_cost": enhanced_cost, "use_extra_pp": use_extra_pp}

            elif choice == 1:
                if not player_field_card_ids:
                    ask(T.NO_FIELD_CARD, {T.MENU_OK: None})
                    continue

                card_choices = {f"{gsm.get_card_name(card_id)} (ID - {card_id})": card_id
                                for card_id in player_field_card_ids}
                card_choices[T.MENU_BACK] = None
                selected_card_id = str(ask(T.FIELD_TITLE, card_choices))

                if selected_card_id == "None":
                    print(T.CHOOSE_AGAIN)
                    continue

                available_actions, card_name = game.get_available_actions(selected_card_id, player_id)
                if not available_actions:
                    ask(T.NO_ACTION_FOR_CARD, {T.MENU_OK: None})
                    continue

                action_choices = {action: action for action in available_actions}
                action_choices[T.MENU_CANCEL] = None
                chosen_action = str(ask(T.ACTION_FOR_CARD_TITLE.format(name=card_name), action_choices))

                if chosen_action == "None":
                    print(T.CHOOSE_AGAIN)
                    continue

                if chosen_action == T.ACTION_ATTACK:
                    print("\n" + T.TARGET_TITLE)
                    opponent_targets_id = [card_id for card_id in opponent_field_card_ids
                                           if gsm.get_type(card_id) == CardType.FOLLOWER] + [opponent_id]
                    possible_targets_id = [target_id for target_id in opponent_targets_id
                                           if game.rule_engine.validate_attack(selected_card_id, target_id)]
                    if not possible_targets_id:
                        ask(T.NO_ATTACK_TARGET, {T.MENU_OK: None})
                        continue

                    target_choices = {f"{gsm.get_card_name(target_id)} (ID - {target_id})": target_id
                                      for target_id in possible_targets_id}
                    target_choices[T.MENU_CANCEL] = None
                    selected_target_id = str(ask(T.TARGET_TITLE, target_choices))

                    # The original compared against None (never true for a str) and then
                    # issued an attack on the id "None", which the engine rejected as a
                    # no-op. Treating it as "cancel" here has the same observable result.
                    if selected_target_id == "None":
                        print(T.CHOOSE_AGAIN)
                        continue

                    return {"type": A.ATTACK, "attacker_id": selected_card_id, "target_id": selected_target_id}

                elif chosen_action == T.ACTION_EVOLVE:
                    return {"type": A.EVOLVE, "card_id": selected_card_id}

                elif chosen_action == T.ACTION_SUPER_EVOLVE:
                    return {"type": A.SUPER_EVOLVE, "card_id": selected_card_id}

                elif chosen_action == T.ACTION_ENGAGE:
                    return {"type": A.ENGAGE, "card_id": selected_card_id}

            elif choice == 2:
                ask(T.END_TURN_CONFIRM.format(player_id=player_id), {T.MENU_OK: None})
                return {"type": A.END_TURN}
            else:
                ask(T.INVALID_CHOICE, {T.MENU_OK: None})

    def notify_action_applied(self, game: Any, player_id: str, action: Action) -> None:
        """Show the same confirmation dialogs the original loop showed."""
        gsm = game.game_state_manager
        ask = self.dialogs.get_user_choice
        kind = action["type"]
        if kind == A.EVOLVE:
            ask(T.EVOLVED_CONFIRM.format(name=gsm.get_card_name(action['card_id'])), {T.MENU_OK: None})
        elif kind == A.SUPER_EVOLVE:
            ask(T.SUPER_EVOLVED_CONFIRM.format(name=gsm.get_card_name(action['card_id'])), {T.MENU_OK: None})
        elif kind == A.ENGAGE:
            ask(T.ENGAGED_CONFIRM.format(name=gsm.get_card_name(action['card_id'])), {T.MENU_OK: None})
