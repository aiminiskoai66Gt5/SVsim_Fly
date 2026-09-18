"""M0: the engine talks to View / Decider only, never to a GUI object."""

import io
import contextlib
import subprocess
import sys
import unittest

from src.common import card_data
from src.engine.main_game_logic import Game
from svai import actions as A
from svai.deciders import AgentDecider, HumanDecider
from svai.agents.base import Agent
from svai.interfaces import Decider, NullView, View
from src.common import text as T

DB_PATH = "card_database/3_parsed_database/card_database_parsed.json"


class RecordingDecider(Decider):
    """Keeps every mulligan card and records who was asked what."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def choose_action(self, game, player_id, legal_actions):
        self.calls.append(("action", player_id))
        return {"type": A.END_TURN}

    def choose_option(self, prompt, choices):
        self.calls.append(("option", prompt))
        return next(iter(choices.values()))

    def choose_mulligan(self, player_id, hand):
        self.calls.append(("mulligan", player_id, len(hand)))
        return []

    def choose_discard(self, player_id, hand, count):
        self.calls.append(("discard", player_id, count))
        return [c.card_id for c in hand[:count]]


class CountingView(View):
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


class ScriptedDialogs:
    """Stands in for GameGUI: answers get_user_choice from a list of labels."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.prompts = []

    def get_user_choice(self, prompt, choices):
        self.prompts.append(prompt)
        label = self.labels.pop(0)
        value = choices[label] if label in choices else choices[next(k for k in choices if k.startswith(label))]
        return str(value)  # GameGUI returns a tkinter StringVar value, i.e. always a str.

    def get_mulligan_choices(self, player_id, hand):
        return []


def new_game(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return Game("player1", "player2", **kwargs)


class TestM0Interfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            card_data.load_card_databases(DB_PATH)

    def test_engine_imports_and_runs_without_tkinter(self):
        code = (
            "import sys, io, contextlib\n"
            "sys.modules['tkinter'] = None\n"  # any 'import tkinter' now raises ImportError
            "from src.common import card_data\n"
            "from src.engine.main_game_logic import Game\n"
            "from svai.interfaces import NullView\n"
            "from tests.test_m0_interfaces import RecordingDecider\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            f"    card_data.load_card_databases({DB_PATH!r})\n"
            "    g = Game('player1', 'player2', view=NullView(), decider=RecordingDecider('x'))\n"
            "    g.end_turn('player1')\n"
            "print('OK', g.game_state_manager.current_turn_player_id)\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.stdout.strip(), "OK player2", res.stderr[-2000:])

    def test_mulligan_is_routed_to_each_players_decider(self):
        d1, d2 = RecordingDecider("p1"), RecordingDecider("p2")
        game = new_game(view=NullView(), decider={"player1": d1, "player2": d2})
        self.assertEqual(d1.calls, [("mulligan", "player1", 4)])
        self.assertEqual(d2.calls, [("mulligan", "player2", 4)])
        self.assertIs(game.decider_for("player2"), d2)
        self.assertIs(game.decider_for(), d1)  # turn player by default

    def test_view_receives_updates_and_never_decides(self):
        view = CountingView()
        game = new_game(view=view, decider=RecordingDecider("x"))
        before = view.updates
        with contextlib.redirect_stdout(io.StringIO()):
            game.end_turn("player1")
        self.assertGreater(view.updates, before)
        self.assertFalse(hasattr(view, "get_user_choice"))

    def test_request_user_choice_goes_through_decider(self):
        d = RecordingDecider("x")
        game = new_game(view=NullView(), decider=d)
        self.assertEqual(game.request_user_choice("pick", {"a": 1, "b": 2}), 1)
        self.assertIn(("option", "pick"), d.calls)

    def test_apply_action_rejects_unknown_type(self):
        game = new_game(view=NullView(), decider=RecordingDecider("x"))
        with self.assertRaises(ValueError):
            A.apply_action(game, "player1", {"type": "DANCE"})

    def test_human_decider_end_turn_flow_matches_original_menu(self):
        dialogs = ScriptedDialogs([T.MENU_END_TURN, T.MENU_OK])
        human = HumanDecider(dialogs)
        game = new_game(view=NullView(), decider=human)
        with contextlib.redirect_stdout(io.StringIO()):
            action = human.choose_action(game, "player1", None)
        self.assertEqual(action, {"type": A.END_TURN})
        self.assertEqual(dialogs.prompts, [T.MENU_TITLE, T.END_TURN_CONFIRM.format(player_id="player1")])

    def test_human_decider_play_card_and_back_navigation(self):
        game = new_game(view=NullView(), decider=RecordingDecider("x"))
        gsm = game.game_state_manager
        player = gsm.players["player1"]
        player.max_pp = player.current_pp = 10  # test setup only: make the whole hand affordable
        with contextlib.redirect_stdout(io.StringIO()):
            hand_ids, ok = game.get_playable_cards_id("player1", False)
        playable = [cid for cid, flag in zip(hand_ids, ok) if flag]
        self.assertTrue(playable)
        label = f"{gsm.get_card_name(playable[0])} (ID - {playable[0]})"
        # open hand -> go back -> open hand again -> pick the card
        dialogs = ScriptedDialogs([T.MENU_PLAY_CARD, T.MENU_BACK, T.MENU_PLAY_CARD, label])
        with contextlib.redirect_stdout(io.StringIO()):
            action = HumanDecider(dialogs).choose_action(game, "player1", None)
        self.assertEqual(action["type"], A.PLAY_CARD)
        self.assertEqual(action["card_id"], playable[0])
        self.assertFalse(action["use_extra_pp"])

    def test_card_display_language(self):
        game = new_game(view=NullView(), decider=RecordingDecider("x"))
        card = game.game_state_manager.players["player1"].hand.get_cards()[0]
        old = card_data.DISPLAY_LANGUAGE
        try:
            card_data.DISPLAY_LANGUAGE = "en"
            self.assertEqual(card.get_display_name(), card.card_data.name)
            card_data.DISPLAY_LANGUAGE = "zh_tw"
            by_id = card_data.ZH_TW_BY_ID.pop(str(card.card_data.card_id), None)
            saved = card_data.ZH_TW_NAME_MAP.pop(card.card_data.name, None)
            card_data.ZH_TW_NAME_MAP[card.card_data.name] = "測試名"
            self.assertEqual(card.get_display_name(), "測試名")
            del card_data.ZH_TW_NAME_MAP[card.card_data.name]
            self.assertEqual(card.get_display_name(), card.card_data.name)  # falls back to English
            if by_id is not None:
                card_data.ZH_TW_BY_ID[str(card.card_data.card_id)] = by_id
                self.assertEqual(card.get_display_name(), by_id["name_zh_tw"])  # official name by id wins
            if saved is not None:
                card_data.ZH_TW_NAME_MAP[card.card_data.name] = saved
        finally:
            card_data.DISPLAY_LANGUAGE = old

    def test_agent_decider_rejects_illegal_action(self):
        class Liar(Agent):
            def act(self, game, player_id, legal_actions):
                return {"type": A.EVOLVE, "card_id": "nope"}

        with self.assertRaises(ValueError):
            AgentDecider(Liar()).choose_action(None, "player1", [{"type": A.END_TURN}])


if __name__ == "__main__":
    unittest.main()
