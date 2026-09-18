"""M4: GreedyAgent lethal detection, PP subset choice, attack and evolve rules."""

import contextlib
import io
import sys
import types
import unittest

if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        class _Stub(types.ModuleType):
            def __getattr__(self, n):
                if n.startswith("__"):
                    raise AttributeError(n)
                return type(n, (), {})
        for _m in ("tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.simpledialog"):
            sys.modules[_m] = _Stub(_m)

from src.common import card_data
from src.common.enums import CardType, EffectType, Zone
from svai import actions as A
from svai.agents.greedy_agent import GreedyAgent, GreedyWeights
from svai.env import SVEnv


def find_card(name):
    for db in (card_data.BASIC_CARD_DATABASE, card_data.LEGENDS_RISE_CARD_DATABASE, card_data.TOKEN_CARD_DATABASE):
        for c in db.values():
            if c.name == name:
                return c
    raise KeyError(name)


_seq = [0]


def vanilla(cost, attack, defense):
    """A synthetic follower with no effects (the database has almost none)."""
    from src.common.enums import ClassType
    _seq[0] += 1
    return card_data.CardData(f"T{_seq[0]}", f"Vanilla {cost}/{attack}/{defense}", cost, CardType.FOLLOWER,
                              ClassType.NEUTRAL, attack, defense)


class GreedyCase(unittest.TestCase):
    def setUp(self):
        self.env = SVEnv()
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.reset(seed=2)
        self.game = self.env.game
        self.gsm = self.game.game_state_manager
        self.me, self.opp = "player1", "player2"
        self.agent = GreedyAgent()
        self.sink = io.StringIO()

    def place(self, card_data_obj, player_id, ready=True):
        with contextlib.redirect_stdout(self.sink):
            card = self.gsm.create_card_instance(card_data_obj, player_id)
            self.gsm.add_card(card, Zone.FIELD, player_id)
            self.game.process_events()
        if ready:
            card.is_summoned = False  # may attack this turn
        return card

    def hand(self, card_data_obj, player_id):
        with contextlib.redirect_stdout(self.sink):
            card = self.gsm.create_card_instance(card_data_obj, player_id)
            self.gsm.add_card(card, Zone.HAND, player_id)
        return card

    def clear_field(self, player_id):
        with contextlib.redirect_stdout(self.sink):
            for c in list(self.gsm.players[player_id].field.get_cards()):
                self.gsm.move_card(c.card_id, Zone.FIELD, Zone.GRAVEYARD)

    def legal(self):
        with contextlib.redirect_stdout(self.sink):
            return A.legal_actions(self.game, self.me)

    def act(self):
        with contextlib.redirect_stdout(self.sink):
            return self.agent.act(self.game, self.me, self.legal())

    def vanilla_with(self, attack, defense=None):
        return vanilla(attack, attack, defense if defense is not None else attack)


class TestLethal(GreedyCase):
    def test_takes_lethal_with_board_attackers(self):
        self.clear_field(self.opp)
        a = self.place(self.vanilla_with(2), self.me)
        b = self.place(self.vanilla_with(2), self.me)
        self.gsm.players[self.opp].current_defense = a.current_attack + b.current_attack
        first = self.act()
        self.assertEqual(first["type"], A.ATTACK)
        self.assertEqual(first["target_id"], self.opp)
        # follow the plan to the end: the opponent must be dead
        with contextlib.redirect_stdout(self.sink):
            A.apply_action(self.game, self.me, first)
            while not self.game.is_game_over():
                action = self.agent.act(self.game, self.me, A.legal_actions(self.game, self.me))
                if action["type"] == A.END_TURN:
                    break
                A.apply_action(self.game, self.me, action)
        self.assertEqual(self.game.winner(), self.me)
        self.assertGreaterEqual(self.agent.stats["lethal_verified"], 1)

    def test_no_lethal_means_no_face_race_when_a_good_trade_exists(self):
        self.clear_field(self.opp)
        self.clear_field(self.me)
        mine = self.place(self.vanilla_with(3, 4), self.me)
        enemy = self.place(vanilla(2, 2, 3), self.opp)  # dies to 3 attack, deals 2 < 4
        self.gsm.players[self.opp].current_defense = 20
        action = self.act()
        self.assertEqual(action["type"], A.ATTACK)
        self.assertEqual(action["target_id"], enemy.card_id)

    def test_ward_blocks_lethal_search(self):
        self.clear_field(self.opp)
        mine = self.place(self.vanilla_with(2), self.me)
        ward = None
        for db in (card_data.BASIC_CARD_DATABASE, card_data.LEGENDS_RISE_CARD_DATABASE):
            for c in db.values():
                if c.card_type == CardType.FOLLOWER and any(e.type == EffectType.WARD for e in c.effects) \
                        and all(e.type == EffectType.WARD for e in c.effects):
                    ward = c
                    break
            if ward:
                break
        if ward is None:
            self.skipTest("no plain Ward follower in the database")
        self.place(ward, self.opp)
        self.gsm.players[self.opp].current_defense = 1
        line = self.agent._find_lethal(self.game, self.me, self.legal())
        self.assertIsNone(line)


class TestPlayAndEvolve(GreedyCase):
    def test_pp_subset_leaves_the_least_pp_unspent(self):
        me = self.gsm.players[self.me]
        with contextlib.redirect_stdout(self.sink):
            for c in list(me.hand.get_cards()):
                self.gsm.move_card(c.card_id, Zone.HAND, Zone.GRAVEYARD)
        self.clear_field(self.me)
        for k in (1, 2, 3, 4):
            self.hand(vanilla(k, k, k), self.me)
        me.max_pp = me.current_pp = 5
        chosen_costs = []
        with contextlib.redirect_stdout(self.sink):
            while True:
                legal = A.legal_actions(self.game, self.me)
                action = self.agent.act(self.game, self.me, legal)
                if action["type"] != A.PLAY_CARD:
                    break
                chosen_costs.append(self.gsm.get_entity_by_id(action["card_id"]).current_cost)
                A.apply_action(self.game, self.me, action)
        self.assertEqual(sum(chosen_costs), 5)          # 5 PP fully used (4+1 or 3+2)
        self.assertEqual(me.current_pp, 0)

    def test_evolve_only_when_it_flips_a_trade(self):
        self.clear_field(self.opp)
        self.clear_field(self.me)
        me = self.gsm.players[self.me]
        me.current_ep = 2
        me.spent_ep_in_turn = False
        mine = self.place(self.vanilla_with(2, 2), self.me, ready=False)
        # an enemy our 2/2 cannot kill now (defense 4) but a 4/4 kills and survives (attack 3)
        enemy = self.place(vanilla(4, 3, 4), self.opp)
        action = self.act()
        self.assertIn(action["type"], (A.EVOLVE, A.SUPER_EVOLVE))
        self.assertEqual(action["card_id"], mine.card_id)

    def test_does_not_evolve_for_nothing(self):
        self.clear_field(self.opp)
        self.clear_field(self.me)
        me = self.gsm.players[self.me]
        me.current_ep = 2
        me.spent_ep_in_turn = False
        self.place(self.vanilla_with(2), self.me, ready=False)
        with contextlib.redirect_stdout(self.sink):
            for c in list(me.hand.get_cards()):
                self.gsm.move_card(c.card_id, Zone.HAND, Zone.GRAVEYARD)
        action = self.act()
        self.assertEqual(action["type"], A.END_TURN)


class TestWeights(unittest.TestCase):
    def test_all_constants_live_in_the_dataclass(self):
        w = GreedyWeights()
        self.assertGreater(w.max_subset_size, 0)
        self.assertGreater(w.evolve_trade_gain_required, 0)


if __name__ == "__main__":
    unittest.main()
