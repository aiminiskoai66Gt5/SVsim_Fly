"""M1: Gymnasium env, injected RNG, game-over, action index space, clone()."""

import contextlib
import copy
import io
import random
import sys
import types
import unittest
from pathlib import Path

# The container running these tests may lack tkinter; deck_builder imports it at module level.
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

import numpy as np

from src.common.enums import Zone
from src.engine.main_game_logic import Game
from svai import actions as A
from svai.agents.random_agent import RandomAgent
from svai.config import ForbiddenPathError, assert_allowed_path, load_config
from svai.env import OBS_SIZE, SVEnv


def play_random_game(seed, policy_seed=None, max_steps=2000):
    """Play one env game with a seeded uniform policy; return a trace."""
    env = SVEnv()
    obs, info = env.reset(seed=seed)
    pol = random.Random(seed if policy_seed is None else policy_seed)
    trace = [obs.copy()]
    done = False
    steps = 0
    reward = 0.0
    while not done and steps < max_steps:
        legal = [i for i, m in enumerate(info["legal_mask"]) if m]
        try:
            obs, reward, term, trunc, info = env.step(pol.choice(legal))
        except Exception as exc:  # engine bugs (M3) must reproduce too
            trace.append(f"{type(exc).__name__}: {exc}")
            info = dict(info, winner="exception", turn=env.game.game_state_manager.turn_number)
            break
        trace.append(obs.copy())
        done = term or trunc
        steps += 1
    return env, trace, reward, info


class TestM1Env(unittest.TestCase):
    def test_observation_shape_and_range(self):
        env = SVEnv()
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_SIZE,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info["legal_mask"].shape, (A.ACTION_SPACE_SIZE,))
        self.assertTrue(info["legal_mask"][A.END_TURN_INDEX])

    def test_same_seed_reproduces_the_whole_game(self):
        for seed in (1, 7, 42):
            env_a, trace_a, r_a, info_a = play_random_game(seed)
            env_b, trace_b, r_b, info_b = play_random_game(seed)
            self.assertEqual(env_a.action_log, env_b.action_log, f"seed {seed}: action sequence differs")
            self.assertEqual(len(trace_a), len(trace_b))
            for oa, ob in zip(trace_a, trace_b):
                if isinstance(oa, str) or isinstance(ob, str):
                    self.assertEqual(oa, ob)
                else:
                    np.testing.assert_array_equal(oa, ob)
            self.assertEqual((r_a, info_a["winner"], info_a["turn"]), (r_b, info_b["winner"], info_b["turn"]))

    def test_different_seeds_differ(self):
        env_a, trace_a, _, _ = play_random_game(1)
        env_b, trace_b, _, _ = play_random_game(2)
        self.assertNotEqual(env_a.action_log, env_b.action_log)

    def test_illegal_action_raises_instead_of_being_skipped(self):
        env = SVEnv()
        _, info = env.reset(seed=3)
        illegal = [i for i, m in enumerate(info["legal_mask"]) if not m]
        self.assertTrue(illegal)
        before = copy.deepcopy(env.action_log)
        with self.assertRaises(A.IllegalActionError):
            env.step(illegal[0])
        self.assertEqual(env.action_log, before)

    def test_encode_decode_round_trip(self):
        env = SVEnv()
        _, info = env.reset(seed=5)
        # walk a few steps so the field is populated
        pol = random.Random(5)
        for _ in range(12):
            if env.game.is_game_over():
                break
            legal = [i for i, m in enumerate(info["legal_mask"]) if m]
            _, _, term, trunc, info = env.step(pol.choice(legal))
            if term or trunc:
                break
        game, player = env.game, env.current_player
        with contextlib.redirect_stdout(io.StringIO()):
            actions = A.legal_actions(game, player)
            indices = [A.encode(game, player, a) for a in actions]
            self.assertEqual(len(indices), len(set(indices)), "two legal actions share an index")
            for idx, action in zip(indices, actions):
                self.assertEqual(A.decode(game, player, idx, actions), action)

    def test_clone_is_independent_and_deterministic(self):
        env = SVEnv()
        _, info = env.reset(seed=11)
        clone = env.clone()
        legal = [i for i, m in enumerate(info["legal_mask"]) if m]
        # Same action on both -> identical result; original untouched by the clone's later steps.
        o1, _, _, _, i1 = env.step(legal[-1])
        o2, _, _, _, i2 = clone.step(legal[-1])
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(i1["legal_mask"], i2["legal_mask"])
        legal2 = [i for i, m in enumerate(i2["legal_mask"]) if m]
        clone.step(legal2[0])
        np.testing.assert_array_equal(env._observation(), o1)
        self.assertIs(clone.game.game_state_manager.rng, clone.rng)
        self.assertIsNot(clone.game, env.game)

    def test_clone_keeps_card_listener_ids_consistent(self):
        """After deepcopy, the ids the engine would use to unsubscribe a card's
        listeners must still match the ids registered in the event manager."""
        env = SVEnv()
        _, info = env.reset(seed=13)
        pol = random.Random(13)
        for _ in range(40):
            if env.game.is_game_over():
                break
            legal = [i for i, m in enumerate(info["legal_mask"]) if m]
            _, _, term, trunc, info = env.step(pol.choice(legal))
            if term or trunc:
                break
        clone = env.clone()
        em = clone.game.event_manager
        gsm = clone.game.game_state_manager
        cards = {c.card_id: c for pid in gsm.players for zone in (Zone.FIELD, Zone.HAND)
                 for c in gsm.get_cards_in_zone(pid, zone)}
        checked = 0
        for listeners in em.listeners.values():
            for listener in listeners:
                if not listener.card_id or listener.card_id not in cards:
                    continue
                card = cards[listener.card_id]
                expected = {clone.game._listener_id(card, eff) for _, eff in card.card_data.required_listeners}
                checked += 1
                self.assertIn(listener.id, expected,
                              f"{card.get_display_name()}: registered id {listener.id} no longer "
                              f"derivable after deepcopy -> could never be unsubscribed")
        self.assertGreater(checked, 0, "seed produced no card listeners; pick another seed")

    def test_deck_out_is_a_loss(self):
        env = SVEnv()
        env.reset(seed=17)
        game = env.game
        gsm = game.game_state_manager
        p2 = gsm.players["player2"]
        with contextlib.redirect_stdout(io.StringIO()):
            for card in p2.deck.get_cards():
                gsm.move_card(card.card_id, Zone.DECK, Zone.GRAVEYARD)
            self.assertIsNone(game.winner())
            game.end_turn("player1")  # player2 draws from an empty deck
        self.assertEqual(game.winner(), "player1")
        self.assertTrue(game.is_game_over())

    def test_leader_defense_zero_is_a_loss_and_double_zero_is_a_draw(self):
        env = SVEnv()
        env.reset(seed=19)
        game = env.game
        gsm = game.game_state_manager
        gsm.players["player1"].current_defense = 0
        self.assertEqual(game.winner(), "player2")
        gsm.players["player2"].current_defense = 0
        self.assertEqual(game.winner(), Game.DRAW)

    def test_step_after_game_over_is_refused(self):
        env = SVEnv()
        env.reset(seed=23)
        env.game.game_state_manager.players["player2"].current_defense = 0
        with self.assertRaises(RuntimeError):
            env.step(A.END_TURN_INDEX)

    def test_self_play_mode_alternates_players(self):
        env = SVEnv(opponent=None)
        _, info = env.reset(seed=29)
        self.assertEqual(info["current_player"], "player1")
        _, _, _, _, info = env.step(A.END_TURN_INDEX)
        self.assertEqual(info["current_player"], "player2")

    def test_random_agent_is_reproducible(self):
        a = RandomAgent(random.Random(1)).act(None, "p", [{"i": i} for i in range(50)])
        b = RandomAgent(random.Random(1)).act(None, "p", [{"i": i} for i in range(50)])
        self.assertEqual(a, b)

    def test_config_paths_are_resolved_and_c_drive_is_refused(self):
        cfg = load_config()
        for key in ("data_root", "datasets", "cache", "checkpoints", "logs", "tmp"):
            self.assertTrue(Path(cfg["paths"][key]).is_absolute())
        if sys.platform.startswith("win"):
            with self.assertRaises(ForbiddenPathError):
                assert_allowed_path(Path("C:/Users/someone/AppData/Local/svai"))
        else:
            with self.assertRaises(ForbiddenPathError):
                assert_allowed_path(Path("/mnt/c/Users/someone/svai"))


if __name__ == "__main__":
    unittest.main()
