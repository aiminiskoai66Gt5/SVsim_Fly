"""M3: engine fixes found by random self-play. Each test failed before its fix.

The cards used here are real database entries whose parsed effects triggered the
crash or hang; see ENGINE_ISSUES.md (EI-009 .. EI-013) for the seeds.
"""

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
from src.common.effect import Effect, Process
from src.common.enums import EffectType, ProcessType, TargetType, Zone
from svai.env import SVEnv


def find_card(name):
    for db in (card_data.BASIC_CARD_DATABASE, card_data.LEGENDS_RISE_CARD_DATABASE, card_data.TOKEN_CARD_DATABASE):
        for c in db.values():
            if c.name == name:
                return c
    raise KeyError(name)


class EngineCase(unittest.TestCase):
    """A fresh seeded game per test, with helpers to place real cards."""

    def setUp(self):
        self.env = SVEnv()
        with contextlib.redirect_stdout(io.StringIO()):
            self.env.reset(seed=1)
        self.game = self.env.game
        self.gsm = self.game.game_state_manager
        self.sink = io.StringIO()

    def summon(self, name, player_id="player1"):
        with contextlib.redirect_stdout(self.sink):
            card = self.gsm.create_card_instance(find_card(name), player_id)
            self.gsm.add_card(card, Zone.FIELD, player_id)
            self.game.process_events()
        return card

    def resolve(self, effect, caster_id, target_id=None):
        with contextlib.redirect_stdout(self.sink):
            self.game.effect_processor.resolve_effect(effect, caster_id, self.gsm, target_id)
            self.game.process_events()


class TestM3Engine(EngineCase):
    def test_evolved_effect_that_evolves_summoned_followers_terminates(self):
        """EI-011: Deprived Destroyer (Evolved: summon a Bat and evolve it) looped forever
        because the caster re-evolved itself and re-published its own Evolved event."""
        card = self.summon("Deprived Destroyer")
        published = []
        original = self.game.event_manager.publish

        def counting_publish(event):
            published.append(event)
            if len(published) > 500:
                raise AssertionError("event storm: the evolve effect does not terminate")
            original(event)

        self.game.event_manager.publish = counting_publish
        player = self.gsm.players["player1"]
        player.current_sep = 1
        with contextlib.redirect_stdout(self.sink):
            self.game.super_evolve_follower(card.card_id, "player1")
        bats = [c for c in player.field.get_cards() if c.card_data.name == "Bat"]
        self.assertTrue(bats)
        self.assertTrue(all(b.is_evolved for b in bats))
        self.assertLess(len(published), 100)

    def test_evolving_an_evolved_follower_is_a_no_op(self):
        card = self.summon("Deprived Destroyer")
        with contextlib.redirect_stdout(self.sink):
            self.gsm.evolve_card(card.card_id)
        attack_before = card.current_attack
        events_before = len(self.game.event_manager.event_queue)
        proc = Process(process=ProcessType.EVOLVE)
        with contextlib.redirect_stdout(self.sink):
            self.game.effect_processor._process_evolve(proc, card, self.gsm)
        self.assertEqual(card.current_attack, attack_before)
        self.assertEqual(len(self.game.event_manager.event_queue), events_before)

    def test_missing_process_field_reads_as_none(self):
        """EI-010: Process raised AttributeError for an absent optional field."""
        p = Process(process=ProcessType.SPELLBOOST_HAND, target=TargetType.OWN_LEADER)
        self.assertIsNone(p.value)
        self.assertIsNone(p.condition)
        with self.assertRaises(AttributeError):
            _ = p.attributes_that_do_not_exist

    def test_unparsed_process_is_reported_not_crashed(self):
        """EI-009: TRIGGER_EFFECT without value (Alfheimr) crashed at dispatch."""
        card = self.summon("Deprived Destroyer")
        effect = Effect(type=EffectType.FANFARE, processes=[Process(process=ProcessType.TRIGGER_EFFECT)])
        self.resolve(effect, card.card_id)
        hits = self.gsm.unimplemented_hits
        self.assertEqual(len(hits), 1)
        self.assertIn("TRIGGER_EFFECT", hits[0]["reason"])
        self.assertIn("[ERROR] unimplemented effect", self.sink.getvalue())

    def test_raw_text_post_action_is_reported(self):
        """EI-009: SUMMON post_action that is only raw text crashed with 'dict has no process'."""
        card = self.summon("Deprived Destroyer")
        effect = Effect(type=EffectType.FANFARE, processes=[
            Process(process=ProcessType.SUMMON, target=TargetType.OWN_LEADER, value=find_card("Bat"),
                    post_action={"raw_action_text": "give it Rush"})])
        before = len(self.gsm.players["player1"].field.get_cards())
        self.resolve(effect, card.card_id)
        self.assertEqual(len(self.gsm.players["player1"].field.get_cards()), before + 1)
        self.assertEqual(len(self.gsm.unimplemented_hits), 1)

    def test_return_to_deck_with_leader_target_is_reported(self):
        """EI-009: Shakdoh's RETURN_TO_DECK targeted the leader (parser gap) -> AttributeError."""
        card = self.summon("Deprived Destroyer")
        effect = Effect(type=EffectType.FANFARE, processes=[
            Process(process=ProcessType.RETURN_TO_DECK, target=TargetType.OWN_LEADER)])
        self.resolve(effect, card.card_id)
        self.assertEqual(len(self.gsm.unimplemented_hits), 1)

    def test_heal_full_restores_defense(self):
        """EI-012: HEAL value 'full' (Azurifrit super-evolve) was added to an int."""
        card = self.summon("Deprived Destroyer")
        with contextlib.redirect_stdout(self.sink):
            card.take_damage(1)
        self.assertLess(card.current_defense, card.max_defense)
        proc = Process(process=ProcessType.HEAL, target=TargetType.SELF, value="full")
        with contextlib.redirect_stdout(self.sink):
            self.game.effect_processor._process_heal(proc, card, self.gsm)
        self.assertEqual(card.current_defense, card.max_defense)

    def test_trigger_effect_with_count_activates_that_many_abilities(self):
        """EI-013: Omegotep 'activate 2 random abilities' had value=2 -> int has no .value."""
        card = self.summon("Omegotep, the Dreaded One")
        proc = Process(process=ProcessType.TRIGGER_EFFECT, value=2)
        with contextlib.redirect_stdout(self.sink):
            self.game.effect_processor._process_trigger_effect(proc, card, self.gsm)
            self.game.process_events()
        self.assertEqual(len(card.activated_abilities), 2)

    def test_effect_draw_from_empty_deck_is_a_loss(self):
        """EI-007 follow-up: DRAW processes bypassed Game._draw_card and never flagged deck-out."""
        card = self.summon("Deprived Destroyer")
        p1 = self.gsm.players["player1"]
        with contextlib.redirect_stdout(self.sink):
            for c in p1.deck.get_cards():
                self.gsm.move_card(c.card_id, Zone.DECK, Zone.GRAVEYARD)
        effect = Effect(type=EffectType.FANFARE, processes=[
            Process(process=ProcessType.DRAW, target=TargetType.OWN_LEADER, value=1)])
        self.resolve(effect, card.card_id)
        self.assertEqual(self.game.winner(), "player2")


if __name__ == "__main__":
    unittest.main()


class TestCardValidation(unittest.TestCase):
    def test_known_gaps_are_flagged_and_pool_is_large(self):
        from src.common.card_validation import unparsed_reasons
        from svai.env import ensure_card_database
        ensure_card_database("card_database/3_parsed_database/card_database_parsed.json")
        self.assertTrue(unparsed_reasons(find_card("Alfheimr")))                 # TRIGGER_EFFECT without value
        self.assertTrue(unparsed_reasons(find_card("Shakdoh, Nightblossom")))    # RETURN_TO_DECK on leader, raw text
        self.assertFalse(unparsed_reasons(find_card("Indomitable Fighter")))     # plain Enhance stat buff
        pool = {**card_data.BASIC_CARD_DATABASE, **card_data.LEGENDS_RISE_CARD_DATABASE}
        clean = sum(1 for c in pool.values() if not unparsed_reasons(c))
        self.assertGreater(clean, 300)

    def test_clean_decks_never_hit_an_unimplemented_effect(self):
        from svai.arena.core import GameSpec, play_game
        for seed in range(6):
            rec = play_game(GameSpec("random", "random", seed=seed, a_is_player1=seed % 2 == 0,
                                     exclude_unparsed_cards=True))
            self.assertNotEqual(rec.outcome, "exception")
            self.assertEqual(rec.unimplemented_effects, 0, f"seed {seed}")
