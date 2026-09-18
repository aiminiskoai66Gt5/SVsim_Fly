# Engine issues

Grouped by root cause. Status: FIXED (engine changed, test added), AGENT-SIDE (worked around outside
the engine, marked in code), OPEN (recorded only). Rule behaviour is never changed to help a bot.

## A. Presentation / decision coupling (found in M0)

### EI-001 `List` used but never imported — FIXED
`src/engine/main_game_logic.py` imported `Dict, Any` only while `Game.__init__` annotates with
`List[Any]`, so `import src.engine.main_game_logic` raised `NameError` at HEAD e4d74e5.
Repro: `python -c "import src.engine.main_game_logic"`. Fix: import `List`.

### EI-002 Fuse asks the GUI for a method that does not exist — FIXED
`effect_processor._process_fuse` called `game.gui.get_fuse_choices(...)`; neither `GameGUI` nor
`MockGUI` defined it, so any fuse with a valid material would raise `AttributeError`.
Not reached in 200 random games (seeds 0-199), found by reading. Now routed to
`Decider.choose_fuse` (default: fuse nothing). `HumanDecider` asks with the generic choice dialog.

### EI-003 Manual discard asks the GUI for a method that does not exist — FIXED (decider side)
`Game.discard_cards_manually` called `gui.get_discard_choices`, which `GameGUI` never had
(only `MockGUI` did). Now `Decider.choose_discard`; `HumanDecider` falls back to the generic dialog.

### EI-004 Effect choices do not say who is choosing — OPEN
`Game.request_user_choice(prompt, choices)` carries no player id, and the ~9 call sites in
`effect_processor.py` do not pass one. M0 routes such choices to the turn player's decider.
That is wrong for an effect owned by the non-turn player (e.g. a Last Words choice on the
opponent's turn). Matters as soon as the two players have different deciders (human vs agent).

### EI-005 Choice buttons can be clipped out of the 1200x800 window — OPEN (upstream, GUI only)
`GameGUI.get_user_choice` re-packs `choice_frame` last, after both expanding player frames. When the
board is tall enough the frame gets no space and the buttons are never mapped. Seen on a virtual
display (Xvfb, default fonts) on both upstream and this branch; with a 1900x2400 window it never
happens. Not verified on Windows. Repro: `xvfb-run python scripts/gui_smoke.py` (reports the count).

### EI-006 `fuzz_runner.py` needs tkinter although it is headless — OPEN
It imports `generate_random_deck` from `deck_builder.py`, whose module top imports tkinter.

## B. Win / loss (decision recorded, not implemented yet)

### EI-007 The engine has no game-over state — FIXED (M1)
No winner / game-over flag anywhere. `main.py` plays 20 turns and never looks at leader defense;
only `fuzz_runner.py` stops at defense <= 0. `_draw_card` on an empty deck prints and returns.
Owner decision (2026-09-18): drawing from an empty deck is a loss. Implemented as
`Game.winner()` / `is_game_over()` (leader defense <= 0 loses, both = draw; first player to draw
from an empty deck loses). The engine still never refuses actions after game over; callers
(env, arena) must check. Tests: `test_deck_out_is_a_loss`, `test_leader_defense_zero_...`.

### EI-008 Listener ids embedded `id(effect)` — FIXED (M1)
`_register_card_listeners` named listeners `f"{card_id}_{type}_{id(effect)}"`. After
`copy.deepcopy(game)` every effect object has a new address, so a cloned game could never
unsubscribe those listeners (they would fire for cards that already left the field). Now the
id uses the effect's index in `card_data.required_listeners`. Test:
`test_clone_keeps_card_listener_ids_consistent` (fails on the old scheme).

## C. Found by random self-play in M3 (2,000 + 10,000 games, seeds 0-1999 / 100000-109999)

Before any fix: 732/2000 games (36.6%) raised, 22 hung (1.1%). Seven distinct crash sites and
one hang, grouped below by root cause. After: 0 exceptions and 0 hangs in 12,200 games.
Every entry has a test in `tests/test_m3_engine.py` that fails on the unfixed engine.

### EI-009 Card data the parser only partly parsed reaches handlers that assume complete data — FIXED (reported, not crashed)
Root cause of 5 of the 7 crash sites (`effect.py:75`, `effect_processor.py:177/747/1028`,
`card.py:17`). The upstream text parser leaves gaps in `card_database_parsed.json`: processes
that are only `raw_action_text` (70 post_actions), `TRIGGER_EFFECT` / `REMOVE_KEYWORD` without
`value` (7), `RETURN_TO_DECK` targeted at a leader ("Return your hand to deck" -> OWN_LEADER),
`TRANSFORM` into a phrase the resolver could not map ("copies of Skeleton"), plain dicts inside
process lists. Handlers dereferenced the missing pieces and crashed the whole game.
The card database is owned by the pipeline and is not modified. Instead the engine now checks
a process before dispatch (`EffectProcessor._process_problem`) and, when it cannot execute it,
records the hit in `game_state_manager.unimplemented_hits`, prints `[ERROR] unimplemented
effect ...` and skips that process only. This is a documented compromise, not a silent skip:
the arena reports "games with an unimplemented card effect" (97% of random-deck games, because
only 413 of the 657 non-token cards are fully executable — `src/common/card_validation.py`).
`config.toml env.exclude_unparsed_cards = true` (or `--exclude-unparsed`) restricts random
decks to fully executable cards, transitively including the tokens they create; with it,
0 hits in 200 games. Seeds: 0 (post_action dict), 4 (TRIGGER_EFFECT), 17 (leader target),
57 (dict in variable resolution), 76 (unresolved TRANSFORM).

### EI-010 `Process.__getattr__` raised for an absent optional field — FIXED
`Process` raised `AttributeError` for `value`/`condition`/... when the parser had not emitted
them, while `Effect` returned None for the same fields. Handlers written as
`if effect_data.value is not None` (e.g. `_process_spellboost_hand`, Paper Shikigami's Last
Words) therefore crashed. Now both classes read a missing core field as None; the parent
effect is still never consulted for these names. Seed 16 after the EI-009 fixes.

### EI-011 Evolve-on-evolved loops forever — FIXED (hang root cause)
Deprived Destroyer: "When this follower evolves, summon a Bat and evolve it". `EVOLVE` with
target `SUMMONED_FOLLOWERS` included the caster itself (the recently-summoned list is only
cleared when a card is played), `_process_evolve` re-evolved an already evolved follower and
re-published `FollowerEvolvedEvent`, which re-triggered the same Evolved effect: an unbounded
event loop. All 22 hangs in 2,000 games were this card. `_process_evolve` now ignores
non-followers and followers that are already evolved or super-evolved (a follower evolves at
most once). Seed 63; test `test_evolved_effect_that_evolves_summoned_followers_terminates`.

### EI-012 `HEAL` with value "full" — FIXED
Azurifrit, Heir to Disdain (Super-Evolve: fully restore defense) was parsed as `value: "full"`
and added to an int (`card.py:65`). "full"/"fully"/"all" now heal to max defense; any other
non-numeric value is reported as unimplemented. Seed 79.

### EI-013 `TRIGGER_EFFECT` with a count — FIXED
Omegotep, the Dreaded One ("Activate 2 random abilities from the following") is parsed as
`TRIGGER_EFFECT value=2`; the handler only understood an EffectType or the string
`random_unactivated` (`effect_processor.py:1059`). An int N now activates N distinct random
unactivated abilities, reusing the existing `random_unactivated` path. Seed 37.

### EI-014 Effect draws from an empty deck were not a loss — FIXED
`_process_draw` bypassed `Game._draw_card`, so a "Draw a card" effect on an empty deck just
logged. It now sets `deck_out_player_id` like the turn-start draw (owner decision, EI-007).

### EI-015 Target types without a handler were logged to stderr only — FIXED (reported)
`_invoke_target_handler` printed "Target type X has no handler" via `logging` and returned no
targets; the effect silently did nothing. It now goes through the same unimplemented report.
Two target types are affected in the database (`ANOTHER_ALLY_FOLLOWER_RANDOM_UNEVOLVED_NO_ATTACK`,
`ALL_NON_ENCROACHER_FOLLOWERS`).

## D. Seen while reading, not yet reproduced
- `attack_follower`: when the attacker has Barrier, `attacker.effects` is rebuilt from `target.effects`.
- `attack_follower`: `target.card_data['name']` in the Barrier log line (elsewhere it is an attribute).
- `Game._on_turn_start` is defined twice; the second definition silently replaces the first.
- `Game._on_damage_dealt` (Drain) is never subscribed to any event.
