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

## C. Seen while reading, not yet reproduced (candidates for M3)
- `attack_follower`: when the attacker has Barrier, `attacker.effects` is rebuilt from `target.effects`.
- `attack_follower`: `target.card_data['name']` in the Barrier log line (elsewhere it is an attribute).
- `Game._on_turn_start` is defined twice; the second definition silently replaces the first.
- `Game._on_damage_dealt` (Drain) is never subscribed to any event.
