# Observation and action encoding (svai.env, M1)

Source of truth: `svai/env.py` (`encode_observation`, constants at the top) and
`svai/actions.py` (index space). This file describes them; if they disagree, the code wins
and this file is wrong.

## Observation: float32 vector of length 315

All values are scaled to roughly [0, 1] by fixed constants (HP/20, PP/10, EP/2, SEP/2,
hand/9, deck/40, graveyard/40, field/5, attack/10, defense/10, cost/10, turn/40). Nothing is
clipped, so a buffed follower can exceed 1. Everything is from the point of view of the
observing player ("me"); the opponent's hand contents are never included, only its size.

| offset | length | content |
|---|---|---|
| 0 | 11 | my leader: hp, max_hp, pp, max_pp, ep, sep, has_extra_pp, hand_size, deck_size, graveyard_size, field_size |
| 11 | 11 | opponent leader: same 11 fields |
| 22 | 9 × 9 = 81 | my hand, slot 0..8 (hand order): present, cost, is_follower, is_spell, is_amulet, attack, defense, playable_now, playable_with_extra_pp |
| 103 | 5 × 21 = 105 | my field, slot 0..4 (field order): present, is_follower, is_amulet, attack, defense, max_defense, cost, can_attack_leader, can_attack_follower, evolved, super_evolved, then 10 keyword flags: ward, storm, rush, bane, drain, barrier, ambush, intimidate, aura, disable |
| 208 | 5 × 21 = 105 | opponent field, same layout |
| 313 | 2 | turn_number / 40, is_my_turn |

Empty slots are all zeros. Followers' attack/defense in hand are the printed values.

Not encoded in M1 (candidates for later): countdown values, crests, hand cards' keywords,
evolution turn restrictions, pending "choose" effects, card identity (no card-id embedding).

## Action space: Discrete(64) with a legal-action mask

`info["legal_mask"]` (bool[64]) and `info["legal_actions"]` (the dicts) are returned by
`reset` and `step`. Stepping with an index whose mask is False raises
`svai.actions.IllegalActionError`; nothing is skipped silently.

| index | meaning |
|---|---|
| 0..17 | PLAY_CARD: `hand_slot * 2 + use_extra_pp` (hand slots 0..8) |
| 18..47 | ATTACK: `18 + attacker_field_slot * 6 + target`, target 0..4 = opponent field slot, 5 = opponent leader |
| 48..52 | EVOLVE field slot 0..4 |
| 53..57 | SUPER_EVOLVE field slot 0..4 |
| 58..62 | ENGAGE field slot 0..4 |
| 63 | END_TURN |

Slots are positions in the engine's zone lists, so they shift when cards leave a zone;
the mask is recomputed after every step.

### Known simplifications (M1)

1. **Enhance** is not a separate choice: a card is played with the largest Enhance cost
   currently affordable, exactly as the original fuzzer did. Cards where "play cheap" and
   "play enhanced" are both sensible cannot express the cheap option.
2. **Effect choices** raised while resolving the controlled player's own card (choose-one
   modes, target selection, discard, fuse, mulligan) are not actions. They are answered by
   the env's `option_agent` (default: first option / keep hand / first cards). Training a
   policy over those choices needs a second decision head; deferred.
3. **Turn structure**: with a scripted `opponent`, the opponent's whole turn runs inside the
   `step` that ends the agent's turn. `opponent=None` gives self-play where `step` applies
   the action for `info["current_player"]`.
4. **Truncation**: `truncated=True` once the engine's turn counter exceeds `max_turns`
   (config, default 40). The scripted opponent is also cut off after
   `opponent_actions_per_turn` (default 30) actions in one turn, mirroring fuzz_runner.

## Reward

+1 when the observing player wins, -1 when it loses, 0 otherwise (draw included).
Win/loss come from `Game.winner()`: a leader at 0 or less defense loses (both at once is
a draw); drawing from an empty deck loses (owner decision, ENGINE_ISSUES EI-007).

## Determinism

`reset(seed)` creates one `random.Random(seed)` that drives deck generation, the engine
(`Game(rng=...)`), and seeds the built-in random opponent. Two envs reset with the same
seed and stepped with the same indices produce identical observations, masks and results,
including identical engine exceptions (`tests/test_m1_env.py`).
`clone()` deep-copies the env, RNG state included.
