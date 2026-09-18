"""Rule-based practice opponent (M4). No machine learning.

Decision order each time ``act`` is called:

1. Lethal: enumerate subsets of playable cards (direct damage spells, Storm
   followers), add followers that can already hit the leader and an evolve if
   EP is available; if the estimate reaches the enemy leader's defense, verify
   the whole line on a clone (``Game.clone`` with a fixed RNG seed) and, if it
   really kills, commit to it.
2. Play cards: choose the subset of playable cards (bitmask, at most
   ``max_subset_size`` cards) that leaves the least PP unspent; ties go to the
   subset adding the most board value. Cards are played one per ``act`` call,
   re-checked against ``legal_actions`` every time.
3. Evolve: only to enable lethal (handled in 1), to break through Ward, or to
   flip a losing trade into a winning one.
4. Attack: prefer trades that kill an enemy follower without losing the
   attacker (Ward and high-value threats first); otherwise go face; never
   attack into a follower we cannot kill unless it is the last Ward blocking
   lethal next turn is not modelled - we simply skip such attacks.
5. End turn.

Every heuristic constant lives in ``GreedyWeights``.
"""

import contextlib
import io
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from src.common.enums import CardType, EffectType, ProcessType, TargetType, Zone
from svai import actions as A
from svai.agents.base import Agent
from svai.interfaces import Action, Decider

THREAT_KEYWORDS = (EffectType.WARD, EffectType.STORM, EffectType.BANE, EffectType.DRAIN,
                   EffectType.AURA, EffectType.BARRIER, EffectType.AMBUSH, EffectType.INTIMIDATE)


@dataclass(frozen=True)
class GreedyWeights:
    """All heuristic constants of GreedyAgent in one place."""
    max_subset_size: int = 4          # cap on cards per PP-subset (2^n enumeration guard)
    max_lethal_subsets: int = 200     # candidate subsets considered (sorted by damage)
    max_clones_per_decision: int = 4  # clone verifications per act() call (each ~15 ms)
    evolve_attack_bonus: int = 2      # +2/+2 on evolve (engine rule, used for estimates)
    follower_value_per_stat: float = 1.0
    spell_value_per_cost: float = 1.5
    amulet_value_per_cost: float = 1.0
    keyword_value: float = 1.5        # per threat keyword on a follower
    ward_priority: float = 6.0        # extra threat score for Ward
    growth_priority: float = 3.0      # for followers with turn-end / evolve effects
    kill_bonus: float = 4.0           # value of removing a follower
    survive_bonus: float = 2.0        # value of the attacker surviving a trade
    face_value_per_damage: float = 0.8
    leader_low_hp: int = 10           # below this we weigh face damage higher
    face_value_low_hp_multiplier: float = 1.6
    trade_min_gain: float = 0.5       # a trade must beat going face by this much
    evolve_trade_gain_required: float = 3.0
    keep_hand_cost_threshold: int = 3  # mulligan: keep cards that cost this or less
    discard_prefer_high_cost: bool = True


class _SimDecider(Decider):
    """Decider used inside look-ahead clones: deterministic, never prompts."""

    def __init__(self, agent: "GreedyAgent"):
        self.agent = agent

    def choose_action(self, game, player_id, legal_actions):
        return {"type": A.END_TURN}

    def choose_option(self, prompt, choices):
        return self.agent.choose_option(prompt, choices)

    def choose_mulligan(self, player_id, hand):
        return []

    def choose_discard(self, player_id, hand, count):
        return self.agent.choose_discard(player_id, hand, count)

    def choose_fuse(self, player_id, base_card, candidates):
        return []


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, rng=None, weights: Optional[GreedyWeights] = None):
        self.rng = rng  # unused: the agent is deterministic; kept for the registry signature
        self.w = weights or GreedyWeights()
        self.plan: List[Action] = []
        self.plan_turn_key: Optional[Tuple[str, int]] = None
        self.stats = {"lethal_found": 0, "lethal_verified": 0, "clones": 0}

    # ------------------------------------------------------------------ act
    def act(self, game: Any, player_id: str, legal_actions: List[Action]) -> Action:
        gsm = game.game_state_manager
        key = (player_id, gsm.turn_number)
        if self.plan_turn_key != key:
            self.plan, self.plan_turn_key = [], key

        # 0. follow a committed plan while its next step is still legal
        while self.plan:
            nxt = self.plan.pop(0)
            if nxt in legal_actions:
                return nxt
        # (a planned step became illegal: fall through and re-decide)

        # 1. lethal
        line = self._find_lethal(game, player_id, legal_actions)
        if line:
            self.plan = line[1:]
            return line[0]

        # 2. cards
        play = self._choose_play(game, player_id, legal_actions)
        if play is not None:
            return play

        # 3. evolve (ward break / trade flip)
        evo = self._choose_evolve(game, player_id, legal_actions)
        if evo is not None:
            return evo

        # 4. attacks
        atk = self._choose_attack(game, player_id, legal_actions)
        if atk is not None:
            return atk

        # 5. end turn
        return {"type": A.END_TURN}

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _by_type(legal_actions: List[Action], kind: str) -> List[Action]:
        return [a for a in legal_actions if a["type"] == kind]

    def _card(self, game, card_id):
        return game.game_state_manager.get_entity_by_id(card_id)

    def _followers(self, game, player_id) -> List[Any]:
        return [c for c in game.game_state_manager.players[player_id].field.get_cards()
                if c.get_type() == CardType.FOLLOWER]

    @staticmethod
    def _direct_damage(card) -> int:
        """Damage a card deals to the enemy leader when played (parsed effects only)."""
        total = 0
        for effect in card.effects:
            if effect.type not in (EffectType.FANFARE, EffectType.SPELL):
                continue
            for proc in getattr(effect, "processes", []):
                if getattr(proc, "process", None) != ProcessType.DEAL_DAMAGE:
                    continue
                tgt = getattr(proc, "target", None) or getattr(effect, "target", None)
                val = getattr(proc, "value", None)
                if isinstance(val, int) and tgt in (TargetType.OPPONENT_LEADER, TargetType.ALL_OPPONENTS):
                    total += val
        return total

    def _card_value(self, card) -> float:
        w = self.w
        t = card.get_type()
        if t == CardType.FOLLOWER:
            v = (card.current_attack + card.current_defense) * w.follower_value_per_stat
            v += sum(1 for k in THREAT_KEYWORDS if card.has_keyword(k)) * w.keyword_value
            return v
        if t == CardType.SPELL:
            return card.current_cost * w.spell_value_per_cost
        return card.current_cost * w.amulet_value_per_cost

    def _threat(self, card) -> float:
        w = self.w
        v = (card.current_attack + card.current_defense) * w.follower_value_per_stat
        v += sum(1 for k in THREAT_KEYWORDS if card.has_keyword(k)) * w.keyword_value
        if card.has_keyword(EffectType.WARD):
            v += w.ward_priority
        if any(e.type in (EffectType.ON_MY_TURN_END, EffectType.ON_MY_TURN_START, EffectType.ON_EVOLVE,
                          EffectType.EVOLVED, EffectType.COUNTDOWN) for e in card.effects):
            v += w.growth_priority
        return v

    # --------------------------------------------------------------- lethal
    def _find_lethal(self, game, player_id, legal_actions) -> Optional[List[Action]]:
        gsm = game.game_state_manager
        opp_id = game.opponent_id[player_id]
        opp = gsm.players[opp_id]
        if any(c.has_keyword(EffectType.WARD) for c in self._followers(game, opp_id)):
            return None  # wards block face damage; not modelled in M4 lethal search
        target_hp = opp.current_defense
        me = gsm.players[player_id]
        pp = me.current_pp + (1 if me.extra_pp > 0 else 0)

        # damage already on board
        attackers = [a for a in self._by_type(legal_actions, A.ATTACK) if a["target_id"] == opp_id]
        board = sum(self._card(game, a["attacker_id"]).current_attack for a in attackers)
        evolves = self._by_type(legal_actions, A.EVOLVE) + self._by_type(legal_actions, A.SUPER_EVOLVE)
        evo_bonus = self.w.evolve_attack_bonus if evolves and attackers else 0
        # (evolving a follower that can already attack keeps its attack permission)

        plays = self._by_type(legal_actions, A.PLAY_CARD)
        cands = []
        for p in plays:
            c = self._card(game, p["card_id"])
            if c is None:
                continue
            dmg = self._direct_damage(c)
            if c.get_type() == CardType.FOLLOWER and c.has_keyword(EffectType.STORM):
                dmg += c.current_attack
            if dmg > 0:
                cands.append((p, c.current_cost, dmg))
        # subsets by bitmask, cost-feasible, best damage first
        options = []
        n = min(len(cands), 10)
        for mask in range(1 << n):
            cost = dmg = 0
            chosen = []
            for i in range(n):
                if mask >> i & 1:
                    p, c_cost, c_dmg = cands[i]
                    cost += c_cost
                    dmg += c_dmg
                    chosen.append(p)
            if cost <= pp:
                options.append((dmg, cost, chosen))
        options.sort(key=lambda o: (-o[0], o[1]))
        clones = 0
        for dmg, cost, chosen in options[: self.w.max_lethal_subsets]:
            estimate = board + dmg + evo_bonus
            if estimate < target_hp:
                break  # sorted by damage: nothing below can reach either
            if clones >= self.w.max_clones_per_decision:
                break  # time budget: give up verifying this turn
            self.stats["lethal_found"] += 1
            clones += 1
            line = self._lethal_line(game, player_id, chosen, evolves[:1] if evo_bonus else [], attackers)
            if line is not None:
                self.stats["lethal_verified"] += 1
                return line
        return None

    def _lethal_line(self, game, player_id, plays, evolves, attackers) -> Optional[List[Action]]:
        """Simulate plays -> evolve -> face attacks on a clone; return the line if it kills."""
        with contextlib.redirect_stdout(io.StringIO()):  # the engine logs every step
            return self._lethal_line_quiet(game, player_id, plays, evolves, attackers)

    def _lethal_line_quiet(self, game, player_id, plays, evolves, attackers) -> Optional[List[Action]]:
        sim = game.clone(deciders={pid: _SimDecider(self) for pid in game.game_state_manager.players},
                         rng_seed=12345)
        self.stats["clones"] += 1
        opp_id = game.opponent_id[player_id]
        line: List[Action] = []
        try:
            for action in plays + evolves:
                legal = A.legal_actions(sim, player_id)
                if action not in legal:
                    return None
                A.apply_action(sim, player_id, action)
                line.append(action)
                if sim.is_game_over():
                    return line
            # attack the leader with everything that can
            while True:
                legal = A.legal_actions(sim, player_id)
                face = [a for a in legal if a["type"] == A.ATTACK and a["target_id"] == opp_id]
                if not face:
                    break
                face.sort(key=lambda a: -self._card(sim, a["attacker_id"]).current_attack)
                A.apply_action(sim, player_id, face[0])
                line.append(face[0])
                if sim.is_game_over():
                    break
        except Exception:  # noqa: BLE001 - a simulation error just means "no verified lethal"
            return None
        return line if sim.winner() == player_id else None

    # ---------------------------------------------------------------- plays
    def _choose_play(self, game, player_id, legal_actions) -> Optional[Action]:
        plays = self._by_type(legal_actions, A.PLAY_CARD)
        if not plays:
            return None
        gsm = game.game_state_manager
        me = gsm.players[player_id]
        field_room = A.FIELD_MAX - me.field.size()
        # one entry per card: prefer the no-extra-PP variant unless extra PP is needed
        by_card: Dict[str, Action] = {}
        for p in plays:
            cur = by_card.get(p["card_id"])
            if cur is None or (cur["use_extra_pp"] and not p["use_extra_pp"]):
                by_card[p["card_id"]] = p
        items = []
        for cid, p in by_card.items():
            c = self._card(game, cid)
            if c is None:
                continue
            cost = max(p["enhanced_cost"], c.current_cost) if p["enhanced_cost"] else c.current_cost
            items.append((p, c, cost, self._card_value(c)))
        pp = me.current_pp
        best = None  # (leftover, -value, subset)
        n = len(items)
        for k in range(1, min(n, self.w.max_subset_size) + 1):
            for combo in combinations(range(n), k):
                cost = sum(items[i][2] for i in combo)
                uses_extra = sum(1 for i in combo if items[i][0]["use_extra_pp"])
                budget = pp + (1 if (uses_extra and me.extra_pp > 0) else 0)
                if cost > budget:
                    continue
                needs_room = sum(1 for i in combo if items[i][1].get_type() in (CardType.FOLLOWER, CardType.AMULET))
                if needs_room > field_room:
                    continue
                value = sum(items[i][3] for i in combo)
                cand = (budget - cost, -value, combo)
                if best is None or cand < best:
                    best = cand
        if best is None:
            return None
        combo = best[2]
        # play the most expensive card first (cheaper ones are more likely to stay legal)
        first = max(combo, key=lambda i: items[i][2])
        return items[first][0]

    # -------------------------------------------------------------- attacks
    def _attack_options(self, game, player_id, legal_actions):
        opp_id = game.opponent_id[player_id]
        out = []
        for a in self._by_type(legal_actions, A.ATTACK):
            attacker = self._card(game, a["attacker_id"])
            if attacker is None:
                continue
            if a["target_id"] == opp_id:
                out.append((a, attacker, None))
            else:
                target = self._card(game, a["target_id"])
                if target is not None:
                    out.append((a, attacker, target))
        return out

    def _trade_score(self, attacker, target, opp_leader_hp) -> float:
        """Score of attacking ``target`` (None = leader) with ``attacker``."""
        w = self.w
        if target is None:
            mult = w.face_value_low_hp_multiplier if opp_leader_hp <= w.leader_low_hp else 1.0
            return attacker.current_attack * w.face_value_per_damage * mult
        kills = attacker.current_attack >= target.current_defense or attacker.has_keyword(EffectType.BANE)
        if target.has_keyword(EffectType.BARRIER):
            kills = False
        survives = (target.current_attack < attacker.current_defense or attacker.has_keyword(EffectType.BARRIER)
                    or attacker.is_super_evolved) and not target.has_keyword(EffectType.BANE)
        if not kills:
            return -w.kill_bonus  # bouncing off is bad
        score = w.kill_bonus + self._threat(target)
        score += w.survive_bonus if survives else -self._card_value(attacker)
        return score

    def _choose_attack(self, game, player_id, legal_actions) -> Optional[Action]:
        options = self._attack_options(game, player_id, legal_actions)
        if not options:
            return None
        opp_hp = game.game_state_manager.players[game.opponent_id[player_id]].current_defense
        scored = []
        for a, attacker, target in options:
            scored.append((self._trade_score(attacker, target, opp_hp), target is None, a))
        # best trade must beat the best face option by trade_min_gain
        faces = [s for s in scored if s[1]]
        trades = [s for s in scored if not s[1]]
        best_face = max(faces, key=lambda s: s[0]) if faces else None
        best_trade = max(trades, key=lambda s: s[0]) if trades else None
        if best_trade and (best_face is None or best_trade[0] >= best_face[0] + self.w.trade_min_gain):
            if best_trade[0] > 0:
                return best_trade[2]
        if best_face:
            return best_face[2]
        if best_trade and best_trade[0] > 0:
            return best_trade[2]
        return None  # only losing attacks are available: keep the followers

    # --------------------------------------------------------------- evolve
    def _choose_evolve(self, game, player_id, legal_actions) -> Optional[Action]:
        evolves = self._by_type(legal_actions, A.EVOLVE) + self._by_type(legal_actions, A.SUPER_EVOLVE)
        if not evolves:
            return None
        opp_id = game.opponent_id[player_id]
        enemies = self._followers(game, opp_id)
        wards = [e for e in enemies if e.has_keyword(EffectType.WARD)]
        opp_hp = game.game_state_manager.players[opp_id].current_defense
        bonus = self.w.evolve_attack_bonus
        best = None
        for evo in evolves:
            c = self._card(game, evo["card_id"])
            if c is None:
                continue
            atk_after, def_after = c.current_attack + bonus, c.current_defense + bonus
            # after evolving, the follower may attack followers this turn (engine rule)
            for target in enemies:
                before = self._trade_score(c, target, opp_hp) if c.can_attack(CardType.FOLLOWER) else -99.0
                kills = atk_after >= target.current_defense or c.has_keyword(EffectType.BANE)
                survives = target.current_attack < def_after and not target.has_keyword(EffectType.BANE)
                after = (self.w.kill_bonus + self._threat(target) + (self.w.survive_bonus if survives else 0)) if kills else -self.w.kill_bonus
                gain = after - max(before, 0.0)
                if target in wards:
                    gain += self.w.ward_priority
                if gain >= self.w.evolve_trade_gain_required and (best is None or gain > best[0]):
                    best = (gain, evo)
        return best[1] if best else None

    # ------------------------------------------------------------ non-action
    def choose_option(self, prompt: str, choices: Dict[str, Any]) -> Any:
        if not choices:
            return None
        # Card-target prompts list "name (ID: n)" labels; prefer the highest stats we can see
        # is impossible from labels alone, so take the first option deterministically.
        return next(iter(choices.values()))

    def choose_mulligan(self, player_id: str, hand: List[Any]) -> List[str]:
        return [c.card_id for c in hand if c.current_cost > self.w.keep_hand_cost_threshold]

    def choose_discard(self, player_id: str, hand: List[Any], count: int) -> List[str]:
        ordered = sorted(hand, key=lambda c: -c.current_cost if self.w.discard_prefer_high_cost else c.current_cost)
        return [c.card_id for c in ordered[:count]]
