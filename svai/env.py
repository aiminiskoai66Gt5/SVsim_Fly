"""Gymnasium environment wrapping the SVsim engine.

Two ways to use it:

* ``SVEnv(opponent=RandomAgent())`` - the env controls ``agent_player`` and the
  opponent plays its whole turn inside ``step`` whenever the agent ends its turn.
* ``SVEnv(opponent=None)`` - self-play: ``step`` applies the action for whoever
  is to act; ``info["current_player"]`` says who that is next.

Everything random derives from ``reset(seed)``: engine RNG, deck generation,
the opponent's RNG and the env's own hooks. Illegal actions raise
``IllegalActionError``; engine exceptions propagate untouched.

Effect choices ("choose one", targets) made by the controlled player are NOT
part of the action space in M1; they go through ``option_agent`` (default:
first option). See OBSERVATION.md for the observation layout.
"""

import contextlib
import copy
import io
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover
    gym = None
    spaces = None

from src.common import card_data
from src.common.enums import CardType, ClassType, EffectType, Zone
from src.engine.main_game_logic import Game
from svai import actions as A
from svai.agents.base import Agent
from svai.agents.random_agent import RandomAgent
from svai.config import load_config, repo_path
from svai.deciders import AgentDecider
from svai.interfaces import Action, Decider, NullView

PLAYER_IDS = ("player1", "player2")

# --- observation layout (keep in sync with OBSERVATION.md) --------------------
KEYWORDS = [EffectType.WARD, EffectType.STORM, EffectType.RUSH, EffectType.BANE, EffectType.DRAIN,
            EffectType.BARRIER, EffectType.AMBUSH, EffectType.INTIMIDATE, EffectType.AURA, EffectType.DISABLE]
LEADER_FEATURES = 11
HAND_SLOT_FEATURES = 9
FIELD_SLOT_FEATURES = 11 + len(KEYWORDS)  # 21
GLOBAL_FEATURES = 2
OBS_SIZE = (2 * LEADER_FEATURES + A.HAND_MAX * HAND_SLOT_FEATURES
            + 2 * A.FIELD_MAX * FIELD_SLOT_FEATURES + GLOBAL_FEATURES)  # 315

_db_loaded = False


def ensure_card_database(path: str) -> None:
    """Load the parsed card database once per process."""
    global _db_loaded
    if not _db_loaded:
        with contextlib.redirect_stdout(io.StringIO()):
            card_data.load_card_databases(str(repo_path(path)))
        _db_loaded = True


def all_cards() -> Dict[str, Any]:
    return {**card_data.BASIC_CARD_DATABASE, **card_data.LEGENDS_RISE_CARD_DATABASE}


def load_deck_file(path: str) -> List[Any]:
    """Deck JSON ({"cards": [{"card_id", "count"}]}) -> list of CardData."""
    import json
    with open(repo_path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    deck = []
    for item in data.get("cards", []):
        c = card_data.get_card_data_by_id(item.get("card_id"))
        if c:
            deck.extend([c] * item.get("count", 1))
    return deck


class _AgentHooksDecider(Decider):
    """Decider for the env-controlled player: actions come from step(), the
    rest (effect options, mulligan, discard, fuse) from ``option_agent``."""

    def __init__(self, option_agent: Agent):
        self.option_agent = option_agent

    def choose_action(self, game, player_id, legal_actions):
        raise RuntimeError("the env-controlled player's actions come from SVEnv.step()")

    def choose_option(self, prompt, choices):
        return self.option_agent.choose_option(prompt, choices)

    def choose_mulligan(self, player_id, hand):
        return self.option_agent.choose_mulligan(player_id, hand)

    def choose_discard(self, player_id, hand, count):
        return self.option_agent.choose_discard(player_id, hand, count)

    def choose_fuse(self, player_id, base_card, candidates):
        return self.option_agent.choose_fuse(player_id, base_card, candidates)


class SVEnv(gym.Env if gym else object):
    """See module docstring."""

    metadata = {"render_modes": []}

    def __init__(self, opponent: Optional[Agent] = "random", agent_player: str = "player1",
                 option_agent: Any = None, config: Optional[Dict[str, Any]] = None,
                 quiet: bool = True, deck_mode: Optional[str] = None, deck_files: Optional[List[str]] = None,
                 max_turns: Optional[int] = None):
        self.cfg = config or load_config()
        env_cfg = self.cfg["env"]
        ensure_card_database(env_cfg["card_database"])
        self.agent_player = agent_player
        self.opponent_player = PLAYER_IDS[1] if agent_player == PLAYER_IDS[0] else PLAYER_IDS[0]
        self._opponent_spec = opponent          # "random", an Agent, or None (self-play)
        # An Agent, or {player_id: Agent} to give each side its own hooks (mulligan, effect options...).
        self.option_agent = option_agent or Agent()
        self.quiet = quiet
        self.deck_mode = deck_mode or env_cfg["deck_mode"]
        self.deck_files = deck_files if deck_files is not None else list(env_cfg["deck_files"])
        self.max_turns = max_turns or env_cfg["max_turns"]
        self.opponent_action_cap = env_cfg["opponent_actions_per_turn"]

        if spaces is not None:
            self.action_space = spaces.Discrete(A.ACTION_SPACE_SIZE)
            self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game: Optional[Game] = None
        self.rng: Optional[random.Random] = None
        self.opponent: Optional[Agent] = None
        self.current_player: str = agent_player
        self.seed_used: Optional[int] = None
        self.action_log: List[Tuple[str, int]] = []
        self._legal_cache: Optional[List[Action]] = None

    # --- helpers ---------------------------------------------------------
    def _silence(self):
        return contextlib.redirect_stdout(io.StringIO()) if self.quiet else contextlib.nullcontext()

    def _make_decks(self) -> Tuple[List[Any], List[Any]]:
        if self.deck_mode == "files":
            if len(self.deck_files) < 1:
                raise ValueError("deck_mode='files' needs at least one entry in deck_files")
            d1 = load_deck_file(self.deck_files[0])
            d2 = load_deck_file(self.deck_files[1 % len(self.deck_files)])
            return d1, d2
        from deck_builder import generate_random_deck
        classes = [c for c in ClassType if c != ClassType.NEUTRAL]
        cards = all_cards()
        d1 = generate_random_deck(self.rng.choice(classes), cards, rng=self.rng)
        d2 = generate_random_deck(self.rng.choice(classes), cards, rng=self.rng)
        return d1, d2

    def _build_opponent(self) -> Optional[Agent]:
        if self._opponent_spec is None:
            return None
        if self._opponent_spec == "random":
            return RandomAgent(random.Random(self.rng.randrange(2**31)))
        agent = self._opponent_spec
        if hasattr(agent, "reseed"):
            agent.reseed(self.rng.randrange(2**31))
        return agent

    def _obs_player(self) -> str:
        return self.current_player if self.opponent is None else self.agent_player

    # --- gymnasium API ---------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if gym:
            super().reset(seed=seed)
        self.seed_used = seed
        self.rng = random.Random(seed)
        self.action_log = []
        self.opponent = self._build_opponent()
        d1, d2 = self._make_decks()
        def hooks_for(pid: str) -> Decider:
            agent = self.option_agent[pid] if isinstance(self.option_agent, dict) else self.option_agent
            return _AgentHooksDecider(agent)

        deciders: Dict[str, Decider] = {self.agent_player: hooks_for(self.agent_player)}
        if self.opponent is not None:
            deciders[self.opponent_player] = AgentDecider(self.opponent)
        else:
            deciders[self.opponent_player] = hooks_for(self.opponent_player)
        with self._silence():
            self.game = Game(PLAYER_IDS[0], PLAYER_IDS[1], d1, d2, view=NullView(), decider=deciders, rng=self.rng)
        self.current_player = PLAYER_IDS[0]
        self._legal_cache = None
        with self._silence():
            if self.opponent is not None and self.current_player == self.opponent_player:
                self._play_opponent_turn()
        return self._observation(), self._info()

    def step(self, action_index: int):
        if self.game is None:
            raise RuntimeError("call reset() first")
        if self.game.is_game_over():
            raise RuntimeError("game is over; call reset()")
        player = self.current_player
        with self._silence():
            action = A.decode(self.game, player, int(action_index), self.legal_actions())
            self.action_log.append((player, int(action_index)))
            A.apply_action(self.game, player, action)
            self._legal_cache = None
            if action["type"] == A.END_TURN and not self.game.is_game_over():
                self.current_player = self.game.opponent_id[player]
                if self.opponent is not None and self.current_player == self.opponent_player:
                    self._play_opponent_turn()
        return self._after_step()

    def _play_opponent_turn(self) -> None:
        """Let the scripted opponent play until it ends its turn (or the game ends)."""
        opp = self.opponent_player
        for _ in range(self.opponent_action_cap):
            if self.game.is_game_over():
                return
            legal = A.legal_actions(self.game, opp)
            action = self.game.decider_for(opp).choose_action(self.game, opp, legal)
            A.apply_action(self.game, opp, action)
            if action["type"] == A.END_TURN:
                break
        else:
            # Safety cap reached (same rule as fuzz_runner): force the turn to end.
            if not self.game.is_game_over():
                A.apply_action(self.game, opp, {"type": A.END_TURN})
        self.current_player = self.game.opponent_id[opp]
        self._legal_cache = None

    def _after_step(self):
        winner = self.game.winner()
        terminated = winner is not None
        truncated = (not terminated) and self.game.game_state_manager.turn_number > self.max_turns
        reward = 0.0
        if terminated and winner != Game.DRAW:
            reward = 1.0 if winner == self._obs_player() else -1.0
        return self._observation(), reward, terminated, truncated, self._info()

    # --- native helpers --------------------------------------------------
    def legal_actions(self) -> List[Action]:
        if self._legal_cache is None:
            with self._silence():
                self._legal_cache = A.legal_actions(self.game, self.current_player)
        return self._legal_cache

    def legal_action_mask(self) -> np.ndarray:
        with self._silence():
            mask = A.legal_action_mask(self.game, self.current_player, self.legal_actions())
        return np.asarray(mask, dtype=bool)

    def winner(self) -> Optional[str]:
        return self.game.winner() if self.game else None

    def clone(self) -> "SVEnv":
        """Deep copy including engine state, RNG state and the opponent."""
        return copy.deepcopy(self)

    def _info(self) -> Dict[str, Any]:
        return {
            "legal_mask": self.legal_action_mask(),
            "legal_actions": list(self.legal_actions()),
            "current_player": self.current_player,
            "turn": self.game.game_state_manager.turn_number,
            "winner": self.game.winner(),
            "seed": self.seed_used,
        }

    # --- observation -----------------------------------------------------
    def _observation(self) -> np.ndarray:
        me = self._obs_player()
        with self._silence():
            return encode_observation(self.game, me)


def _leader_features(game: Game, pid: str) -> List[float]:
    p = game.game_state_manager.players[pid]
    return [p.current_defense / 20.0, p.max_defense / 20.0, p.current_pp / 10.0, p.max_pp / 10.0,
            p.current_ep / 2.0, p.current_sep / 2.0, float(p.extra_pp > 0),
            p.hand.size() / A.HAND_MAX, p.deck.size() / 40.0, p.graveyard.size() / 40.0,
            p.field.size() / A.FIELD_MAX]


def _type_onehot(card) -> List[float]:
    t = card.get_type()
    return [float(t == CardType.FOLLOWER), float(t == CardType.SPELL), float(t == CardType.AMULET)]


def _hand_features(game: Game, pid: str) -> List[float]:
    gsm = game.game_state_manager
    hand = gsm.players[pid].hand.get_cards()
    _, playable = game.get_playable_cards_id(pid, False)
    if game.has_extra_pp(pid):
        _, playable_extra = game.get_playable_cards_id(pid, True)
    else:
        playable_extra = playable
    out: List[float] = []
    for i in range(A.HAND_MAX):
        if i < len(hand):
            c = hand[i]
            out += [1.0, c.current_cost / 10.0] + _type_onehot(c) + [
                c.current_attack / 10.0, c.current_defense / 10.0, float(playable[i]), float(playable_extra[i])]
        else:
            out += [0.0] * HAND_SLOT_FEATURES
    return out


def _field_features(game: Game, pid: str) -> List[float]:
    field = game.game_state_manager.players[pid].field.get_cards()
    out: List[float] = []
    for i in range(A.FIELD_MAX):
        if i < len(field):
            c = field[i]
            is_follower = c.get_type() == CardType.FOLLOWER
            out += [1.0, float(is_follower), float(c.get_type() == CardType.AMULET),
                    c.current_attack / 10.0, c.current_defense / 10.0, c.max_defense / 10.0, c.current_cost / 10.0,
                    float(is_follower and c.can_attack(CardType.LEADER)),
                    float(is_follower and c.can_attack(CardType.FOLLOWER)),
                    float(c.is_evolved), float(c.is_super_evolved)]
            out += [float(c.has_keyword(k)) for k in KEYWORDS]
        else:
            out += [0.0] * FIELD_SLOT_FEATURES
    return out


def encode_observation(game: Game, me: str) -> np.ndarray:
    """Fixed-length float32 vector from ``me``'s point of view (see OBSERVATION.md)."""
    opp = game.opponent_id[me]
    gsm = game.game_state_manager
    vec = (_leader_features(game, me) + _leader_features(game, opp)
           + _hand_features(game, me)
           + _field_features(game, me) + _field_features(game, opp)
           + [gsm.turn_number / 40.0, float(gsm.current_turn_player_id == me)])
    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape == (OBS_SIZE,), arr.shape
    return arr
