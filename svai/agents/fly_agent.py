"""FlyAgent: plays with a trained (or randomly initialised) FlyPolicy."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from svai import actions as A
from svai.agents.base import Agent
from svai.config import load_config
from svai.env import encode_observation
from svai.interfaces import Action


class FlyAgent(Agent):
    """Wraps FlyPolicy for the arena / GUI. Deterministic (argmax) by default.

    The recurrent state is reset whenever a new game is detected (a different
    Game object) and carried across the agent's own decisions otherwise.
    """

    name = "fly"

    def __init__(self, rng=None, checkpoint: Optional[str] = None, policy=None, graph=None, sample: bool = False,
                 cfg: Optional[Dict[str, Any]] = None):
        self.rng = rng
        self.sample = sample
        self.cfg = cfg or load_config()
        if policy is None:
            from svai.fly.train import load_or_build_graph, make_policy
            graph = graph or load_or_build_graph(self.cfg, verbose=False)
            policy = make_policy(graph, self.cfg)
            path = Path(checkpoint) if checkpoint else Path(self.cfg["fly"]["checkpoint"])
            self.checkpoint_loaded = False
            if path.exists():
                state = torch.load(path, map_location="cpu", weights_only=False)
                policy.load_state_dict(state["state_dict"])
                self.checkpoint_loaded = True
        else:
            self.checkpoint_loaded = True
        self.policy = policy.eval()
        self.graph = graph
        self._game_id = None
        self._h = None
        self.gen = torch.Generator().manual_seed(rng.randrange(2**31) if rng is not None else 0)

    def act(self, game: Any, player_id: str, legal_actions: List[Action]) -> Action:
        if self._game_id != id(game) or self._h is None:
            self._game_id = id(game)
            self._h = self.policy.initial_state()
        with contextlib.redirect_stdout(io.StringIO()):  # the engine logs inside can_attack()
            obs = encode_observation(game, player_id)
            table = A.index_table(game, player_id, legal_actions)
        mask = [i in table for i in range(A.ACTION_SPACE_SIZE)]
        with torch.no_grad():
            idx, _, _, self._h = self.policy.act(obs, self._h, mask, sample=self.sample, generator=self.gen)
        return table[idx]
