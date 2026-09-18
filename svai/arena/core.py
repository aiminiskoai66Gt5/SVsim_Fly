"""One arena game, its record, and the aggregate report.

A game is fully described by a ``GameSpec`` (agent names, seed, limits). The
seed derives everything: engine RNG and deck generation (via SVEnv), and one
``random.Random`` per agent. Running the same spec twice gives the same
``GameRecord`` (tests/test_m2_arena.py).
"""

import dataclasses
import math
import statistics
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from svai import actions as A
from svai.agents import make_agent
from svai.env import SVEnv


@dataclass
class GameSpec:
    agent_a: str
    agent_b: str
    seed: int
    a_is_player1: bool = True
    max_turns: int = 40
    actions_per_turn_cap: int = 100
    deck_mode: str = "random"
    deck_files: List[str] = field(default_factory=list)
    exclude_unparsed_cards: bool = False


@dataclass
class GameRecord:
    seed: int
    agent_a: str
    agent_b: str
    a_is_player1: bool
    outcome: str = "unknown"       # "a" | "b" | "draw" | "truncated" | "exception" | "timeout"
    winner_player: Optional[str] = None
    turns: int = 0
    actions: int = 0
    branching_sum: int = 0         # sum of legal-action counts over all decisions
    decisions: int = 0
    decision_time_a: List[float] = field(default_factory=list)
    decision_time_b: List[float] = field(default_factory=list)
    turn_caps_hit: int = 0
    unimplemented_effects: int = 0   # card effects the engine reported as unimplemented (data gaps)
    exception_type: Optional[str] = None
    exception_site: Optional[str] = None
    exception_message: Optional[str] = None
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _site(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return "?"
    fr = tb[-1]
    import os
    return f"{os.path.basename(fr.filename)}:{fr.lineno}"


def play_game(spec: GameSpec) -> GameRecord:
    """Play one game in-process. Engine exceptions are recorded, never swallowed silently."""
    import random
    rec = GameRecord(seed=spec.seed, agent_a=spec.agent_a, agent_b=spec.agent_b, a_is_player1=spec.a_is_player1)
    t0 = time.perf_counter()
    base = random.Random(spec.seed)
    rng_a = random.Random(base.randrange(2**31))
    rng_b = random.Random(base.randrange(2**31))
    agents = {"a": make_agent(spec.agent_a, rng_a), "b": make_agent(spec.agent_b, rng_b)}
    side_of = {"player1": "a" if spec.a_is_player1 else "b", "player2": "b" if spec.a_is_player1 else "a"}

    # Self-play env: each player's non-action decisions (mulligan, effect options,
    # discard, fuse) go to that player's own agent.
    env = SVEnv(opponent=None, quiet=True, deck_mode=spec.deck_mode, deck_files=spec.deck_files,
                max_turns=spec.max_turns, exclude_unparsed_cards=spec.exclude_unparsed_cards,
                option_agent={pid: agents[side] for pid, side in side_of.items()})
    try:
        obs, info = env.reset(seed=spec.seed)
        actions_this_turn = 0
        current = info["current_player"]
        done = False
        while not done:
            player = info["current_player"]
            if player != current:
                current = player
                actions_this_turn = 0
            side = side_of[player]
            legal = info["legal_actions"]
            rec.branching_sum += len(legal)
            rec.decisions += 1
            if actions_this_turn >= spec.actions_per_turn_cap:
                rec.turn_caps_hit += 1
                action = {"type": A.END_TURN}
            else:
                t1 = time.perf_counter()
                action = agents[side].act(env.game, player, legal)
                (rec.decision_time_a if side == "a" else rec.decision_time_b).append(time.perf_counter() - t1)
            index = A.encode(env.game, player, action)
            obs, reward, term, trunc, info = env.step(index)
            rec.actions += 1
            actions_this_turn += 1
            done = term or trunc
        rec.turns = info["turn"]
        rec.unimplemented_effects = len(env.game.game_state_manager.unimplemented_hits)
        winner = info["winner"]
        rec.winner_player = winner
        if trunc and not term:
            rec.outcome = "truncated"
        elif winner == "draw":
            rec.outcome = "draw"
        else:
            rec.outcome = side_of[winner]
    except RecursionError as exc:
        rec.outcome = "exception"
        rec.exception_type = "RecursionError"
        rec.exception_site = _site(exc)
        rec.exception_message = "maximum recursion depth exceeded"
        rec.turns = env.game.game_state_manager.turn_number if env.game else 0
    except Exception as exc:  # noqa: BLE001 - recorded, reported, counted
        rec.outcome = "exception"
        rec.exception_type = type(exc).__name__
        rec.exception_site = _site(exc)
        rec.exception_message = str(exc)[:200]
        rec.turns = env.game.game_state_manager.turn_number if env.game else 0
    if env.game is not None:
        rec.unimplemented_effects = len(env.game.game_state_manager.unimplemented_hits)
    rec.wall_time = time.perf_counter() - t0
    return rec


# --- aggregate ---------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ms_stats(values: List[float]) -> str:
    if not values:
        return "n/a"
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    return f"mean {1000 * statistics.mean(values):.3f} / p95 {1000 * p95:.3f} / max {1000 * values[-1]:.1f} ms"


def summarize(records: List[GameRecord], agent_a: str, agent_b: str) -> Dict[str, Any]:
    n = len(records)
    by = lambda o: sum(1 for r in records if r.outcome == o)  # noqa: E731
    wins_a, wins_b, draws = by("a"), by("b"), by("draw")
    decided = wins_a + wins_b
    exceptions, timeouts, truncated = by("exception"), by("timeout"), by("truncated")
    p, lo, hi = wilson(wins_a, decided)
    p_all, lo_all, hi_all = wilson(wins_a, n)
    finished = [r for r in records if r.outcome in ("a", "b", "draw", "truncated")]
    sites: Dict[str, int] = {}
    for r in records:
        if r.outcome == "exception":
            key = f"{r.exception_type} @ {r.exception_site}: {(r.exception_message or '')[:60]}"
            sites[key] = sites.get(key, 0) + 1
    return {
        "games": n,
        "agent_a": agent_a, "agent_b": agent_b,
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "winrate_a_of_decided": (p, lo, hi), "decided": decided,
        "winrate_a_of_all": (p_all, lo_all, hi_all),
        "exceptions": exceptions, "timeouts": timeouts, "truncated": truncated,
        "exception_rate": exceptions / n if n else 0.0,
        "timeout_rate": timeouts / n if n else 0.0,
        "mean_turns": statistics.mean([r.turns for r in finished]) if finished else 0.0,
        "mean_actions": statistics.mean([r.actions for r in finished]) if finished else 0.0,
        "mean_branching": (sum(r.branching_sum for r in records) / max(1, sum(r.decisions for r in records))),
        "decision_time_a": [t for r in records for t in r.decision_time_a],
        "decision_time_b": [t for r in records for t in r.decision_time_b],
        "turn_caps_hit": sum(r.turn_caps_hit for r in records),
        "games_with_unimplemented": sum(1 for r in records if r.unimplemented_effects),
        "unimplemented_total": sum(r.unimplemented_effects for r in records),
        "exception_sites": dict(sorted(sites.items(), key=lambda kv: -kv[1])),
        "mean_wall_time": statistics.mean([r.wall_time for r in records]) if records else 0.0,
        "median_wall_time": statistics.median([r.wall_time for r in records]) if records else 0.0,
    }


def format_report(s: Dict[str, Any]) -> str:
    p, lo, hi = s["winrate_a_of_decided"]
    pa, loa, hia = s["winrate_a_of_all"]
    lines = [
        f"## Arena: {s['agent_a']} (A) vs {s['agent_b']} (B), {s['games']} games",
        "",
        "| metric | value |",
        "|---|---|",
        f"| A wins / B wins / draws | {s['wins_a']} / {s['wins_b']} / {s['draws']} |",
        f"| A win rate (of {s['decided']} decided games), Wilson 95% | {_pct(p)} [{_pct(lo)}, {_pct(hi)}] |",
        f"| A win rate (of all games) | {_pct(pa)} [{_pct(loa)}, {_pct(hia)}] |",
        f"| exceptions | {s['exceptions']} ({_pct(s['exception_rate'])}) |",
        f"| timeouts (hangs) | {s['timeouts']} ({_pct(s['timeout_rate'])}) |",
        f"| truncated at max_turns | {s['truncated']} |",
        f"| mean game length (finished games) | {s['mean_turns']:.1f} turns, {s['mean_actions']:.1f} actions |",
        f"| mean branching factor | {s['mean_branching']:.2f} |",
        f"| decision time A | {_ms_stats(s['decision_time_a'])} |",
        f"| decision time B | {_ms_stats(s['decision_time_b'])} |",
        f"| per-turn action cap hits | {s['turn_caps_hit']} |",
        f"| games with an unimplemented (unparsed) card effect | {s['games_with_unimplemented']} ({_pct(s['games_with_unimplemented'] / max(1, s['games']))}), {s['unimplemented_total']} hits |",
        f"| wall time per game (mean / median) | {1000 * s['mean_wall_time']:.0f} / {1000 * s['median_wall_time']:.0f} ms |",
    ]
    if s["exception_sites"]:
        lines += ["", "### Exceptions by site", "", "| count | exception |", "|---|---|"]
        for key, count in s["exception_sites"].items():
            lines.append(f"| {count} | {key} |")
    return "\n".join(lines)
