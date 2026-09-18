"""REINFORCE training of FlyPolicy against a scripted opponent, CPU only.

    python scripts/train_fly.py --games 2000 --opponent random --eval-every 200

Each game is one episode: the policy keeps its recurrent state across its own
decisions, the opponent's whole turn runs inside ``SVEnv.step``. Return is +1
win / -1 loss / 0 otherwise. Loss = -(G - b) * sum(log pi) - beta * entropy,
with a moving-average baseline b. Checkpoints go to ``fly.checkpoint_dir``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from svai import actions as A
from svai.agents import make_agent
from svai.config import ensure_dir, load_config
from svai.env import OBS_SIZE, SVEnv
from svai.fly.graph import FlyGraph, build_graph
from svai.fly.policy import FlyPolicy


def load_or_build_graph(cfg: Dict[str, Any], verbose: bool = True) -> FlyGraph:
    fly = cfg["fly"]
    cache = Path(fly["graph_cache"]) / f"{fly['subset']}_min{fly['min_synapses']}.npz"
    if cache.exists():
        g = FlyGraph.load(cache)
        if verbose:
            print(f"loaded {cache}: {g.describe()}")
        return g
    g = build_graph(Path(fly["release_dir"]), subset=fly["subset"], min_synapses=int(fly["min_synapses"]),
                    status_values=fly["status_values"], verbose=verbose)
    ensure_dir(cache.parent)
    g.save(cache)
    if verbose:
        print(f"saved {cache}")
    return g


def make_policy(graph: FlyGraph, cfg: Dict[str, Any], seed: int = 0) -> FlyPolicy:
    fly = cfg["fly"]
    return FlyPolicy(graph, OBS_SIZE, A.ACTION_SPACE_SIZE, n_input=int(fly["n_input"]), n_output=int(fly["n_output"]),
                     microsteps=int(fly["microsteps"]), spectral_radius_target=float(fly["spectral_radius"]),
                     learn_edge_gain=bool(fly["learn_edge_gain"]), seed=seed)


def play_episode(policy: FlyPolicy, env: SVEnv, seed: int, sample: bool = True,
                 generator: Optional[torch.Generator] = None):
    """Run one game; returns (return, list of log-probs, list of entropies, steps, exception or None)."""
    obs, info = env.reset(seed=seed)
    h = policy.initial_state()
    logps, ents = [], []
    ret, steps, exc = 0.0, 0, None
    done = False
    try:
        while not done:
            a, logp, ent, h = policy.act(obs, h, info["legal_mask"], sample=sample, generator=generator)
            obs, r, term, trunc, info = env.step(a)
            logps.append(logp)
            ents.append(ent)
            ret += r
            steps += 1
            done = term or trunc
    except Exception as e:  # noqa: BLE001 - counted and reported, never hidden
        exc = f"{type(e).__name__}: {str(e)[:120]}"
    return ret, logps, ents, steps, exc


def evaluate(policy: FlyPolicy, opponent: str, games: int, seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy (argmax) play, sides alternating; Wilson 95% interval on the win rate."""
    from svai.arena.core import wilson
    env_p1 = SVEnv(opponent=make_agent(opponent, random.Random(seed)), agent_player="player1",
                   exclude_unparsed_cards=True, config=cfg)
    env_p2 = SVEnv(opponent=make_agent(opponent, random.Random(seed + 1)), agent_player="player2",
                   exclude_unparsed_cards=True, config=cfg)
    wins = losses = excs = 0
    t0 = time.perf_counter()
    decision_times = []
    with torch.no_grad():
        for i in range(games):
            env = env_p1 if i % 2 == 0 else env_p2
            obs, info = env.reset(seed=seed + i)
            h = policy.initial_state()
            done = False
            try:
                while not done:
                    t1 = time.perf_counter()
                    a, _, _, h = policy.act(obs, h, info["legal_mask"], sample=False)
                    decision_times.append(time.perf_counter() - t1)
                    obs, r, term, trunc, info = env.step(a)
                    done = term or trunc
                if term and r > 0:
                    wins += 1
                elif term and r < 0:
                    losses += 1
            except Exception:  # noqa: BLE001
                excs += 1
    p, lo, hi = wilson(wins, games)
    return {"games": games, "wins": wins, "losses": losses, "exceptions": excs, "winrate": p, "ci95": (lo, hi),
            "decision_ms_mean": 1000 * float(np.mean(decision_times)) if decision_times else 0.0,
            "decision_ms_p95": 1000 * float(np.percentile(decision_times, 95)) if decision_times else 0.0,
            "decision_ms_max": 1000 * float(np.max(decision_times)) if decision_times else 0.0,
            "seconds": time.perf_counter() - t0}


def train(games: int = 1000, opponent: str = "random", lr: float = 3e-3, entropy_beta: float = 0.01,
          eval_every: int = 200, eval_games: int = 100, seed: int = 0, cfg: Optional[Dict[str, Any]] = None,
          policy: Optional[FlyPolicy] = None, graph: Optional[FlyGraph] = None, log_name: str = "train",
          verbose: bool = True) -> Dict[str, Any]:
    cfg = cfg or load_config()
    fly = cfg["fly"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    graph = graph or load_or_build_graph(cfg, verbose)
    policy = policy or make_policy(graph, cfg, seed)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    env = SVEnv(opponent=make_agent(opponent, random.Random(seed)), exclude_unparsed_cards=True, config=cfg)
    ckpt_dir = ensure_dir(Path(fly["checkpoint_dir"]))
    log_dir = ensure_dir(Path(fly["log_dir"]))
    log_path = log_dir / f"{log_name}.jsonl"
    baseline, history = 0.0, []
    wins = losses = excs = 0
    t0 = time.perf_counter()
    if verbose:
        print(f"policy: {policy.meta['n_neurons']:,} neurons, {policy.meta['n_edges']:,} edges, "
              f"rho {policy.meta['spectral_radius_raw']:.2f} -> {policy.meta['spectral_radius_target']} "
              f"(scale {policy.meta['scale']:.3g}); trainable {policy.trainable_summary()}")
    with open(log_path, "a", encoding="utf-8") as log:
        ev0 = evaluate(policy, opponent, eval_games, seed=10_000_000, cfg=cfg)  # untrained reference
        history.append({"game": 0, "eval": ev0})
        log.write(json.dumps({"game": 0, "eval": ev0, "meta": {k: v for k, v in policy.meta.items() if k != "graph_meta"}}) + "\n")
        if verbose:
            lo, hi = ev0["ci95"]
            print(f"[train] game 0 (untrained): eval vs {opponent}: {100 * ev0['winrate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] "
                  f"over {ev0['games']} | decision {ev0['decision_ms_mean']:.1f} ms mean / {ev0['decision_ms_max']:.0f} max")
        for g in range(1, games + 1):
            ret, logps, ents, steps, exc = play_episode(policy, env, seed=seed * 100000 + g, generator=gen)
            if exc:
                excs += 1
                log.write(json.dumps({"game": g, "exception": exc}) + "\n")
                continue
            wins += ret > 0
            losses += ret < 0
            baseline = 0.95 * baseline + 0.05 * ret if g > 1 else ret
            if logps:
                adv = ret - baseline
                loss = -adv * torch.stack(logps).sum() - entropy_beta * torch.stack(ents).sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            if g % eval_every == 0 or g == games:
                ev = evaluate(policy, opponent, eval_games, seed=10_000_000 + g, cfg=cfg)
                rec = {"game": g, "train_wins": wins, "train_losses": losses, "train_exceptions": excs,
                       "eval": ev, "elapsed_s": time.perf_counter() - t0}
                history.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save({"state_dict": policy.state_dict(), "meta": policy.meta, "game": g,
                            "config_fly": {k: str(v) for k, v in fly.items()}}, ckpt_dir / "latest.pt")
                torch.save({"state_dict": policy.state_dict(), "meta": policy.meta, "game": g},
                           ckpt_dir / f"game{g}.pt")
                if verbose:
                    lo, hi = ev["ci95"]
                    print(f"[train] game {g}: train W/L {wins}/{losses} exc {excs} | eval vs {opponent}: "
                          f"{100 * ev['winrate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] over {ev['games']} | "
                          f"decision {ev['decision_ms_mean']:.1f} ms mean / {ev['decision_ms_max']:.0f} max | "
                          f"{time.perf_counter() - t0:.0f}s")
    return {"history": history, "policy": policy, "graph": graph, "wins": wins, "losses": losses, "exceptions": excs}
