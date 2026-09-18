"""Replay one arena game in-process with the full traceback (engine debugging aid).

    python scripts/repro.py --seed 17 [--a random --b random] [--max-turns 40] [--verbose]

Reproduces exactly the game the arena played for that seed (same GameSpec
derivation). Prints the traceback of the engine exception, the turn, the
acting player and the last actions; --verbose also echoes the engine log of
the failing step.
"""

import argparse
import contextlib
import io
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main(argv=None) -> int:
    from svai.arena.runner import _headless_tkinter_stub
    _headless_tkinter_stub()
    import random
    from svai import actions as A
    from svai.agents import make_agent
    from svai.arena.core import GameSpec
    from svai.env import SVEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--a", default="random")
    ap.add_argument("--b", default="random")
    ap.add_argument("--a-player1", default=None, help="true/false; default: arena rule (seed parity)")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    a_is_p1 = (args.seed % 2 == 0) if args.a_player1 is None else args.a_player1.lower() == "true"
    spec = GameSpec(args.a, args.b, seed=args.seed, a_is_player1=a_is_p1, max_turns=args.max_turns)

    base = random.Random(spec.seed)
    agents = {"a": make_agent(spec.agent_a, random.Random(base.randrange(2**31))),
              "b": make_agent(spec.agent_b, random.Random(base.randrange(2**31)))}
    side_of = {"player1": "a" if a_is_p1 else "b", "player2": "b" if a_is_p1 else "a"}
    env = SVEnv(opponent=None, quiet=False, max_turns=spec.max_turns,
                option_agent={pid: agents[s] for pid, s in side_of.items()})
    history = []
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            obs, info = env.reset(seed=spec.seed)
        done = False
        while not done:
            player = info["current_player"]
            action = agents[side_of[player]].act(env.game, player, info["legal_actions"])
            history.append((info["turn"], player, action))
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                obs, r, term, trunc, info = env.step(A.encode(env.game, player, action))
            done = term or trunc
        print(f"seed {spec.seed}: finished without exception; winner={info['winner']} turn={info['turn']}")
        return 0
    except Exception:
        tb = traceback.format_exc()
        turn, player, action = history[-1] if history else (0, "?", None)
        print(f"seed {spec.seed}: EXCEPTION at turn {turn}, acting player {player}, action {action}")
        print("last 5 actions:")
        for h in history[-5:]:
            print("  ", h)
        if args.verbose:
            print("---- engine log of the failing step ----")
            print(log.getvalue()[-6000:])
        print("---- traceback ----")
        print(tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
