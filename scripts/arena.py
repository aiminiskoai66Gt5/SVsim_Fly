"""Headless agent-vs-agent evaluation.

    python scripts/arena.py --a random --b random --games 200 --seed 0 --timeout 30

Game i uses seed ``seed + i``; sides alternate (A is player1 on even i). Every
game runs in a worker process with a wall-clock timeout so a hang is counted,
not waited for. Output: a Markdown table on stdout and JSON + Markdown files
under ``paths.logs``/arena (config.toml).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main(argv=None) -> int:
    from svai.agents import agent_names
    from svai.arena.runner import run_arena

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="random", help=f"agent A ({', '.join(agent_names())})")
    ap.add_argument("--b", default="random", help="agent B")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="base seed; game i uses seed+i")
    ap.add_argument("--timeout", type=float, default=30.0, help="wall-clock seconds per game")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: cpu-1)")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--no-swap", action="store_true", help="A is always player1")
    ap.add_argument("--deck-mode", choices=["random", "files"], default="random")
    ap.add_argument("--deck-files", nargs="*", default=None)
    ap.add_argument("--label", default=None, help="output file name (default: timestamped)")
    ap.add_argument("--exclude-unparsed", action="store_true",
                    help="random decks only use cards whose effects the engine can fully execute")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    result = run_arena(args.a, args.b, args.games, args.seed, timeout=args.timeout, workers=args.workers,
                       swap_sides=not args.no_swap, max_turns=args.max_turns, deck_mode=args.deck_mode,
                       deck_files=args.deck_files, label=args.label, progress=not args.quiet,
                       exclude_unparsed_cards=args.exclude_unparsed)
    print(result["report"])
    print(f"written: {result['path']}.json / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
