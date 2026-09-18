"""Train the FlyAgent policy on CPU (REINFORCE) and log win rates.

    python scripts/train_fly.py --games 2000 --opponent random --eval-every 200 --eval-games 100
    python scripts/train_fly.py --graph data/cache/fly/synthetic_central_brain_min3.npz ...

Checkpoints: <fly.checkpoint_dir>/latest.pt (+ game<N>.pt); log: <fly.log_dir>/<name>.jsonl.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main(argv=None) -> int:
    from svai.arena.runner import _headless_tkinter_stub
    _headless_tkinter_stub()
    from svai.config import load_config
    from svai.fly.graph import FlyGraph
    from svai.fly.train import train

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--opponent", default="random", help="random | greedy")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", default=None, help="a built .npz graph (default: fly.graph_cache/<subset>_min<k>.npz)")
    ap.add_argument("--name", default="train")
    args = ap.parse_args(argv)
    cfg = load_config()
    graph = FlyGraph.load(args.graph) if args.graph else None
    train(games=args.games, opponent=args.opponent, lr=args.lr, entropy_beta=args.entropy,
          eval_every=args.eval_every, eval_games=args.eval_games, seed=args.seed, cfg=cfg, graph=graph,
          log_name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
