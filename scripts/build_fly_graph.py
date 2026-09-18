"""Build the FlyAgent graph from the MaleCNS release files (run once per subset).

    python scripts/build_fly_graph.py [--subset central_brain] [--min-synapses 3]

Reads <fly.release_dir> (config.toml), writes <fly.graph_cache>/<subset>_min<k>.npz and
prints neuron / edge counts and memory. Use --synthetic to write and use a small random
release instead (dry run without the 1.1 GB download).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main(argv=None) -> int:
    from svai.config import ensure_dir, load_config
    from svai.fly.graph import build_graph, synthetic_release

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--min-synapses", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="write a small random release and build from it")
    args = ap.parse_args(argv)
    cfg = load_config()
    fly = cfg["fly"]
    subset = args.subset or fly["subset"]
    min_syn = args.min_synapses if args.min_synapses is not None else int(fly["min_synapses"])
    release = Path(fly["release_dir"])
    if args.synthetic:
        release = ensure_dir(Path(fly["release_dir"]).parent / "malecns_synthetic")
        synthetic_release(release)
        print(f"synthetic release written to {release}")
    graph = build_graph(release, subset=subset, min_synapses=min_syn, status_values=fly["status_values"])
    out = ensure_dir(Path(fly["graph_cache"])) / (f"{subset}_min{min_syn}.npz" if not args.synthetic
                                                  else f"synthetic_{subset}_min{min_syn}.npz")
    graph.save(out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
