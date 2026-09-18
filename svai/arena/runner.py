"""Run many arena games in worker processes with a wall-clock timeout per game.

Workers are persistent (each loads the card database once). A game that
exceeds ``timeout`` gets its worker killed and is recorded as ``timeout``; a
fresh worker replaces it. Results are independent of scheduling because every
game is seeded on its own (``GameSpec.seed``).
"""

import json
import multiprocessing as mp
import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from svai.arena.core import GameRecord, GameSpec, format_report, play_game, summarize
from svai.config import ensure_dir, load_config


def _headless_tkinter_stub() -> None:
    """deck_builder imports tkinter at module level; arena workers never need a display."""
    import types
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        class _Stub(types.ModuleType):
            def __getattr__(self, n):
                if n.startswith("__"):
                    raise AttributeError(n)
                return type(n, (), {})
        for m in ("tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.simpledialog"):
            sys.modules[m] = _Stub(m)


def _worker(task_q: "mp.Queue", result_q: "mp.Queue", repo_root: str) -> None:
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)
    _headless_tkinter_stub()
    from svai.arena.core import play_game as _play  # re-import inside the spawned process
    while True:
        item = task_q.get()
        if item is None:
            return
        spec: GameSpec = item
        result_q.put((spec.seed, _play(spec)))


def make_specs(agent_a: str, agent_b: str, games: int, seed: int, swap_sides: bool = True,
               **kwargs) -> List[GameSpec]:
    """Game i uses seed ``seed + i``; sides alternate when ``swap_sides``."""
    return [GameSpec(agent_a=agent_a, agent_b=agent_b, seed=seed + i,
                     a_is_player1=(i % 2 == 0) if swap_sides else True, **kwargs)
            for i in range(games)]


def run_specs(specs: Iterable[GameSpec], timeout: float = 30.0, workers: Optional[int] = None,
              progress: bool = True) -> List[GameRecord]:
    """Play ``specs`` in worker processes; return records sorted by seed."""
    specs = list(specs)
    if not specs:
        return []
    workers = workers or max(1, min(len(specs), (os.cpu_count() or 2) - 1))
    ctx = mp.get_context("spawn")
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    pending = list(reversed(specs))          # pop() gives ascending seeds
    records: Dict[int, GameRecord] = {}
    spec_by_seed = {s.seed: s for s in specs}
    procs: List[Dict[str, Any]] = []         # {"proc", "task_q", "result_q", "spec", "started"}

    def spawn() -> Dict[str, Any]:
        tq, rq = ctx.Queue(), ctx.Queue()
        p = ctx.Process(target=_worker, args=(tq, rq, repo_root), daemon=True)
        p.start()
        return {"proc": p, "task_q": tq, "result_q": rq, "spec": None, "started": 0.0}

    def assign(w: Dict[str, Any]) -> None:
        if pending:
            w["spec"] = pending.pop()
            w["started"] = time.perf_counter()
            w["task_q"].put(w["spec"])

    for _ in range(workers):
        w = spawn()
        assign(w)
        procs.append(w)

    t_start = time.perf_counter()
    last_print = 0.0
    try:
        while len(records) < len(specs):
            now = time.perf_counter()
            for w in procs:
                if w["spec"] is None:
                    continue
                try:
                    seed, rec = w["result_q"].get_nowait()
                except queue.Empty:
                    if now - w["started"] > timeout:
                        spec = w["spec"]
                        w["proc"].kill()
                        w["proc"].join(5)
                        rec = GameRecord(seed=spec.seed, agent_a=spec.agent_a, agent_b=spec.agent_b,
                                         a_is_player1=spec.a_is_player1, outcome="timeout",
                                         wall_time=now - w["started"])
                        records[spec.seed] = rec
                        fresh = spawn()
                        procs[procs.index(w)] = fresh
                        assign(fresh)
                    continue
                records[seed] = rec
                w["spec"] = None
                assign(w)
            if progress and now - last_print > 2.0:
                last_print = now
                done = len(records)
                print(f"[arena] {done}/{len(specs)} games, {now - t_start:.0f}s", file=sys.stderr, flush=True)
            time.sleep(0.005)
    finally:
        for w in procs:
            try:
                w["task_q"].put(None)
            except Exception:  # noqa: BLE001
                pass
        for w in procs:
            w["proc"].join(2)
            if w["proc"].is_alive():
                w["proc"].kill()
    return [records[s.seed] for s in specs]


def run_arena(agent_a: str, agent_b: str, games: int, seed: int, timeout: float = 30.0,
              workers: Optional[int] = None, swap_sides: bool = True, max_turns: int = 40,
              deck_mode: str = "random", deck_files: Optional[List[str]] = None,
              out_dir: Optional[Path] = None, label: Optional[str] = None,
              progress: bool = True) -> Dict[str, Any]:
    """Play the match, print the report, and write JSON + Markdown under the logs dir."""
    specs = make_specs(agent_a, agent_b, games, seed, swap_sides=swap_sides, max_turns=max_turns,
                       deck_mode=deck_mode, deck_files=deck_files or [])
    t0 = time.perf_counter()
    records = run_specs(specs, timeout=timeout, workers=workers, progress=progress)
    elapsed = time.perf_counter() - t0
    summary = summarize(records, agent_a, agent_b)
    summary["seed"] = seed
    summary["timeout"] = timeout
    summary["elapsed_s"] = elapsed
    report = format_report(summary) + f"\n\nseed base {seed}, per-game timeout {timeout}s, wall {elapsed:.1f}s, workers {workers or 'auto'}\n"

    cfg = load_config()
    out_dir = ensure_dir(Path(out_dir) if out_dir else Path(cfg["paths"]["logs"]) / "arena")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = label or f"{stamp}_{agent_a}_vs_{agent_b}_{games}g_seed{seed}"
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": {k: v for k, v in summary.items() if not k.startswith("decision_time")},
                   "records": [r.to_dict() for r in records]}, f, ensure_ascii=False, indent=1)
    with open(out_dir / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(report)
    return {"summary": summary, "records": records, "report": report, "path": str(out_dir / name)}
