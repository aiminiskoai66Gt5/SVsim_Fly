"""M2: arena reproducibility, timeout handling, statistics."""

import sys
import types
import unittest

if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        class _Stub(types.ModuleType):
            def __getattr__(self, n):
                if n.startswith("__"):
                    raise AttributeError(n)
                return type(n, (), {})
        for _m in ("tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.simpledialog"):
            sys.modules[_m] = _Stub(_m)

from svai.arena.core import GameSpec, play_game, summarize, wilson
from svai.arena.runner import make_specs, run_specs


def _comparable(rec):
    d = rec.to_dict()
    for k in ("wall_time", "decision_time_a", "decision_time_b"):
        d.pop(k)
    return d


class TestM2Arena(unittest.TestCase):
    def test_same_spec_gives_same_record(self):
        for seed in (0, 1, 2, 3):
            spec = GameSpec("random", "random", seed=seed, a_is_player1=(seed % 2 == 0))
            a, b = play_game(spec), play_game(spec)
            self.assertEqual(_comparable(a), _comparable(b), f"seed {seed} not reproducible")
            self.assertIn(a.outcome, ("a", "b", "draw", "truncated", "exception"))
            self.assertGreater(a.decisions, 0)

    def test_sides_alternate_and_seeds_are_consecutive(self):
        specs = make_specs("random", "random", 6, seed=100)
        self.assertEqual([s.seed for s in specs], [100, 101, 102, 103, 104, 105])
        self.assertEqual([s.a_is_player1 for s in specs], [True, False] * 3)

    def test_worker_run_matches_in_process_run(self):
        specs = make_specs("random", "random", 4, seed=10)
        direct = [_comparable(play_game(s)) for s in specs]
        via_workers = [_comparable(r) for r in run_specs(specs, timeout=60, workers=2, progress=False)]
        self.assertEqual(direct, via_workers)

    def test_hanging_game_is_counted_not_waited_for(self):
        specs = [GameSpec("hang", "random", seed=5), GameSpec("random", "random", seed=6)]
        records = run_specs(specs, timeout=3, workers=2, progress=False)
        self.assertEqual(records[0].outcome, "timeout")
        self.assertNotEqual(records[1].outcome, "timeout")
        s = summarize(records, "hang", "random")
        self.assertEqual(s["timeouts"], 1)

    def test_wilson_interval(self):
        p, lo, hi = wilson(50, 100)
        self.assertAlmostEqual(p, 0.5)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(wilson(0, 0), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
