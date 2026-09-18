"""M5: connectome graph construction, policy constraints, training plumbing (synthetic release)."""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

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

import numpy as np
import torch

from svai import actions as A
from svai.config import load_config
from svai.env import OBS_SIZE
from svai.fly.graph import DEFAULT_SIGN, FlyGraph, build_graph, read_neurons, synthetic_release
from svai.fly.policy import FlyPolicy, spectral_radius


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.release = synthetic_release(Path(cls.tmp.name) / "release", seed=1)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.graph = build_graph(cls.release, subset="central_brain", min_synapses=3, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_status_filter_and_subset(self):
        neurons_all = read_neurons(self.release, status_values=None)
        neurons_traced = read_neurons(self.release, status_values=("Traced",))
        self.assertLess(len(neurons_traced), len(neurons_all))
        self.assertTrue((neurons_traced["status"] == "Traced").all())
        self.assertTrue(all(sc.startswith("cb_") or sc == "descending_neuron" for sc in self.graph.superclass))
        self.assertFalse(any(sc.startswith("vnc_") for sc in self.graph.superclass))

    def test_edge_signs_follow_presynaptic_transmitter(self):
        coo = self.graph.W.tocoo()
        for pre, val in zip(coo.col[:500], coo.data[:500]):
            nt = self.graph.transmitter[pre]
            expected = DEFAULT_SIGN.get(nt, 1.0)
            self.assertEqual(np.sign(val), expected, f"edge from {nt} has sign {np.sign(val)}")
            self.assertGreaterEqual(abs(val), 3)  # min_synapses applied

    def test_min_synapses_threshold(self):
        with contextlib.redirect_stdout(io.StringIO()):
            g1 = build_graph(self.release, subset="central_brain", min_synapses=1, verbose=False)
        self.assertGreater(g1.n_edges, self.graph.n_edges)

    def test_save_load_round_trip(self):
        path = Path(self.tmp.name) / "g.npz"
        self.graph.save(path)
        g2 = FlyGraph.load(path)
        self.assertEqual(g2.n, self.graph.n)
        self.assertEqual(g2.n_edges, self.graph.n_edges)
        np.testing.assert_array_equal(g2.W.data, self.graph.W.data)
        self.assertEqual(g2.meta["subset"], "central_brain")
        self.assertEqual(list(g2.output_idx), list(self.graph.output_idx))


class TestPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        release = synthetic_release(Path(cls.tmp.name) / "release", seed=2)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.graph = build_graph(release, subset="central_brain", min_synapses=3, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make(self, **kw):
        return FlyPolicy(self.graph, OBS_SIZE, A.ACTION_SPACE_SIZE, n_input=16, n_output=16, microsteps=3,
                         spectral_radius_target=0.9, **kw)

    def test_spectral_radius_rescaled(self):
        p = self.make()
        raw = spectral_radius(self.graph.W)
        self.assertGreater(raw, 1.0)
        W_eff = self.graph.W.copy().astype(np.float64) * float(p.scale)
        self.assertAlmostEqual(spectral_radius(W_eff), 0.9, places=2)
        self.assertAlmostEqual(p.meta["scale"], 0.9 / raw, places=6)

    def test_only_allowed_parameters_are_trainable(self):
        p = self.make()
        names = set(p.trainable_summary())
        self.assertEqual(names, {"input_proj.weight", "input_proj.bias", "readout.weight", "readout.bias",
                                 "log_gain", "alpha_logit"})
        p2 = self.make(learn_edge_gain=True)
        self.assertIn("edge_log_gain", p2.trainable_summary())

    def test_fixed_counts_and_signs_survive_training_step(self):
        p = self.make(learn_edge_gain=True)
        before = p.w_fixed.clone()
        obs = torch.zeros(1, OBS_SIZE)
        mask = np.zeros(A.ACTION_SPACE_SIZE, dtype=bool)
        mask[A.END_TURN_INDEX] = mask[0] = True
        opt = torch.optim.SGD([q for q in p.parameters() if q.requires_grad], lr=0.5)
        for _ in range(3):
            _, logp, _, _ = p.act(obs.numpy()[0], p.initial_state(), mask, sample=True)
            opt.zero_grad()
            (-logp).backward()
            opt.step()
        torch.testing.assert_close(p.w_fixed, before)
        w = p.edge_weights().detach()
        self.assertTrue(torch.equal(torch.sign(w), torch.sign(before)))   # signs never flip
        self.assertTrue(torch.all(torch.exp(p.edge_log_gain) > 0))

    def test_illegal_actions_never_sampled(self):
        p = self.make()
        mask = np.zeros(A.ACTION_SPACE_SIZE, dtype=bool)
        mask[A.END_TURN_INDEX] = True
        for _ in range(20):
            a, _, _, _ = p.act(np.random.rand(OBS_SIZE).astype(np.float32), p.initial_state(), mask, sample=True)
            self.assertEqual(a, A.END_TURN_INDEX)

    def test_deterministic_argmax(self):
        p = self.make()
        obs = np.random.default_rng(0).random(OBS_SIZE).astype(np.float32)
        mask = np.ones(A.ACTION_SPACE_SIZE, dtype=bool)
        a1, _, _, h1 = p.act(obs, p.initial_state(), mask, sample=False)
        a2, _, _, h2 = p.act(obs, p.initial_state(), mask, sample=False)
        self.assertEqual(a1, a2)
        torch.testing.assert_close(h1, h2)


class TestTrainingPlumbing(unittest.TestCase):
    def test_short_training_run_and_arena_agent(self):
        from svai.fly.train import train
        from svai.agents.fly_agent import FlyAgent
        from svai.arena.core import GameSpec, play_game
        tmp = tempfile.TemporaryDirectory()
        release = synthetic_release(Path(tmp.name) / "release", seed=3)
        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_graph(release, subset="central_brain", min_synapses=3, verbose=False)
        cfg = load_config()
        cfg["fly"]["checkpoint_dir"] = Path(tmp.name) / "ckpt"
        cfg["fly"]["log_dir"] = Path(tmp.name) / "log"
        cfg["fly"]["n_input"], cfg["fly"]["n_output"], cfg["fly"]["microsteps"] = 16, 16, 2
        out = train(games=6, opponent="random", eval_every=6, eval_games=4, seed=1, cfg=cfg, graph=graph, verbose=False)
        self.assertEqual(out["exceptions"], 0)
        self.assertTrue((Path(tmp.name) / "ckpt" / "latest.pt").exists())
        agent = FlyAgent(policy=out["policy"], graph=graph, cfg=cfg)
        # the agent plays a whole game inside the arena harness without illegal moves
        import svai.agents as reg
        reg._REGISTRY["fly_test"] = lambda r: agent
        rec = play_game(GameSpec("fly_test", "random", seed=5, exclude_unparsed_cards=True))
        self.assertNotEqual(rec.outcome, "exception", rec.exception_message)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
