"""Connectome-constrained recurrent policy (the "fly" opponent).

Not a simulation of the animal. The MaleCNS release gives a static wiring
diagram: which neuron contacts which, with how many synapses, and a predicted
transmitter. That is all we use from biology, and it is used only as a
*constraint*: the recurrent matrix has the connectome's sparsity pattern and
edge signs, nothing else is biological. Rates, time constants, gains, the
input mapping and the readout are invented here and learned.

Dynamics (one microstep, per neuron i):

    x_i  = g_i * sum_j W_ij h_j + u_i
    h_i <- (1 - a_i) h_i + a_i * clamp(relu(x_i), 0, h_max)

with W = scale * sign * synapse_count (fixed), g_i a learnable gain, a_i a
learnable leak (1 / time constant), u the projected observation injected into
the designated input neurons. ``scale`` rescales W to a target spectral radius
(slightly below 1) so the recurrence neither dies nor explodes; the value is
stored in ``meta``. Optional: a learnable positive per-edge gain (multiplies
the magnitude, never flips the sign; the original counts stay in ``w_fixed``).

Trainable parameters: input projection, readout, per-neuron gain, per-neuron
time constant, (optional) per-edge log-gain. Synapse counts and signs are not.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from svai.fly.graph import FlyGraph


def spectral_radius(W: sp.csr_matrix, iters: int = 100, seed: int = 0) -> float:
    """Dominant eigenvalue magnitude of a (signed) sparse matrix by power iteration."""
    n = W.shape[0]
    if n == 0 or W.nnz == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n).astype(np.float64)
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    Wd = W.astype(np.float64)
    for _ in range(iters):
        w = Wd @ v
        nrm = np.linalg.norm(w)
        if nrm < 1e-30:
            return 0.0
        lam = nrm
        v = w / nrm
    return float(lam)


class FlyPolicy(nn.Module):
    def __init__(self, graph: FlyGraph, obs_dim: int, n_actions: int, n_input: int = 256, n_output: int = 256,
                 microsteps: int = 4, spectral_radius_target: float = 0.9, alpha_init: float = 0.3,
                 h_max: float = 10.0, learn_edge_gain: bool = False, input_gain: float = 1.0, seed: int = 0):
        super().__init__()
        self.n = graph.n
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.microsteps = microsteps
        self.h_max = h_max
        rng = np.random.default_rng(seed)

        coo = graph.W.tocoo()
        self.register_buffer("post", torch.as_tensor(coo.row.astype(np.int64)))
        self.register_buffer("pre", torch.as_tensor(coo.col.astype(np.int64)))
        self.register_buffer("w_fixed", torch.as_tensor(coo.data.astype(np.float32)))  # signed counts, never trained
        rho = spectral_radius(graph.W)
        scale = spectral_radius_target / rho if rho > 0 else 1.0
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.meta: Dict[str, object] = {"spectral_radius_raw": rho, "spectral_radius_target": spectral_radius_target,
                                        "scale": float(scale), "n_neurons": self.n, "n_edges": int(coo.nnz),
                                        "graph_meta": dict(graph.meta)}

        # designated input / output neurons (subsets of the graph's sets; random but seeded)
        in_pool = graph.input_idx if len(graph.input_idx) else np.arange(self.n)
        out_pool = graph.output_idx if len(graph.output_idx) else np.arange(self.n)
        in_idx = rng.choice(in_pool, size=min(n_input, len(in_pool)), replace=False)
        out_idx = rng.choice(out_pool, size=min(n_output, len(out_pool)), replace=False)
        self.register_buffer("input_idx", torch.as_tensor(np.sort(in_idx).astype(np.int64)))
        self.register_buffer("output_idx", torch.as_tensor(np.sort(out_idx).astype(np.int64)))

        # trainable
        self.input_proj = nn.Linear(obs_dim, len(self.input_idx))
        self.readout = nn.Linear(len(self.output_idx), n_actions)
        self.log_gain = nn.Parameter(torch.zeros(self.n))                       # per-neuron gain g = exp
        a = torch.full((self.n,), float(alpha_init))
        self.alpha_logit = nn.Parameter(torch.log(a / (1 - a)))                  # per-neuron leak
        self.edge_log_gain = nn.Parameter(torch.zeros(coo.nnz), requires_grad=learn_edge_gain)
        self.input_gain = input_gain
        nn.init.zeros_(self.readout.bias)

    # --- derived -----------------------------------------------------------
    def edge_weights(self) -> torch.Tensor:
        """Effective signed weights: scale * sign * count * exp(edge gain)."""
        return self.w_fixed * self.scale * torch.exp(self.edge_log_gain)

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.n, dtype=torch.float32)

    def _recurrent(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(h)
        y.index_add_(1, self.post, h[:, self.pre] * w)
        return y

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs (B, obs_dim), h (B, N) -> logits (B, n_actions), new h."""
        w = self.edge_weights()
        gain = torch.exp(self.log_gain)
        alpha = torch.sigmoid(self.alpha_logit)
        u = torch.zeros_like(h)
        u[:, self.input_idx] = self.input_gain * self.input_proj(obs)
        for _ in range(self.microsteps):
            x = gain * self._recurrent(h, w) + u
            x = torch.relu(x).clamp(max=self.h_max)
            h = (1 - alpha) * h + alpha * x
        logits = self.readout(h[:, self.output_idx])
        return logits, h

    def act(self, obs: np.ndarray, h: torch.Tensor, legal_mask: np.ndarray, sample: bool = True,
            generator: Optional[torch.Generator] = None):
        """One decision. Returns (action index, log-prob, entropy, new h). Illegal actions get -inf."""
        logits, h = self.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), h)
        mask = torch.as_tensor(np.asarray(legal_mask, dtype=bool)).unsqueeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            a = torch.multinomial(dist.probs, 1, generator=generator).squeeze(1)
        else:
            a = torch.argmax(logits, dim=1)
        return int(a.item()), dist.log_prob(a).squeeze(0), dist.entropy().squeeze(0), h

    def trainable_summary(self) -> Dict[str, int]:
        return {name: int(p.numel()) for name, p in self.named_parameters() if p.requires_grad}
