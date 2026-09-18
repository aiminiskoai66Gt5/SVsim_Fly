"""Build a signed sparse graph from the MaleCNS v1.0 flat-connectome release.

Input files (Janelia, CC-BY 4.0, https://male-cns.janelia.org/download/), expected under
``paths.datasets``/malecns/ (config.toml):

    body-annotations-male-cns-v1.0-minconf-0.5.feather     (~13 MB)
    body-neurotransmitters-male-cns-v1.0.feather           (~42 MB)
    connectome-weights-male-cns-v1.0-minconf-0.5.feather   (~1.1 GB, 152M rows)

Output: ``FlyGraph`` - scipy.sparse CSR matrix W (post x pre, signed synapse counts),
the neuron id list, per-neuron type / superclass / transmitter tables and the input /
output neuron index sets. Saved as an ``.npz`` + ``.csv`` pair so training never reads
the 1.1 GB table again.

What this is NOT: a simulation of the animal. The release is a static wiring diagram
(neuron ids, synapse counts, predicted transmitters). Everything dynamic here - rates,
time constants, gains, input mapping, readout - is invented and learned (see policy.py).

The loader mirrors zhengxuyu/nfly (MIT): pyarrow batch filtering of the weight table,
Dale's-law signs from the presynaptic transmitter, superclass-based flow labels.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SHORT_NAMES = {"body-annotations.feather": ANNOTATIONS, "body-neurotransmitters.feather": TRANSMITTERS,
               "connectome-weights.feather": WEIGHTS}

# Sign of every outgoing edge, from the presynaptic neuron's predicted transmitter (Dale's law).
# Glutamate and histamine act mostly through chloride channels in Drosophila -> inhibitory.
# Unknown / unclear -> excitatory (same convention as nfly); the count is reported.
DEFAULT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0,
                "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}

AFFERENT = {"ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending",
            "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc"}
EFFERENT = {"vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "vnc_endocrine", "cb_endocrine",
            "efferent_ascending", "efferent_descending"}
DESCENDING = {"descending_neuron", "descending", "sensory_descending", "efferent_descending"}

# Named subsets (superclass rules). "central_brain": everything whose superclass starts with cb_
# plus the descending neurons (the natural readout), no optic lobes, no VNC.
SUBSETS = {
    "all": lambda sc: np.ones(len(sc), dtype=bool),
    "brain": lambda sc: ~np.char.startswith(sc.astype(str), "vnc_"),
    "central_brain": lambda sc: np.char.startswith(sc.astype(str), "cb_") | np.isin(sc, list(DESCENDING)),
}


@dataclasses.dataclass
class FlyGraph:
    W: sp.csr_matrix                 # (N, N) float32, W[post, pre] = sign * synapse count
    body_ids: np.ndarray             # (N,) int64
    superclass: np.ndarray           # (N,) str
    cell_type: np.ndarray            # (N,) str
    transmitter: np.ndarray          # (N,) str ("unclear" when unknown)
    status: np.ndarray               # (N,) str
    input_idx: np.ndarray            # indices of designated input neurons
    output_idx: np.ndarray           # indices of designated output (descending) neurons
    meta: Dict[str, object]

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def n_edges(self) -> int:
        return int(self.W.nnz)

    def memory_bytes(self) -> int:
        return int(self.W.data.nbytes + self.W.indices.nbytes + self.W.indptr.nbytes)

    def describe(self) -> str:
        exc = float((self.W.data > 0).mean()) if self.W.nnz else 0.0
        return (f"FlyGraph[{self.meta.get('subset', '?')}]: {self.n:,} neurons, {self.n_edges:,} edges "
                f"(min_synapses={self.meta.get('min_synapses')}), {exc:.1%} excitatory, "
                f"{len(self.input_idx):,} input / {len(self.output_idx):,} output neurons, "
                f"CSR {self.memory_bytes() / 1e6:.1f} MB")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        as_str = lambda a: np.asarray(a).astype(str)  # noqa: E731 - unicode arrays load without pickle
        np.savez_compressed(path, data=self.W.data, indices=self.W.indices, indptr=self.W.indptr,
                            shape=np.asarray(self.W.shape), body_ids=self.body_ids, superclass=as_str(self.superclass),
                            cell_type=as_str(self.cell_type), transmitter=as_str(self.transmitter),
                            status=as_str(self.status), input_idx=self.input_idx, output_idx=self.output_idx,
                            meta=np.asarray([repr(self.meta)]))
        return path

    @staticmethod
    def load(path: Path) -> "FlyGraph":
        z = np.load(Path(path), allow_pickle=False)
        W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"]))
        import ast
        meta = ast.literal_eval(str(z["meta"][0]))
        return FlyGraph(W, z["body_ids"], z["superclass"], z["cell_type"], z["transmitter"], z["status"],
                        z["input_idx"], z["output_idx"], meta)


def _resolve(data_dir: Path, name: str) -> Path:
    for cand in (name, {v: k for k, v in SHORT_NAMES.items()}.get(name, name)):
        if (data_dir / cand).exists():
            return data_dir / cand
    raise FileNotFoundError(f"{name} not found in {data_dir}. Download it from "
                            f"https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/{name}")


def _pick(names, *cands):
    for c in cands:
        if c in names:
            return c
    raise KeyError(f"none of {cands} in {list(names)}")


def read_neurons(data_dir: Path, status_values: Optional[Iterable[str]] = ("Traced",)):
    """Annotated neurons: body id, superclass, type, status, transmitter. Filters by status."""
    import pandas as pd
    import pyarrow.feather as feather

    ann = feather.read_feather(_resolve(data_dir, ANNOTATIONS))
    sc_col = _pick(ann.columns, "superclass", "super_class")
    ann = ann[ann[sc_col].notna()]
    status_col = next((c for c in ("status", "statusLabel", "status_label") if c in ann.columns), None)
    status = ann[status_col].astype(str) if status_col else pd.Series("unknown", index=ann.index)
    if status_values and status_col:
        keep = status.isin(list(status_values))
        log.info("status filter %s keeps %d of %d annotated neurons", list(status_values), int(keep.sum()), len(ann))
        ann, status = ann[keep], status[keep]
    id_col = _pick(ann.columns, "bodyId", "body", "root_id")
    type_col = next((c for c in ("type", "cell_type") if c in ann.columns), None)
    df = pd.DataFrame({
        "body_id": ann[id_col].astype(np.int64).to_numpy(),
        "superclass": ann[sc_col].astype(str).to_numpy(),
        "cell_type": (ann[type_col].astype(object).where(ann[type_col].notna(), "").astype(str).to_numpy()
                      if type_col else np.full(len(ann), "", dtype=object)),
        "status": status.to_numpy(),
    })
    nt = feather.read_feather(_resolve(data_dir, TRANSMITTERS))
    nt_id = _pick(nt.columns, "body", "bodyId", "root_id")
    nt_col = _pick(nt.columns, "consensus_nt", "predicted_nt", "nt")
    nt = pd.DataFrame({"body_id": nt[nt_id].astype(np.int64), "transmitter": nt[nt_col].astype(object)})
    df = df.merge(nt, on="body_id", how="left")
    df["transmitter"] = df["transmitter"].where(df["transmitter"].notna(), "unclear").astype(str).str.lower()
    return df.sort_values("body_id").reset_index(drop=True)


def read_edges(data_dir: Path, keep_ids: np.ndarray, min_synapses: int = 1):
    """(pre, post, count) rows between ``keep_ids`` with count >= min_synapses, filtered per batch."""
    import pyarrow as pa
    import pyarrow.compute as pc

    reader = pa.ipc.open_file(_resolve(data_dir, WEIGHTS))
    names = reader.schema.names
    pre, post, wt = _pick(names, "body_pre", "pre"), _pick(names, "body_post", "post"), _pick(names, "weight", "syn_count")
    ids = pa.array(np.asarray(keep_ids, dtype=np.int64))
    parts = []
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i).select([pre, post, wt])
        mask = pc.and_(pc.is_in(b.column(pre), value_set=ids), pc.is_in(b.column(post), value_set=ids))
        if min_synapses > 1:
            mask = pc.and_(mask, pc.greater_equal(b.column(wt), min_synapses))
        b = b.filter(mask)
        if b.num_rows:
            parts.append(b)
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    t = pa.Table.from_batches(parts)
    return (t.column(pre).to_numpy().astype(np.int64), t.column(post).to_numpy().astype(np.int64),
            t.column(wt).to_numpy().astype(np.float32))


def build_graph(data_dir: Path, subset: str = "central_brain", min_synapses: int = 3,
                status_values: Optional[Iterable[str]] = ("Traced",), sign_map: Optional[Dict[str, float]] = None,
                input_superclasses: Optional[Iterable[str]] = None, output_superclasses: Optional[Iterable[str]] = None,
                verbose: bool = True) -> FlyGraph:
    """Read the release and return the signed CSR graph of ``subset``."""
    data_dir = Path(data_dir)
    sign_map = sign_map or DEFAULT_SIGN
    neurons = read_neurons(data_dir, status_values)
    keep = SUBSETS[subset](neurons["superclass"].to_numpy())
    neurons = neurons[keep].reset_index(drop=True)
    ids = neurons["body_id"].to_numpy()
    pre, post, cnt = read_edges(data_dir, ids, min_synapses)
    lut = {int(b): i for i, b in enumerate(ids)}
    pre_i = np.fromiter((lut[int(b)] for b in pre), dtype=np.int64, count=len(pre))
    post_i = np.fromiter((lut[int(b)] for b in post), dtype=np.int64, count=len(post))
    nt = neurons["transmitter"].to_numpy()
    sign = np.array([sign_map.get(nt[p], 1.0) for p in pre_i], dtype=np.float32)
    unclear = int(sum(1 for p in pre_i if nt[p] not in sign_map))
    W = sp.csr_matrix((sign * cnt, (post_i, pre_i)), shape=(len(ids), len(ids)), dtype=np.float32)
    W.sum_duplicates()
    sc = neurons["superclass"].to_numpy()
    in_sc = set(input_superclasses) if input_superclasses else (AFFERENT | {"cb_sensory", "cb_sensory_tbc"})
    out_sc = set(output_superclasses) if output_superclasses else DESCENDING
    input_idx = np.flatnonzero(np.isin(sc, list(in_sc)))
    output_idx = np.flatnonzero(np.isin(sc, list(out_sc)))
    graph = FlyGraph(W, ids, sc, neurons["cell_type"].to_numpy().astype(str), nt.astype(str),
                     neurons["status"].to_numpy().astype(str), input_idx, output_idx,
                     {"subset": subset, "min_synapses": min_synapses, "status_values": list(status_values or []),
                      "sign_map": dict(sign_map), "edges_from_unclear_transmitter": unclear,
                      "source": "MaleCNS v1.0 flat-connectome (Janelia, CC-BY 4.0)"})
    if verbose:
        print(graph.describe())
        print(f"  edges from neurons with unclear transmitter (treated as +): {unclear:,}")
    return graph


def synthetic_release(data_dir: Path, n_central: int = 400, n_sensory: int = 40, n_dn: int = 30, n_vnc: int = 60,
                      edges: int = 8000, seed: int = 0) -> Path:
    """Write a small random release in the MaleCNS feather layout (tests / dry runs without the 1.1 GB file)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    nts = list(DEFAULT_SIGN) + ["unclear"]
    p = [0.5, 0.05, 0.03, 0.02, 0.2, 0.15, 0.02, 0.03]
    rows = []
    bid = 10000
    groups = [("cb_sensory", n_sensory), ("cb_intrinsic", n_central), ("descending_neuron", n_dn), ("vnc_motor", n_vnc)]
    for sc, n in groups:
        for i in range(n):
            bid += 1
            rows.append({"bodyId": bid, "superclass": sc, "type": f"{sc[:3].upper()}{i % 25}",
                         "status": "Traced" if rng.random() > 0.05 else "Roughly traced",
                         "nt": str(rng.choice(nts, p=p))})
    rows.append({"bodyId": bid + 1, "superclass": None, "type": None, "status": "Traced", "nt": None})
    df = pd.DataFrame(rows)
    ids = df["bodyId"].to_numpy()[:-1]
    pre = rng.choice(ids, edges)
    post = rng.choice(ids, edges)
    w = rng.integers(1, 30, edges)
    df[["bodyId", "superclass", "type", "status"]].to_feather(data_dir / "body-annotations.feather")
    nt = df[df["nt"].notna()]
    pd.DataFrame({"body": nt["bodyId"], "consensus_nt": nt["nt"]}).reset_index(drop=True).to_feather(
        data_dir / "body-neurotransmitters.feather")
    pd.DataFrame({"body_pre": pre, "body_post": post, "weight": w}).to_feather(data_dir / "connectome-weights.feather")
    return data_dir
