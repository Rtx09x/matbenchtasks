from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .configs import TaskConfig


MAT2VEC_URL = "https://storage.googleapis.com/mat2vec/"
MAT2VEC_FILES = (
    "pretrained_embeddings",
    "pretrained_embeddings.wv.vectors.npy",
    "pretrained_embeddings.trainables.syn1neg.npy",
)
MAX_NEIGHBORS = 12
CUTOFF = 8.0
N_RBF_DIST = 40
N_RBF_ANGLE = 8


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _nan_to_num(arr: Sequence[float]) -> np.ndarray:
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def gaussian_rbf(values: np.ndarray, n_bins: int, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(vmin, vmax, n_bins, dtype=np.float32).reshape(1, -1)
    gamma = 1.0 / max((centers[0, 1] - centers[0, 0]) ** 2, 1e-6)
    return np.exp(-gamma * (values - centers) ** 2).astype(np.float32)


def _composition_from_obj(obj):
    from pymatgen.core import Composition

    if hasattr(obj, "composition"):
        return obj.composition
    return Composition(str(obj))


def structure_fingerprint(structure) -> str:
    try:
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float64).round(4).reshape(-1)
        coords = np.asarray(structure.frac_coords, dtype=np.float64).round(4).reshape(-1)
        z = [int(site.specie.Z) for site in structure]
        payload = {
            "formula": structure.composition.reduced_formula,
            "n": len(structure),
            "volume": round(float(structure.volume), 4),
            "z": z,
            "lattice": lattice.tolist(),
            "coords_head": coords[: min(60, coords.size)].tolist(),
        }
    except Exception:
        payload = {"fallback": str(structure)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class Mat2VecPooler:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.kv = None
        self.available = False
        self._load()

    def _load(self) -> None:
        try:
            from gensim.models import Word2Vec

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for name in MAT2VEC_FILES:
                path = self.cache_dir / name
                if not path.exists():
                    print(f"[features] downloading Mat2Vec {name}", flush=True)
                    urllib.request.urlretrieve(MAT2VEC_URL + name, path)
            t0 = time.time()
            model = Word2Vec.load(str(self.cache_dir / "pretrained_embeddings"), mmap="r")
            self.kv = model.wv
            self.available = True
            print(f"[features] Mat2Vec ready in {time.time() - t0:.1f}s vocab={len(self.kv)}", flush=True)
        except Exception as exc:
            print(f"[features] Mat2Vec unavailable; using zero embeddings. Reason: {exc}")
            self.kv = None
            self.available = False

    def pool(self, comp) -> np.ndarray:
        vec = np.zeros(200, dtype=np.float32)
        total = 0.0
        if self.kv is None:
            return vec
        try:
            for symbol, amount in comp.get_el_amt_dict().items():
                if symbol in self.kv.key_to_index:
                    vec += float(amount) * self.kv[symbol]
                    total += float(amount)
        except Exception:
            return vec
        if total > 0:
            vec /= total
        return vec


class CompositionFeatureBuilder:
    def __init__(self, cache_dir: Path, flavor: str):
        self.cache_dir = cache_dir
        self.flavor = flavor
        self.pooler = Mat2VecPooler(cache_dir / "mat2vec")
        self.extra_sizes: Dict[str, int] = {}
        self._init_featurizers()

    def _init_featurizers(self) -> None:
        from matminer.featurizers.composition import ElementProperty

        self.magpie = ElementProperty.from_preset("magpie")
        self.n_magpie = len(self.magpie.feature_labels())

        from matminer.featurizers.composition import BandCenter, IonProperty, Stoichiometry, ValenceOrbital
        from matminer.featurizers.composition.element import ElementFraction, TMetalFraction

        extras = [("ElementFraction", ElementFraction()), ("Stoichiometry", Stoichiometry()), ("ValenceOrbital", ValenceOrbital())]
        if self.flavor == "electronic_hybrid":
            extras.append(("IonProperty", IonProperty()))
        if "electronic" in self.flavor or self.flavor in {"formation_graph", "elastic_graph"}:
            extras.append(("BandCenter", BandCenter()))
        extras.append(("TMetalFraction", TMetalFraction()))
        self.extra_featurizers = extras
        for name, featurizer in self.extra_featurizers:
            try:
                self.extra_sizes[name] = len(featurizer.feature_labels())
            except Exception:
                self.extra_sizes[name] = 1

    def _magpie(self, comp) -> np.ndarray:
        try:
            return _nan_to_num(self.magpie.featurize(comp))
        except Exception:
            return np.zeros(self.n_magpie, dtype=np.float32)

    def _extras(self, comp) -> np.ndarray:
        parts = []
        for name, featurizer in self.extra_featurizers:
            try:
                values = _nan_to_num(featurizer.featurize(comp))
                self.extra_sizes[name] = len(values)
            except Exception:
                values = np.zeros(self.extra_sizes.get(name, 1), dtype=np.float32)
            parts.append(values)
        parts.append(_homo_lumo_features(comp))
        parts.append(_composition_sensor_features(comp))
        return np.concatenate(parts).astype(np.float32)

    def build(self, comps: Sequence, structures: Optional[Sequence] = None) -> np.ndarray:
        rows = []
        for idx, comp in enumerate(tqdm(comps, desc="composition features", leave=False)):
            structure = structures[idx] if structures is not None else None
            row = [
                self._magpie(comp),
                self._extras(comp),
                _structure_metadata(structure),
                _perovskite_features(comp),
                self.pooler.pool(comp),
            ]
            rows.append(np.concatenate(row).astype(np.float32))
        return np.vstack(rows).astype(np.float32)


def _homo_lumo_features(comp) -> np.ndarray:
    # Small hand-built orbital prior. Missing values degrade to zeros.
    orbital = {
        "H": (-13.6, -13.6), "Li": (-5.4, -5.4), "Be": (-9.3, -9.3),
        "B": (-8.3, -14.0), "C": (-11.3, -19.4), "N": (-14.5, -25.6),
        "O": (-13.6, -32.4), "F": (-17.4, -40.2), "Na": (-5.1, -5.1),
        "Mg": (-7.6, -7.6), "Al": (-6.0, -11.3), "Si": (-8.2, -15.0),
        "P": (-10.5, -18.7), "S": (-10.4, -22.7), "Cl": (-13.0, -25.3),
        "K": (-4.3, -4.3), "Ca": (-6.1, -6.1), "Ti": (-8.5, -6.8),
        "V": (-8.3, -6.7), "Cr": (-8.7, -6.8), "Mn": (-9.5, -7.4),
        "Fe": (-10.0, -7.9), "Co": (-10.0, -7.9), "Ni": (-10.0, -7.6),
        "Cu": (-11.7, -7.7), "Zn": (-17.3, -9.4), "Ga": (-6.0, -12.6),
        "Ge": (-7.9, -15.6), "As": (-9.8, -18.6), "Se": (-9.8, -21.1),
        "Br": (-11.8, -24.0), "Rb": (-4.2, -4.2), "Sr": (-5.7, -5.7),
        "Y": (-7.4, -6.5), "Zr": (-8.3, -6.8), "Nb": (-8.5, -6.9),
        "Mo": (-8.9, -7.1), "Ru": (-8.7, -7.4), "Rh": (-8.8, -7.5),
        "Pd": (-8.3, -8.3), "Ag": (-12.3, -7.6), "Cd": (-16.7, -9.0),
        "In": (-5.8, -12.0), "Sn": (-7.3, -14.6), "Sb": (-8.6, -16.5),
        "Te": (-9.0, -19.0), "I": (-10.5, -21.1), "Cs": (-3.9, -3.9),
        "Ba": (-5.2, -5.2), "La": (-7.5, -5.6), "Hf": (-8.1, -7.0),
        "Ta": (-9.6, -7.9), "W": (-9.8, -8.0), "Re": (-9.2, -7.9),
        "Os": (-10.0, -8.4), "Ir": (-10.7, -9.1), "Pt": (-10.5, -9.0),
        "Au": (-12.8, -9.2), "Pb": (-7.4, -15.0), "Bi": (-7.3, -16.7),
    }
    total = 0.0
    homo = 0.0
    lumo = 0.0
    seen = 0
    try:
        for symbol, amount in comp.get_el_amt_dict().items():
            if symbol not in orbital:
                continue
            h, l = orbital[symbol]
            homo += float(amount) * h
            lumo += float(amount) * l
            total += float(amount)
            seen += 1
    except Exception:
        return np.zeros(3, dtype=np.float32)
    if total <= 0:
        return np.zeros(3, dtype=np.float32)
    homo /= total
    lumo /= total
    return np.array([homo, lumo, abs(lumo - homo) if seen else 0.0], dtype=np.float32)


def _composition_sensor_features(comp) -> np.ndarray:
    vals = []
    try:
        elements = list(comp.elements)
        amounts = np.array([float(comp[el]) for el in elements], dtype=np.float32)
        frac = amounts / max(float(amounts.sum()), 1e-8)
        en = np.array([_safe_float(el.X) for el in elements], dtype=np.float32)
        radii = np.array([_safe_float(getattr(el, "atomic_radius", 0.0)) for el in elements], dtype=np.float32)
        masses = np.array([_safe_float(el.atomic_mass) for el in elements], dtype=np.float32)
        valence = np.array([_safe_float(sum(getattr(el, "full_electronic_structure", [])[-1][2:3] or [0])) for el in elements], dtype=np.float32)
        vals.extend([
            float(np.dot(frac, en)), float(en.max() - en.min()) if len(en) else 0.0,
            float(np.dot(frac, radii)), float(radii.max() - radii.min()) if len(radii) else 0.0,
            float(np.dot(frac, masses)), float(masses.max() / max(masses.min(), 1e-6)) if len(masses) else 0.0,
            float(np.dot(frac, radii ** 3)), float(np.dot(frac, valence)),
        ])
    except Exception:
        vals = [0.0] * 8
    return np.array(vals, dtype=np.float32)


def _structure_metadata(structure) -> np.ndarray:
    if structure is None:
        return np.zeros(11, dtype=np.float32)
    vals = []
    try:
        lattice = structure.lattice
        vals.extend([lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma])
        vals.append(structure.volume / max(len(structure), 1))
        vals.append(structure.density)
        vals.append(float(len(structure)))
        # Space group analysis is too slow for notebook smoke/full sequential runs.
        vals.append(0.0)
        try:
            total = 0.0
            for site in structure:
                r = _safe_float(getattr(site.specie, "atomic_radius", 0.0))
                total += (4.0 / 3.0) * math.pi * r ** 3
            vals.append(total / max(float(structure.volume), 1e-8))
        except Exception:
            vals.append(0.0)
    except Exception:
        vals = [0.0] * 11
    return _nan_to_num(vals[:11] + [0.0] * max(0, 11 - len(vals)))


def _perovskite_features(comp) -> np.ndarray:
    try:
        elements = list(comp.elements)
        if not elements:
            return np.zeros(6, dtype=np.float32)
        frac = np.array([float(comp[el]) for el in elements], dtype=np.float32)
        frac /= max(float(frac.sum()), 1e-8)
        radii = np.array([_safe_float(getattr(el, "atomic_radius", 0.0)) for el in elements], dtype=np.float32)
        en = np.array([_safe_float(el.X) for el in elements], dtype=np.float32)
        r_small = float(np.min(radii)) if len(radii) else 0.0
        r_mid = float(np.median(radii)) if len(radii) else 0.0
        r_large = float(np.max(radii)) if len(radii) else 0.0
        tolerance = (r_large + r_small) / (math.sqrt(2.0) * max(r_mid + r_small, 1e-6))
        octa = r_mid / max(r_small, 1e-6)
        return np.array([
            tolerance,
            octa,
            r_large / max(r_small, 1e-6),
            float(np.dot(frac, radii)),
            float(en.max() - en.min()) if len(en) else 0.0,
            float(len(elements)),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


def build_element_table() -> np.ndarray:
    from pymatgen.core.periodic_table import Element

    table = np.zeros((103, 12), dtype=np.float32)
    for z in range(1, 103):
        try:
            el = Element.from_Z(z)
            row = [
                z,
                _safe_float(el.atomic_mass),
                _safe_float(el.X),
                _safe_float(getattr(el, "atomic_radius", 0.0)),
                _safe_float(getattr(el, "average_ionic_radius", 0.0)),
                _safe_float(getattr(el, "row", 0.0)),
                _safe_float(getattr(el, "group", 0.0)),
                _safe_float(getattr(el, "mendeleev_no", 0.0)),
                _safe_float(getattr(el, "melting_point", 0.0)),
                _safe_float(getattr(el, "boiling_point", 0.0)),
                _safe_float(getattr(el, "density_of_solid", 0.0)),
                float(1 if getattr(el, "is_transition_metal", False) else 0),
            ]
            table[z] = _nan_to_num(row)
        except Exception:
            continue
    mean = np.nanmean(table[1:], axis=0)
    std = np.nanstd(table[1:], axis=0) + 1e-6
    table[1:] = (table[1:] - mean) / std
    return table.astype(np.float32)


def _build_graph_worker(payload):
    structure, elem_table = payload
    return build_graph(structure, elem_table)


def build_graphs(structures: Sequence, workers: int = 1) -> List[Dict[str, torch.Tensor]]:
    elem_table = build_element_table()
    if workers and workers > 1:
        # Thread workers avoid notebook/container ProcessPool hangs caused by
        # pickling large pymatgen Structure objects and forking after CUDA libs load.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(tqdm(pool.map(_build_graph_worker, [(s, elem_table) for s in structures]), total=len(structures), desc="graph features"))
    return [build_graph(s, elem_table) for s in tqdm(structures, desc="graph features", leave=False)]


def build_graph(structure, elem_table: np.ndarray) -> Dict[str, torch.Tensor]:
    atom_z = []
    atom_features = []
    for site in structure:
        z = int(getattr(site.specie, "Z", 0))
        atom_z.append(z)
        base = elem_table[z] if 0 <= z < len(elem_table) else np.zeros(12, dtype=np.float32)
        atom_features.append(np.concatenate([base, np.zeros(6, dtype=np.float32)]))

    src_list: List[int] = []
    dst_list: List[int] = []
    vecs: List[np.ndarray] = []
    dists: List[float] = []
    physics: List[np.ndarray] = []
    try:
        all_neighbors = structure.get_all_neighbors(CUTOFF)
        for i, neighs in enumerate(all_neighbors):
            neighs = sorted(neighs, key=lambda n: n.nn_distance)[:MAX_NEIGHBORS]
            for nb in neighs:
                j = int(nb.index)
                if i == j and nb.nn_distance < 1e-6:
                    continue
                src_list.append(i)
                dst_list.append(j)
                vec = np.asarray(nb.coords - structure[i].coords, dtype=np.float32)
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-6:
                    continue
                vecs.append(vec / dist)
                dists.append(dist)
                physics.append(_bond_physics(structure[i].specie, structure[j].specie, dist))
    except Exception:
        pass

    n_atoms = len(atom_z)
    if not dists:
        if n_atoms >= 2:
            src_list, dst_list = [0, 1], [1, 0]
            vecs = [np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([-1.0, 0.0, 0.0], dtype=np.float32)]
            dists = [1.0, 1.0]
            physics = [np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]
        else:
            src_list, dst_list = [0], [0]
            vecs = [np.zeros(3, dtype=np.float32)]
            dists = [1.0]
            physics = [np.zeros(8, dtype=np.float32)]

    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    edge_vec = np.vstack(vecs).astype(np.float32)
    edge_dist = np.asarray(dists, dtype=np.float32)
    edge_rbf = gaussian_rbf(edge_dist, N_RBF_DIST, 0.0, CUTOFF)
    edge_physics = np.vstack(physics).astype(np.float32)
    triplets, angle_feat = _build_triplets(edge_index, edge_vec)

    atom_features = np.vstack(atom_features).astype(np.float32)
    degree = np.bincount(edge_index[1], minlength=n_atoms).astype(np.float32)
    atom_features[:, 12] = degree / max(MAX_NEIGHBORS, 1)
    atom_features[:, 13] = len(edge_dist) / max(n_atoms, 1)
    atom_features[:, 14] = _safe_float(structure.volume / max(n_atoms, 1))
    atom_features[:, 15] = _safe_float(structure.density)
    atom_features[:, 16] = float(n_atoms)
    atom_features[:, 17] = float(len(set(atom_z)))

    return {
        "atom_z": torch.tensor(atom_z, dtype=torch.long),
        "atom_feat": torch.tensor(atom_features, dtype=torch.float32),
        "ei": torch.tensor(edge_index, dtype=torch.long),
        "rbf": torch.tensor(edge_rbf, dtype=torch.float32),
        "vec": torch.tensor(edge_vec, dtype=torch.float32),
        "phys": torch.tensor(edge_physics, dtype=torch.float32),
        "triplets": torch.tensor(triplets, dtype=torch.long),
        "angle_feat": torch.tensor(angle_feat, dtype=torch.float32),
        "n_atoms": int(n_atoms),
        "n_edges": int(edge_index.shape[1]),
    }


def _bond_physics(el_i, el_j, dist: float) -> np.ndarray:
    zi = _safe_float(getattr(el_i, "Z", 0.0))
    zj = _safe_float(getattr(el_j, "Z", 0.0))
    en_i = _safe_float(getattr(el_i, "X", 0.0))
    en_j = _safe_float(getattr(el_j, "X", 0.0))
    m_i = _safe_float(getattr(el_i, "atomic_mass", 0.0), 1.0)
    m_j = _safe_float(getattr(el_j, "atomic_mass", 0.0), 1.0)
    r_i = _safe_float(getattr(el_i, "atomic_radius", 0.0), 1.0)
    r_j = _safe_float(getattr(el_j, "atomic_radius", 0.0), 1.0)
    en_diff = abs(en_i - en_j)
    mu = (m_i * m_j) / max(m_i + m_j, 1e-6)
    k_est = (en_diff + 0.1) / max(dist * dist, 1e-6)
    omega = math.sqrt(max(k_est / max(mu, 1e-6), 0.0))
    return np.array([
        dist,
        1.0 / max(dist, 1e-6),
        en_diff,
        mu,
        omega,
        (r_i + r_j) / max(dist, 1e-6),
        max(m_i, m_j) / max(min(m_i, m_j), 1e-6),
        abs(zi - zj) / 100.0,
    ], dtype=np.float32)


def _build_triplets(edge_index: np.ndarray, edge_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    incoming: Dict[int, List[int]] = {}
    for edge_id, dst in enumerate(edge_index[1].tolist()):
        incoming.setdefault(int(dst), []).append(edge_id)
    pairs = []
    angles = []
    for edge_ids in incoming.values():
        if len(edge_ids) < 2:
            continue
        for a in edge_ids:
            for b in edge_ids:
                if a == b:
                    continue
                va = -edge_vec[a]
                vb = -edge_vec[b]
                denom = max(float(np.linalg.norm(va) * np.linalg.norm(vb)), 1e-8)
                cosang = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
                angle = math.acos(cosang)
                pairs.append((a, b))
                angles.append(angle)
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, N_RBF_ANGLE), dtype=np.float32)
    return np.asarray(pairs, dtype=np.int64).T, gaussian_rbf(np.asarray(angles), N_RBF_ANGLE, 0.0, math.pi)


def _global_physics(structures: Sequence, comps: Sequence, flavor: str) -> np.ndarray:
    rows = []
    for comp, structure in zip(comps, structures):
        rows.append(np.concatenate([
            _structure_metadata(structure),
            _perovskite_features(comp),
            _composition_sensor_features(comp),
        ]))
    return np.vstack(rows).astype(np.float32)


def _cache_path(root: Path, task: TaskConfig) -> Path:
    group = task.cache_group if task.is_graph and task.cache_group else task.key
    return root / "_feature_cache" / f"{group}_{task.feature_flavor}.pt"


def load_or_build_features(
    task: TaskConfig,
    structures: Optional[Sequence],
    comps: Sequence,
    root: Path,
    workers: int,
    force_rebuild: bool = False,
) -> Dict:
    cache_file = _cache_path(root, task)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fingerprints = [structure_fingerprint(s) for s in structures] if structures is not None else [str(c) for c in comps]

    if cache_file.exists() and not force_rebuild:
        data = torch.load(cache_file, weights_only=False, map_location="cpu")
        if data.get("fingerprints") == fingerprints:
            print(f"[cache] reuse {cache_file}")
            return data
        print(f"[cache] fingerprint mismatch for {cache_file}; rebuilding")

    print(f"[cache] building {task.key} features -> {cache_file}")
    t0 = time.time()
    print(f"[cache:{task.key}] composition features start ({len(comps)} samples)", flush=True)
    builder = CompositionFeatureBuilder(root / "_feature_cache", task.feature_flavor)
    x_comp = builder.build(comps, structures)
    print(f"[cache:{task.key}] composition features done in {time.time() - t0:.1f}s dim={x_comp.shape[1]}", flush=True)
    data: Dict = {
        "task": task.key,
        "mode": "graph" if task.is_graph else "hybrid",
        "fingerprints": fingerprints,
        "comp_features": torch.tensor(x_comp, dtype=torch.float32),
        "manifest": {
            "task": task.key,
            "dataset": task.dataset_name,
            "feature_flavor": task.feature_flavor,
            "n_samples": len(comps),
            "comp_dim": int(x_comp.shape[1]),
            "built_seconds": None,
        },
    }
    if task.is_graph:
        assert structures is not None, f"{task.key} requires structures"
        t_global = time.time()
        print(f"[cache:{task.key}] global physics start", flush=True)
        data["global_physics"] = torch.tensor(_global_physics(structures, comps, task.feature_flavor), dtype=torch.float32)
        print(f"[cache:{task.key}] global physics done in {time.time() - t_global:.1f}s dim={data['global_physics'].shape[1]}", flush=True)
        t_graph = time.time()
        print(f"[cache:{task.key}] graph features start workers={workers}", flush=True)
        data["graphs"] = build_graphs(structures, workers=workers)
        print(f"[cache:{task.key}] graph features done in {time.time() - t_graph:.1f}s", flush=True)
        data["manifest"]["global_dim"] = int(data["global_physics"].shape[1])
    data["manifest"]["built_seconds"] = round(time.time() - t0, 2)
    print(f"[cache:{task.key}] saving cache", flush=True)
    torch.save(data, cache_file)
    print(f"[cache:{task.key}] saved in {time.time() - t0:.1f}s", flush=True)
    return data


def write_feature_manifest(task_dir: Path, feature_data: Dict) -> None:
    manifest = dict(feature_data.get("manifest", {}))
    manifest["mode"] = feature_data.get("mode")
    manifest["cache_verified"] = True
    with open(task_dir / "feature_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
