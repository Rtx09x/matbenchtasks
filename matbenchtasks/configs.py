from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


FOLD_SEED = 18012019
DEFAULT_SEED = 42
TASK_ORDER = (
    "dielectric",
    "gvrh",
    "kvrh",
    "perovskites",
    "mp_e_form",
    "mp_gap",
    "mp_is_metal",
)


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    d_attn: int = 32
    d_hidden: int = 64
    ff_dim: int = 96
    d_graph: int = 56
    heads: int = 4
    min_cycles: int = 4
    max_cycles: int = 8
    dropout: float = 0.1
    epochs: int = 80
    swa_start: int = 60
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    max_steps: int = 16


@dataclass(frozen=True)
class TaskConfig:
    key: str
    dataset_name: str
    target_col: str
    metric: str
    task_type: str
    input_type: str
    model: ModelConfig
    aliases: Tuple[str, ...]
    feature_flavor: str
    cache_group: Optional[str] = None
    target_unit: str = ""
    target_candidates: Tuple[str, ...] = ()
    smaller_is_better: bool = True
    baseline_notes: str = ""

    @property
    def is_classification(self) -> bool:
        return self.task_type == "classification"

    @property
    def is_graph(self) -> bool:
        return self.model.kind == "graph"


TASKS: Dict[str, TaskConfig] = {
    "dielectric": TaskConfig(
        key="dielectric",
        dataset_name="matbench_dielectric",
        target_col="n",
        target_candidates=("n",),
        metric="mae",
        task_type="regression",
        input_type="structure",
        aliases=("matbench_dielectric", "diel"),
        feature_flavor="electronic_hybrid",
        target_unit="unitless",
        model=ModelConfig(
            kind="hybrid",
            d_attn=32,
            d_hidden=64,
            ff_dim=96,
            dropout=0.15,
            max_steps=16,
            epochs=260,
            swa_start=190,
            batch_size=1024,
            lr=1e-3,
            grad_clip=1.0,
        ),
    ),
    "gvrh": TaskConfig(
        key="gvrh",
        dataset_name="matbench_log_gvrh",
        target_col="log10(G_VRH)",
        target_candidates=("log10(G_VRH)", "log10 G_VRH", "log10_g_vrh"),
        metric="mae",
        task_type="regression",
        input_type="structure",
        aliases=("matbench_log_gvrh", "log_gvrh"),
        feature_flavor="elastic_graph",
        cache_group="elastic_moduli",
        target_unit="log10(GPa)",
        model=ModelConfig(
            kind="graph",
            d_graph=56,
            heads=4,
            min_cycles=4,
            max_cycles=12,
            dropout=0.10,
            epochs=200,
            swa_start=150,
            batch_size=160,
        ),
    ),
    "kvrh": TaskConfig(
        key="kvrh",
        dataset_name="matbench_log_kvrh",
        target_col="log10(K_VRH)",
        target_candidates=("log10(K_VRH)", "log10 K_VRH", "log10_k_vrh"),
        metric="mae",
        task_type="regression",
        input_type="structure",
        aliases=("matbench_log_kvrh", "log_kvrh"),
        feature_flavor="elastic_graph",
        cache_group="elastic_moduli",
        target_unit="log10(GPa)",
        model=ModelConfig(
            kind="graph",
            d_graph=56,
            heads=4,
            min_cycles=4,
            max_cycles=12,
            dropout=0.10,
            epochs=200,
            swa_start=150,
            batch_size=160,
        ),
    ),
    "perovskites": TaskConfig(
        key="perovskites",
        dataset_name="matbench_perovskites",
        target_col="e_form",
        target_candidates=("e_form", "formation_energy", "formation energy"),
        metric="mae",
        task_type="regression",
        input_type="structure",
        aliases=("matbench_perovskites", "perov"),
        feature_flavor="perovskite_graph",
        cache_group="perovskites",
        target_unit="eV/unit cell",
        model=ModelConfig(
            kind="graph",
            d_graph=56,
            heads=4,
            min_cycles=4,
            max_cycles=10,
            dropout=0.08,
            epochs=160,
            swa_start=120,
            batch_size=192,
        ),
    ),
    "mp_e_form": TaskConfig(
        key="mp_e_form",
        dataset_name="matbench_mp_e_form",
        target_col="e_form",
        target_candidates=("e_form", "formation_energy", "formation energy"),
        metric="mae",
        task_type="regression",
        input_type="structure",
        aliases=("matbench_mp_e_form", "mp_eform", "mp_e_form"),
        feature_flavor="formation_graph",
        cache_group="mp_structures",
        target_unit="eV/atom",
        model=ModelConfig(
            kind="graph",
            d_graph=56,
            heads=4,
            min_cycles=4,
            max_cycles=8,
            dropout=0.05,
            epochs=80,
            swa_start=60,
            batch_size=192,
        ),
    ),
    "mp_gap": TaskConfig(
        key="mp_gap",
        dataset_name="matbench_mp_gap",
        target_col="gap pbe",
        target_candidates=("gap pbe", "gap_pbe", "band_gap", "gap"),
        metric="mae",
        task_type="regression",
        input_type="structure",
        aliases=("matbench_mp_gap", "mp_gap"),
        feature_flavor="electronic_graph",
        cache_group="mp_structures",
        target_unit="eV",
        model=ModelConfig(
            kind="graph",
            d_graph=56,
            heads=4,
            min_cycles=4,
            max_cycles=8,
            dropout=0.08,
            epochs=90,
            swa_start=65,
            batch_size=192,
        ),
    ),
    "mp_is_metal": TaskConfig(
        key="mp_is_metal",
        dataset_name="matbench_mp_is_metal",
        target_col="is_metal",
        target_candidates=("is_metal", "is metal", "metal"),
        metric="roc_auc",
        task_type="classification",
        input_type="structure",
        aliases=("matbench_mp_is_metal", "mp_is_metal", "mp_metal"),
        feature_flavor="electronic_graph",
        cache_group="mp_structures",
        target_unit="roc_auc",
        smaller_is_better=False,
        model=ModelConfig(
            kind="graph",
            d_graph=56,
            heads=4,
            min_cycles=4,
            max_cycles=8,
            dropout=0.08,
            epochs=60,
            swa_start=45,
            batch_size=192,
        ),
    ),
}


def resolve_tasks(spec: str) -> Sequence[TaskConfig]:
    if spec.strip().lower() == "all":
        return [TASKS[k] for k in TASK_ORDER]
    aliases = {}
    for key, cfg in TASKS.items():
        aliases[key] = key
        for alias in cfg.aliases:
            aliases[alias] = key
    out = []
    for part in spec.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in aliases:
            valid = ", ".join(TASK_ORDER)
            raise ValueError(f"Unknown task '{name}'. Valid tasks: {valid}, or all")
        out.append(TASKS[aliases[name]])
    if not out:
        raise ValueError("No tasks selected")
    return out

