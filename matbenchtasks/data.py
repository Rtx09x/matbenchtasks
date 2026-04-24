from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .configs import TaskConfig


def load_matbench_frame(task: TaskConfig, max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, Optional[Sequence], Sequence]:
    from matminer.datasets import load_dataset

    df = load_dataset(task.dataset_name)
    target_col = find_column(df, (task.target_col,) + task.target_candidates)
    targets = df[target_col].to_numpy()
    if task.is_classification:
        targets = targets.astype(np.float32)
    else:
        targets = targets.astype(np.float32)

    structures = None
    if "structure" in df.columns:
        structures = df["structure"].tolist()
        comps = [s.composition for s in structures]
    else:
        comp_col = find_column(df, ("composition", "formula"))
        comps = [_composition_from_string(x) for x in df[comp_col].tolist()]

    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].reset_index(drop=True)
        targets = targets[:max_samples]
        comps = comps[:max_samples]
        if structures is not None:
            structures = structures[:max_samples]
    return df, targets, structures, comps


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = str(candidate).strip().lower().replace("_", " ")
        for col in df.columns:
            if str(col).strip().lower().replace("_", " ") == key:
                return col
    raise KeyError(f"Could not find any of {candidates} in columns: {df.columns.tolist()}")


def _composition_from_string(value):
    from pymatgen.core import Composition

    return Composition(str(value))

