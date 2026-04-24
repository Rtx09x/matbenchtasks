from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel, SWALR

from .configs import FOLD_SEED, TaskConfig
from .models import build_model, count_parameters


def make_folds(targets: np.ndarray, task: TaskConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(targets))
    if task.is_classification:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
        return [(tr, te) for tr, te in splitter.split(indices, targets)]
    splitter = KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    return [(tr, te) for tr, te in splitter.split(indices)]


def inner_split(targets: np.ndarray, task: TaskConfig, seed: int, val_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    if task.is_classification:
        train, val = [], []
        for cls in sorted(set(targets.astype(int).tolist())):
            members = np.where(targets.astype(int) == cls)[0]
            if len(members) == 0:
                continue
            n_val = max(1, int(len(members) * val_size))
            chosen = rng.choice(members, size=n_val, replace=False)
            val.extend(chosen.tolist())
            train.extend(np.setdiff1d(members, chosen).tolist())
        return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)
    bins = np.percentile(targets, [25, 50, 75])
    labels = np.digitize(targets, bins)
    train, val = [], []
    for b in range(4):
        members = np.where(labels == b)[0]
        if len(members) == 0:
            continue
        n_val = max(1, int(len(members) * val_size))
        chosen = rng.choice(members, size=n_val, replace=False)
        val.extend(chosen.tolist())
        train.extend(np.setdiff1d(members, chosen).tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


class DenseLoader:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, indices: Sequence[int], batch_size: int, shuffle: bool):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = len(self.indices)

    def __len__(self) -> int:
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch = idx[start : start + self.batch_size]
            batch_t = torch.as_tensor(batch, dtype=torch.long, device=self.x.device)
            yield self.x[batch_t], self.y[batch_t]


class GraphLoader:
    def __init__(
        self,
        graphs: Sequence[Dict],
        comp: torch.Tensor,
        global_phys: torch.Tensor,
        targets: torch.Tensor,
        indices: Sequence[int],
        batch_size: int,
        device: torch.device,
        shuffle: bool,
        pin_memory: bool,
    ):
        self.graphs = graphs
        self.comp = comp
        self.global_phys = global_phys
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.pin_memory = pin_memory and device.type == "cuda"
        self.dataset_len = len(self.indices)

    def __len__(self) -> int:
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            batch_t = torch.as_tensor(batch_idx, dtype=torch.long)
            comp = self.comp[batch_t]
            glob = self.global_phys[batch_t]
            target = self.targets[batch_t]
            graph = collate_graphs([self.graphs[int(i)] for i in batch_idx])
            if self.pin_memory:
                comp = comp.pin_memory()
                glob = glob.pin_memory()
                target = target.pin_memory()
                for key, value in graph.items():
                    if isinstance(value, torch.Tensor):
                        graph[key] = value.pin_memory()
            yield (
                comp.to(self.device, non_blocking=True),
                glob.to(self.device, non_blocking=True),
                move_graph_to_device(graph, self.device),
                target.to(self.device, non_blocking=True),
            )


def collate_graphs(graphs: Sequence[Dict]) -> Dict:
    atom_z, atom_feat = [], []
    ei, rbf, vec, phys = [], [], [], []
    triplets, angles = [], []
    n_atoms = []
    atom_offset = 0
    edge_offset = 0
    for graph in graphs:
        na = int(graph["n_atoms"])
        ne = int(graph["n_edges"])
        atom_z.append(graph["atom_z"])
        atom_feat.append(graph["atom_feat"])
        ei.append(graph["ei"] + atom_offset)
        rbf.append(graph["rbf"])
        vec.append(graph["vec"])
        phys.append(graph["phys"])
        if graph["triplets"].numel() > 0:
            triplets.append(graph["triplets"] + edge_offset)
            angles.append(graph["angle_feat"])
        n_atoms.append(na)
        atom_offset += na
        edge_offset += ne
    return {
        "atom_z": torch.cat(atom_z),
        "atom_feat": torch.cat(atom_feat),
        "ei": torch.cat(ei, dim=1),
        "rbf": torch.cat(rbf),
        "vec": torch.cat(vec),
        "phys": torch.cat(phys),
        "triplets": torch.cat(triplets, dim=1) if triplets else torch.zeros(2, 0, dtype=torch.long),
        "angle_feat": torch.cat(angles) if angles else torch.zeros(0, 8, dtype=torch.float32),
        "n_crystals": len(graphs),
        "n_atoms": n_atoms,
    }


def move_graph_to_device(graph: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in graph.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def deep_supervision_regression(preds: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    weights = torch.arange(1, len(preds) + 1, device=target.device, dtype=target.dtype)
    weights = weights / weights.sum()
    losses = torch.stack([F.l1_loss(pred, target) for pred in preds])
    return (weights * losses).sum()


def deep_supervision_bce(preds: Sequence[torch.Tensor], target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    weights = torch.arange(1, len(preds) + 1, device=target.device, dtype=target.dtype)
    weights = weights / weights.sum()
    losses = torch.stack([
        F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        for pred in preds
    ])
    return (weights * losses).sum()


def autocast_context(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp.lower() != "off"
    dtype = torch.bfloat16 if amp.lower() == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def train_one_task(
    task: TaskConfig,
    feature_data: Dict,
    targets_np: np.ndarray,
    root: Path,
    seed: int,
    device: torch.device,
    amp: str,
    fold_limit: int | None,
    epochs_override: int | None,
) -> Dict:
    task_dir = root / task.key
    ckpt_dir = task_dir / "checkpoints"
    pred_dir = task_dir / "predictions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    folds = make_folds(targets_np, task)
    if fold_limit:
        folds = folds[:fold_limit]

    fold_metrics = []
    all_prediction_rows = []
    t0 = time.time()
    for fold_idx, (trainval_idx, test_idx) in enumerate(folds, start=1):
        metric_path = task_dir / f"fold_{fold_idx}_metrics.json"
        pred_path = pred_dir / f"fold_{fold_idx}_predictions.csv"
        ckpt_path = ckpt_dir / f"{task.key}_seed{seed}_fold{fold_idx}.pt"
        if metric_path.exists() and pred_path.exists() and ckpt_path.exists():
            print(f"[{task.key}] fold {fold_idx}: already complete; skipping")
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            fold_metrics.append(metric)
            all_prediction_rows.extend(_read_prediction_rows(pred_path))
            continue

        print(f"[{task.key}] fold {fold_idx}/{len(folds)}")
        train_rel, val_rel = inner_split(targets_np[trainval_idx], task, seed + fold_idx)
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
        metric, rows = _train_one_fold(
            task=task,
            feature_data=feature_data,
            targets_np=targets_np,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            fold_idx=fold_idx,
            ckpt_path=ckpt_path,
            seed=seed,
            device=device,
            amp=amp,
            epochs_override=epochs_override,
        )
        for row in rows:
            row["fold"] = fold_idx
        _write_prediction_rows(pred_path, rows)
        metric_path.write_text(json.dumps(metric, indent=2), encoding="utf-8")
        fold_metrics.append(metric)
        all_prediction_rows.extend(rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined_metric = _combine_metrics(task, fold_metrics)
    summary = {
        "task": task.key,
        "dataset": task.dataset_name,
        "target": task.target_col,
        "metric": task.metric,
        "seed": seed,
        "n_folds": len(folds),
        "model_kind": task.model.kind,
        "model_config": task.model.__dict__,
        "fold_metrics": fold_metrics,
        "combined": combined_metric,
        "runtime_minutes": round((time.time() - t0) / 60.0, 2),
    }
    (task_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
    _write_prediction_rows(task_dir / "fold_predictions.csv", all_prediction_rows)
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _train_one_fold(
    task: TaskConfig,
    feature_data: Dict,
    targets_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_idx: int,
    ckpt_path: Path,
    seed: int,
    device: torch.device,
    amp: str,
    epochs_override: int | None,
) -> Tuple[Dict, List[Dict]]:
    torch.manual_seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + fold_idx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    cfg = task.model
    epochs = epochs_override if epochs_override is not None else cfg.epochs
    swa_start = min(cfg.swa_start, max(1, epochs - 1))
    targets = torch.tensor(targets_np.astype(np.float32), dtype=torch.float32)

    if task.is_graph:
        comp_scaled, glob_scaled = _fit_transform_graph_features(feature_data, train_idx)
        target_tensor, target_info = _prepare_targets(targets, targets_np, train_idx, task)
        train_loader = GraphLoader(feature_data["graphs"], comp_scaled, glob_scaled, target_tensor, train_idx, cfg.batch_size, device, True, True)
        val_loader = GraphLoader(feature_data["graphs"], comp_scaled, glob_scaled, target_tensor, val_idx, cfg.batch_size, device, False, True)
        test_loader = GraphLoader(feature_data["graphs"], comp_scaled, glob_scaled, target_tensor, test_idx, cfg.batch_size, device, False, True)
        input_dim = comp_scaled.shape[1]
        global_dim = glob_scaled.shape[1]
    else:
        x_scaled = _fit_transform_dense_features(feature_data, train_idx)
        target_tensor, target_info = _prepare_targets(targets, targets_np, train_idx, task)
        x_device = x_scaled.to(device)
        y_device = target_tensor.to(device)
        train_loader = DenseLoader(x_device, y_device, train_idx, cfg.batch_size, True)
        val_loader = DenseLoader(x_device, y_device, val_idx, cfg.batch_size, False)
        test_loader = DenseLoader(x_device, y_device, test_idx, cfg.batch_size, False)
        input_dim = x_scaled.shape[1]
        global_dim = 0

    output_dim = 1
    model = build_model(task.model.kind, input_dim, global_dim, cfg, output_dim).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start, eta_min=cfg.lr * 0.05)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg.lr * 0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp.lower() == "fp16"))
    best_score = -float("inf") if task.is_classification else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    pos_weight = _pos_weight(targets_np[train_idx], device) if task.is_classification else torch.tensor(1.0, device=device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                if task.is_graph:
                    comp, glob, graph, y = batch
                    preds = model(comp, glob, graph, deep_supervision=True)
                    loss = _loss_for_task(task, preds, y, pos_weight, model)
                else:
                    x, y = batch
                    preds = model(x, deep_supervision=True)
                    loss = _loss_for_task(task, preds, y, pos_weight, model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(y)
            train_n += len(y)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        val_score = _evaluate_score(model, val_loader, task, target_info, device, amp)
        improved = val_score > best_score if task.is_classification else val_score < best_score
        if improved:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch + 1 == epochs:
            loss_avg = train_loss / max(train_n, 1)
            print(f"  fold {fold_idx} epoch {epoch+1}/{epochs} loss={loss_avg:.5f} val_{task.metric}={val_score:.5f}")

    if epochs > swa_start:
        model.load_state_dict(swa_model.module.state_dict())
    else:
        model.load_state_dict(best_state)
    test_rows, test_metric = _predict_rows(model, test_loader, test_idx, targets_np, task, target_info, device, amp)
    torch.save(
        {
            "model_state": model.state_dict(),
            "task": task.key,
            "fold": fold_idx,
            "seed": seed,
            "metric": test_metric,
            "params": n_params,
            "target_info": target_info,
            "config": task.model.__dict__,
        },
        ckpt_path,
    )
    metric = {
        "fold": fold_idx,
        "metric": task.metric,
        "value": float(test_metric),
        "best_val": float(best_score),
        "params": int(n_params),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    return metric, test_rows


def _loss_for_task(task: TaskConfig, preds: Sequence[torch.Tensor], target: torch.Tensor, pos_weight: torch.Tensor, model) -> torch.Tensor:
    if task.is_classification:
        return deep_supervision_bce(preds, target, pos_weight)
    loss = deep_supervision_regression(preds, target)
    if hasattr(model, "_gate_sparsity"):
        loss = loss + 1e-3 * model._gate_sparsity
    return loss


def _fit_transform_dense_features(feature_data: Dict, train_idx: np.ndarray) -> torch.Tensor:
    x = feature_data["comp_features"].cpu().numpy()
    scaler = StandardScaler().fit(x[train_idx])
    xs = np.nan_to_num(scaler.transform(x), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.tensor(xs, dtype=torch.float32)


def _fit_transform_graph_features(feature_data: Dict, train_idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    comp = feature_data["comp_features"].cpu().numpy()
    glob = feature_data["global_physics"].cpu().numpy()
    sc_comp = StandardScaler().fit(comp[train_idx])
    sc_glob = StandardScaler().fit(glob[train_idx])
    comp_s = np.nan_to_num(sc_comp.transform(comp), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    glob_s = np.nan_to_num(sc_glob.transform(glob), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.tensor(comp_s, dtype=torch.float32), torch.tensor(glob_s, dtype=torch.float32)


def _prepare_targets(targets: torch.Tensor, targets_np: np.ndarray, train_idx: np.ndarray, task: TaskConfig):
    if task.is_classification:
        return targets.float(), {"mean": 0.0, "std": 1.0}
    mean = float(np.mean(targets_np[train_idx]))
    std = float(np.std(targets_np[train_idx]) + 1e-8)
    return ((targets - mean) / std).float(), {"mean": mean, "std": std}


def _pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = float(np.sum(y > 0.5))
    negatives = float(len(y) - positives)
    weight = negatives / max(positives, 1.0)
    return torch.tensor(weight, dtype=torch.float32, device=device)


def _evaluate_score(model, loader, task: TaskConfig, target_info: Dict, device: torch.device, amp: str) -> float:
    rows, metric = _predict_rows(model, loader, None, None, task, target_info, device, amp)
    return float(metric)


def _predict_rows(
    model,
    loader,
    original_indices,
    original_targets,
    task: TaskConfig,
    target_info: Dict,
    device: torch.device,
    amp: str,
) -> Tuple[List[Dict], float]:
    model.eval()
    preds = []
    y_true = []
    with torch.inference_mode():
        for batch in loader:
            with autocast_context(device, amp):
                if task.is_graph:
                    comp, glob, graph, y = batch
                    pred = model(comp, glob, graph)
                else:
                    x, y = batch
                    pred = model(x)
            preds.append(pred.detach().float().cpu())
            y_true.append(y.detach().float().cpu())
    pred_all = torch.cat(preds).view(-1)
    y_all = torch.cat(y_true).view(-1)
    if task.is_classification:
        prob = torch.sigmoid(pred_all).numpy()
        target = y_all.numpy()
        try:
            metric = roc_auc_score(target, prob)
        except Exception:
            metric = 0.5
        values = prob
    else:
        values = pred_all.numpy() * float(target_info["std"]) + float(target_info["mean"])
        target = y_all.numpy() * float(target_info["std"]) + float(target_info["mean"])
        metric = float(np.mean(np.abs(values - target)))

    if original_indices is None:
        return [], float(metric)
    rows = []
    for i, idx in enumerate(original_indices):
        rows.append({
            "sample_index": int(idx),
            "target": float(target[i]),
            "prediction": float(values[i]),
        })
    return rows, float(metric)


def _combine_metrics(task: TaskConfig, fold_metrics: Sequence[Dict]) -> Dict:
    values = [float(m["value"]) for m in fold_metrics]
    return {
        "metric": task.metric,
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
        "values": values,
        "smaller_is_better": task.smaller_is_better,
    }


def _write_prediction_rows(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fold", "sample_index", "target", "prediction"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _read_prediction_rows(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]
