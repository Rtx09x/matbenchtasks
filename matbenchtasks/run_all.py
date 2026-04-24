from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

from .configs import DEFAULT_SEED, TASKS, resolve_tasks
from .data import load_matbench_frame
from .features import load_cached_features, load_or_build_features, write_feature_manifest
from .models import build_model, count_parameters
from .train import train_one_task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run remaining Matbench TRIADS tasks sequentially.")
    parser.add_argument("--root", type=str, default="/workspace/matbench_triads_runs")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=str, default="bf16", choices=("bf16", "fp16", "off"))
    parser.add_argument("--memory-profile", type=str, default="a100_80gb")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override every task epoch count; intended for smoke tests.")
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def pick_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[runner] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def preflight(tasks) -> Dict:
    results = {}
    for task in tasks:
        if task.model.kind == "hybrid":
            input_dim = 132 + 240 + 200
            global_dim = 0
        else:
            input_dim = 132 + 240 + 200
            global_dim = 25
        model = build_model(task.model.kind, input_dim, global_dim, task.model, output_dim=1)
        params = count_parameters(model)
        limit = 150_000 if task.model.kind == "hybrid" else 350_000
        ok = params < limit
        results[task.key] = {"params": params, "limit": limit, "ok": ok}
        print(f"[preflight] {task.key:<16} params={params:,} limit={limit:,} ok={ok}")
        if not ok:
            raise RuntimeError(f"{task.key} parameter budget exceeded: {params} >= {limit}")
    return results


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = list(resolve_tasks(args.tasks))
    device = pick_device(args.device)

    print("=" * 80)
    print("TRIADS MatbenchTasks sequential runner")
    print("=" * 80)
    print(f"root:           {root}")
    print(f"tasks:          {[t.key for t in tasks]}")
    print(f"seed:           {args.seed}")
    print(f"device:         {device}")
    print(f"amp:            {args.amp}")
    print(f"memory profile: {args.memory_profile}")
    print(f"workers:        {args.workers}")
    if device.type == "cuda":
        print(f"gpu:            {torch.cuda.get_device_name(0)}")
        print(f"vram:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print("=" * 80)

    preflight_result = preflight(tasks)
    if args.preflight_only:
        (root / "preflight.json").write_text(json.dumps(preflight_result, indent=2), encoding="utf-8")
        return 0

    all_summaries: List[Dict] = []
    start = time.time()
    for idx, task in enumerate(tasks, start=1):
        print("\n" + "#" * 80)
        print(f"[{idx}/{len(tasks)}] {task.key} ({task.dataset_name})")
        print("#" * 80)
        task_dir = root / task.key
        task_dir.mkdir(parents=True, exist_ok=True)

        target_file = task_dir / "targets.npy"
        feature_data = None
        targets = None
        if args.max_samples is None and not args.force_rebuild_features and target_file.exists():
            feature_data = load_cached_features(task, root)
            if feature_data is not None:
                import numpy as np

                targets = np.load(target_file).astype("float32")
                print(f"[{task.key}] prebuilt cache samples={len(targets):,} target={task.target_col} type={task.task_type}", flush=True)
        if feature_data is None or targets is None:
            df, targets, structures, comps = load_matbench_frame(task, max_samples=args.max_samples)
            print(f"[{task.key}] samples={len(targets):,} target={task.target_col} type={task.task_type}")
            feature_data = load_or_build_features(
                task=task,
                structures=structures,
                comps=comps,
                root=root,
                workers=args.workers,
                force_rebuild=args.force_rebuild_features,
            )
        write_feature_manifest(task_dir, feature_data)
        summary = train_one_task(
            task=task,
            feature_data=feature_data,
            targets_np=targets,
            root=root,
            seed=args.seed,
            device=device,
            amp=args.amp,
            fold_limit=args.fold_limit,
            epochs_override=args.epochs,
        )
        all_summaries.append(summary)
        (root / "remaining_matbench_summary.json").write_text(
            json.dumps({
                "status": "partial" if idx < len(tasks) else "complete",
                "completed_tasks": [s["task"] for s in all_summaries],
                "summaries": all_summaries,
                "elapsed_minutes": round((time.time() - start) / 60.0, 2),
            }, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 80)
    print("Complete")
    print("=" * 80)
    for summary in all_summaries:
        combined = summary["combined"]
        print(f"{summary['task']:<16} {combined['metric']}: {combined['mean']:.6f} +/- {combined['std']:.6f}")
    print(f"summary: {root / 'remaining_matbench_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
