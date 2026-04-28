from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .configs import resolve_tasks
from .data import load_matbench_frame
from .features import load_or_build_features, write_feature_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build TRIADS Matbench feature/graph caches on a CPU pod.")
    parser.add_argument("--root", type=str, default="/workspace/matbench_triads_dataset_cache")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--graph-backend", type=str, default="thread", choices=("thread", "process"))
    parser.add_argument("--hf-repo", type=str, default=None, help="Optional Hugging Face dataset repo, e.g. Rtx09/matbench-triads-cache")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-type", type=str, default="dataset")
    return parser.parse_args(argv)


def upload_to_hf(root: Path, repo_id: str, private: bool, repo_type: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(root),
        path_in_repo=".",
        commit_message="Upload TRIADS Matbench feature caches",
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = list(resolve_tasks(args.tasks))
    workers = max(1, int(args.workers))

    # Composition featurization benefits from process fan-out. Graph building
    # defaults to threads because sending large pymatgen Structure objects into
    # notebook child processes can hang before the first progress update.
    os.environ["MATBENCHTASKS_PROCESS_FEATURES"] = "1"

    print("=" * 80, flush=True)
    print("TRIADS MatbenchTasks CPU dataset builder", flush=True)
    print("=" * 80, flush=True)
    print(f"root:    {root}", flush=True)
    print(f"tasks:   {[t.key for t in tasks]}", flush=True)
    print(f"workers: {workers}", flush=True)
    print(f"graph backend: {args.graph_backend}", flush=True)
    print("=" * 80, flush=True)

    summaries = []
    start = time.time()
    for idx, task in enumerate(tasks, start=1):
        print("\n" + "#" * 80, flush=True)
        print(f"[build {idx}/{len(tasks)}] {task.key} ({task.dataset_name})", flush=True)
        print("#" * 80, flush=True)
        task_dir = root / task.key
        task_dir.mkdir(parents=True, exist_ok=True)

        t_task = time.time()
        df, targets, structures, comps = load_matbench_frame(task, max_samples=args.max_samples)
        print(f"[{task.key}] samples={len(targets):,} target={task.target_col} type={task.task_type}", flush=True)
        feature_data = load_or_build_features(
            task=task,
            structures=structures,
            comps=comps,
            root=root,
            workers=workers,
            force_rebuild=args.force_rebuild_features,
            worker_backend=args.graph_backend,
        )
        write_feature_manifest(task_dir, feature_data)
        np.save(task_dir / "targets.npy", targets.astype(np.float32))
        (task_dir / "dataset_info.json").write_text(
            json.dumps({
                "task": task.key,
                "dataset": task.dataset_name,
                "target": task.target_col,
                "target_unit": task.target_unit,
                "task_type": task.task_type,
                "n_samples": int(len(targets)),
                "feature_cache_ready": True,
            }, indent=2),
            encoding="utf-8",
        )
        summary = {
            "task": task.key,
            "dataset": task.dataset_name,
            "samples": len(targets),
            "mode": feature_data.get("mode"),
            "cache_file": str((root / "_feature_cache").resolve()),
            "feature_manifest": feature_data.get("manifest", {}),
            "elapsed_minutes": round((time.time() - t_task) / 60.0, 2),
        }
        summaries.append(summary)
        (root / "dataset_build_summary.json").write_text(
            json.dumps({
                "status": "partial" if idx < len(tasks) else "complete",
                "completed_tasks": [s["task"] for s in summaries],
                "summaries": summaries,
                "elapsed_minutes": round((time.time() - start) / 60.0, 2),
            }, indent=2),
            encoding="utf-8",
        )

    if args.upload:
        if not args.hf_repo:
            raise ValueError("--upload requires --hf-repo")
        print(f"[hf] uploading {root} -> {args.hf_repo}", flush=True)
        upload_to_hf(root, args.hf_repo, private=args.hf_private, repo_type=args.repo_type)
        print("[hf] upload complete", flush=True)

    print("\n[build] complete", flush=True)
    print(f"summary: {root / 'dataset_build_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
