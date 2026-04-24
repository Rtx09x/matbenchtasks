from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .configs import TASK_ORDER, TASKS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run task-specific TRIADS scripts one by one.")
    parser.add_argument("--root", type=str, default="/workspace/matbench_triads_runs")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", type=str, default="bf16", choices=("bf16", "fp16", "off"))
    parser.add_argument("--memory-profile", type=str, default="a100_80gb")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def resolve_task_keys(spec: str):
    if spec.strip().lower() == "all":
        return list(TASK_ORDER)
    aliases = {}
    for key, cfg in TASKS.items():
        aliases[key] = key
        for alias in cfg.aliases:
            aliases[alias] = key
    keys = []
    for part in spec.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in aliases:
            raise ValueError(f"Unknown task '{name}'. Valid: {', '.join(TASK_ORDER)}, or all")
        keys.append(aliases[name])
    if not keys:
        raise ValueError("No tasks selected")
    return keys


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    task_keys = resolve_task_keys(args.tasks)

    print("=" * 80, flush=True)
    print("TRIADS MatbenchTasks task-script runner", flush=True)
    print("=" * 80, flush=True)
    print(f"root:    {root}", flush=True)
    print(f"tasks:   {task_keys}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print("=" * 80, flush=True)

    start = time.time()
    completed = []
    summaries = []
    for idx, task in enumerate(task_keys, start=1):
        print("\n" + "#" * 80, flush=True)
        print(f"[script {idx}/{len(task_keys)}] {task}", flush=True)
        print("#" * 80, flush=True)
        cmd = [
            sys.executable,
            "-m",
            f"matbenchtasks.tasks.{task}",
            "--root", args.root,
            "--seed", str(args.seed),
            "--device", args.device,
            "--amp", args.amp,
            "--memory-profile", args.memory_profile,
            "--workers", str(args.workers),
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.fold_limit is not None:
            cmd.extend(["--fold-limit", str(args.fold_limit)])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.force_rebuild_features:
            cmd.append("--force-rebuild-features")
        if args.preflight_only:
            cmd.append("--preflight-only")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[runner] task {task} failed with exit code {rc}", flush=True)
            return rc
        completed.append(task)
        summary_path = root / task / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        (root / "task_script_progress.json").write_text(
            json.dumps({
                "status": "partial" if idx < len(task_keys) else "complete",
                "completed_tasks": completed,
                "elapsed_minutes": round((time.time() - start) / 60.0, 2),
            }, indent=2),
            encoding="utf-8",
        )
        (root / "remaining_matbench_summary.json").write_text(
            json.dumps({
                "status": "partial" if idx < len(task_keys) else "complete",
                "completed_tasks": completed,
                "summaries": summaries,
                "elapsed_minutes": round((time.time() - start) / 60.0, 2),
            }, indent=2),
            encoding="utf-8",
        )
    print("\n[runner] all selected task scripts completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
