from __future__ import annotations

import argparse
from typing import Optional

from .configs import TASKS
from .run_all import main as run_all_main


def parse_args(default_task: str, argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description=f"Run TRIADS task: {default_task}")
    parser.add_argument("--root", type=str, default="/workspace/matbench_triads_runs")
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


def main(default_task: str, argv: Optional[list[str]] = None) -> int:
    if default_task not in TASKS:
        raise ValueError(f"Unknown default task: {default_task}")
    args = parse_args(default_task, argv)
    forwarded = [
        "--root", args.root,
        "--tasks", default_task,
        "--seed", str(args.seed),
        "--device", args.device,
        "--amp", args.amp,
        "--memory-profile", args.memory_profile,
        "--workers", str(args.workers),
    ]
    if args.max_samples is not None:
        forwarded.extend(["--max-samples", str(args.max_samples)])
    if args.fold_limit is not None:
        forwarded.extend(["--fold-limit", str(args.fold_limit)])
    if args.epochs is not None:
        forwarded.extend(["--epochs", str(args.epochs)])
    if args.force_rebuild_features:
        forwarded.append("--force-rebuild-features")
    if args.preflight_only:
        forwarded.append("--preflight-only")
    return run_all_main(forwarded)

