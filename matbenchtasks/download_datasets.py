from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download prebuilt TRIADS Matbench caches from Hugging Face.")
    parser.add_argument("--root", type=str, default="/workspace/matbench_triads_runs")
    parser.add_argument("--hf-repo", type=str, required=True)
    parser.add_argument("--repo-type", type=str, default="dataset")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from huggingface_hub import snapshot_download

    args = parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[hf] downloading {args.hf_repo} -> {root}", flush=True)
    snapshot_download(
        repo_id=args.hf_repo,
        repo_type=args.repo_type,
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )
    print("[hf] download complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

