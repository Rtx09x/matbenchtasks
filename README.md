# TRIADS MatbenchTasks

One-command TRIADS runner for the remaining Matbench v0.1 tasks.

Target tasks:

- `matbench_dielectric`
- `matbench_log_gvrh`
- `matbench_log_kvrh`
- `matbench_perovskites`
- `matbench_mp_e_form`
- `matbench_mp_gap`
- `matbench_mp_is_metal`

The default run is built for a single RunPod A100 80GB notebook: one seed
(`42`), one small TRIADS config per task, official 5-fold Matbench-style
splits, cached features, and resumable fold-level outputs.

## Notebook Command

The preferred path is the task-script runner. It launches one benchmark script
at a time, so logs are clean and Python memory is released between tasks.

```python
!rm -rf /workspace/matbenchtasks && git clone https://github.com/Rtx09x/matbenchtasks.git /workspace/matbenchtasks && cd /workspace/matbenchtasks && pip install -r requirements.txt && python -m matbenchtasks.run_all_tasks --root /workspace/matbench_triads_runs --tasks all --seed 42 --device cuda --amp bf16 --memory-profile a100_80gb --workers 16
```

## Smoke Test

Run this before spending the full A100 session:

```python
!cd /workspace/matbenchtasks && python -m matbenchtasks.tasks.mp_gap --root /workspace/matbench_triads_smoke --fold-limit 1 --max-samples 256 --epochs 1 --device cuda --amp bf16 --memory-profile a100_80gb --workers 16
```

## Task Scripts

Each task can be run directly:

- `python -m matbenchtasks.tasks.dielectric`
- `python -m matbenchtasks.tasks.gvrh`
- `python -m matbenchtasks.tasks.kvrh`
- `python -m matbenchtasks.tasks.perovskites`
- `python -m matbenchtasks.tasks.mp_e_form`
- `python -m matbenchtasks.tasks.mp_gap`
- `python -m matbenchtasks.tasks.mp_is_metal`

All task scripts accept the same options: `--root`, `--seed`, `--device`,
`--amp`, `--workers`, `--max-samples`, `--fold-limit`, `--epochs`,
`--force-rebuild-features`, and `--preflight-only`.

## Outputs

Each task writes to `<root>/<task>/`:

- `summary.json`
- `fold_metrics.json`
- `fold_predictions.csv`
- `feature_manifest.json`
- `checkpoints/<task>_seed42_fold*.pt`

The full run also writes `<root>/remaining_matbench_summary.json`.

## Compute Discipline

- Feature caches are built once and reused.
- Completed folds are skipped on resume.
- Small dense tasks keep tensors on GPU.
- Graph tasks keep variable-size graph caches in CPU memory and transfer only
  current batches to GPU.
- Shared graph cache groups are reused only after structure fingerprints match.
- Matbench metadata is never recursively serialized.
