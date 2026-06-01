#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/matbench_triads_dataset_cache}"
REPO="${REPO:-Rtx09x/triadsdataset}"
WORKERS="${WORKERS:-128}"
TASKS="${TASKS:-all}"
CACHE_PROFILE="${CACHE_PROFILE:-full}"
GRAPH_BACKEND="${GRAPH_BACKEND:-thread}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
UPLOAD_MODE="${UPLOAD_MODE:-each}"

sudo apt-get update
sudo apt-get install -y git "${PYTHON_BIN}" "${PYTHON_BIN}-venv" python3-pip tmux htop

mkdir -p /workspace
if [ ! -d /workspace/matbenchtasks/.git ]; then
  git clone https://github.com/Rtx09x/matbenchtasks.git /workspace/matbenchtasks
fi

cd /workspace/matbenchtasks
git pull --ff-only

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements-builder.txt
python -m pip install huggingface_hub

mkdir -p "${ROOT}"
UPLOAD_FLAGS=(--upload)
if [ "${UPLOAD_MODE}" = "each" ]; then
  UPLOAD_FLAGS=(--upload-each)
elif [ "${UPLOAD_MODE}" = "both" ]; then
  UPLOAD_FLAGS=(--upload-each --upload)
fi

python -m matbenchtasks.build_datasets \
  --root "${ROOT}" \
  --tasks "${TASKS}" \
  --workers "${WORKERS}" \
  --graph-backend "${GRAPH_BACKEND}" \
  --cache-profile "${CACHE_PROFILE}" \
  --hf-repo "${REPO}" \
  "${UPLOAD_FLAGS[@]}" 2>&1 | tee /workspace/build_datasets.log
