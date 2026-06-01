import os

REPO = os.environ.get("TRIADS_HF_REPO", "Rtx09x/triadsdataset")
ROOT = os.environ.get("TRIADS_ROOT", "/kaggle/working/triads_runs")

commands = [
    "rm -rf /kaggle/working/matbenchtasks",
    "git clone https://github.com/Rtx09x/matbenchtasks.git /kaggle/working/matbenchtasks",
    "cd /kaggle/working/matbenchtasks && pip install -r requirements.txt",
    f"cd /kaggle/working/matbenchtasks && python -m matbenchtasks.download_datasets --root {ROOT} --hf-repo {REPO}",
    f"cd /kaggle/working/matbenchtasks && python -m matbenchtasks.tasks.dielectric --root {ROOT} --fold-limit 1 --max-samples 512 --epochs 2 --device cuda --amp fp16 --workers 2",
]

for command in commands:
    print(f"\n$ {command}", flush=True)
    code = os.system(command)
    if code != 0:
        raise SystemExit(code)

