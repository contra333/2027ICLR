# Task: 101 M1 Smoke Pipeline via Goal Mode

Created: 2026-05-12
Status: completed

## Goal

Run the Milestone 1 smoke validation on server `101` using the Linux shell and the currently active Python environment, after first checking PyTorch, torchvision, CUDA, and required Python dependencies.

The run is complete only if:

- `code/tests/smoke_checks.py` passes the pre-run checks that are feasible on `101`.
- A 2 epoch CIFAR-10 `standard_cifar_resnet18` SGD smoke run finishes.
- Shared cache extraction finishes.
- Post-hoc evaluation writes the expected JSON metric files.
- `code/tests/smoke_checks.py --check run-dir` passes against the run output directory.
- The server output path and remaining import steps are reported.

## Context

Repo path:

```bash
/home/ghjin/2027ICLR/2027ICLR
```

Expected branch:

```bash
exp/m1-smoke-pipeline
```

Config:

```bash
configs/smoke/cifar10_standard-cifar-resnet18_sgd_2ep_seed0.yaml
```

Relevant contracts and workflow notes already reviewed:

- `AGENTS.md`
- `AI_CONTEXT.md`
- `configs/AGENTS.md`
- `code/AGENTS.md`
- `code/IMPLEMENTATION_CONTRACT.md`
- `code/models/ARCHITECTURE_CONTRACT.md`
- `reports/METRIC_DEFINITIONS.md`
- `ops/AGENTS.md`
- `ops/MULTI_SERVER_GIT_WORKFLOW.md`
- `ops/SERVER_CLONE_TO_FIRST_RUN.md`
- `ops/SERVER_RUN_TEMPLATE.md`
- `ops/RESULT_SYNC_GUIDE.md`
- `ops/RUN_MANIFEST_RULES.md`

Experiment axes:

- dataset: `cifar10`
- OOD dataset: `svhn`
- model: `standard_cifar_resnet18`
- architecture style: `standard`
- spectral norm: disabled
- mod: disabled
- optimizer: `sgd`
- seed: `0`
- epochs: `2`
- weight decay policy: `weights_only_no_bias_norm`
- feature layer: `pre_classifier_flattened_pool_output`

## Approval Gates

Do not run the 2 epoch training pipeline until the user approves the plan in the active `/goal` session.

Do not install, upgrade, or remove packages without explicit approval.

Stop and report before training if any of these occur:

- `torch` or `torchvision` import fails.
- CUDA is unavailable when the user expects GPU execution.
- `torch`, `torchvision`, and CUDA versions appear incompatible.
- the repo path or branch is not as expected.
- required dataset download/cache access fails during the data smoke check.

## Linux Workflow

### 1. Connect and verify server context

Run on server `101`:

```bash
hostname
pwd
echo "$SHELL"
whoami
which python
python -V
which pip || true
```

If conda or micromamba exists, inspect but do not switch environments unless the user has already specified the intended environment:

```bash
conda info --envs 2>/dev/null || true
micromamba env list 2>/dev/null || true
```

### 2. Verify repo and branch

```bash
cd /home/ghjin/2027ICLR/2027ICLR
git status --short --branch
git fetch origin
git checkout exp/m1-smoke-pipeline
git pull --ff-only
git rev-parse HEAD
```

Expected: clean or explainable worktree on `exp/m1-smoke-pipeline`.

### 3. Check CUDA and PyTorch stack

```bash
nvidia-smi
python - <<'PY'
import torch
import torchvision

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("cuda device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0", torch.cuda.get_device_name(0))
PY
```

Check basic non-torch requirements:

```bash
python - <<'PY'
import numpy
import yaml

print("numpy", numpy.__version__)
print("yaml ok")
PY
```

### 4. Run pre-run smoke checks

```bash
cd /home/ghjin/2027ICLR/2027ICLR
CONFIG=configs/smoke/cifar10_standard-cifar-resnet18_sgd_2ep_seed0.yaml

python code/tests/smoke_checks.py --config "$CONFIG" --check model
python code/tests/smoke_checks.py --config "$CONFIG" --check data
python code/tests/smoke_checks.py --config "$CONFIG" --check train-step
```

Notes:

- The `data` and `train-step` checks call torchvision datasets with `download=True`.
- Dataset root is `${HOME}/datasets` after config expansion.
- If the server cannot download CIFAR-10 or SVHN and the cache is absent, stop and report.

### 5. Run the 2 epoch smoke pipeline

Use bash syntax, not PowerShell syntax:

```bash
cd /home/ghjin/2027ICLR/2027ICLR

CONFIG=configs/smoke/cifar10_standard-cifar-resnet18_sgd_2ep_seed0.yaml
RUN_ID="$(date +%Y%m%d_%H%M)_101_cifar10_standard-cifar-resnet18_sgd_seed0"
OUT_ROOT="$HOME/iclr2027_runs"
OUT_DIR="$OUT_ROOT/$RUN_ID"

mkdir -p "$OUT_DIR"
git rev-parse HEAD > "$OUT_DIR/git_commit.txt"
sha256sum "$CONFIG" > "$OUT_DIR/config.sha256"
cp "$CONFIG" "$OUT_DIR/config_snapshot.yaml"

{
  echo "CONFIG=$CONFIG"
  echo "RUN_ID=$RUN_ID"
  echo "OUT_DIR=$OUT_DIR"
  echo "python code/train.py --config \"$CONFIG\" --out-dir \"$OUT_DIR\""
  echo "python code/extract_cache.py --config \"$CONFIG\" --run-dir \"$OUT_DIR\""
  echo "python code/eval_posthoc.py --config \"$CONFIG\" --run-dir \"$OUT_DIR\""
  echo "python code/tests/smoke_checks.py --config \"$CONFIG\" --run-dir \"$OUT_DIR\" --check run-dir"
} > "$OUT_DIR/command.txt"

python code/train.py --config "$CONFIG" --out-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/train.log"
python code/extract_cache.py --config "$CONFIG" --run-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/extract_cache.log"
python code/eval_posthoc.py --config "$CONFIG" --run-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/eval_posthoc.log"
python code/tests/smoke_checks.py --config "$CONFIG" --run-dir "$OUT_DIR" --check run-dir 2>&1 | tee "$OUT_DIR/run_dir_check.log"

cat "$OUT_DIR"/train.log "$OUT_DIR"/extract_cache.log "$OUT_DIR"/eval_posthoc.log "$OUT_DIR"/run_dir_check.log > "$OUT_DIR/run.log"
```

### 6. Verify expected output files

```bash
find "$OUT_DIR" -maxdepth 2 -type f | sort

python -m json.tool "$OUT_DIR/metrics_classification.json" >/dev/null
python -m json.tool "$OUT_DIR/metrics_calibration.json" >/dev/null
python -m json.tool "$OUT_DIR/metrics_ood_logit.json" >/dev/null
python -m json.tool "$OUT_DIR/metrics_ood_feature.json" >/dev/null
python -m json.tool "$OUT_DIR/metrics_ood_nc_hybrid.json" >/dev/null
python -m json.tool "$OUT_DIR/metrics_geometry.json" >/dev/null
python -m json.tool "$OUT_DIR/detector_params.json" >/dev/null
python -m json.tool "$OUT_DIR/feature_stats.json" >/dev/null
```

Expected run files:

- `checkpoint_final.pt`
- `train_metrics.jsonl`
- `val_metrics.jsonl`
- `training_summary.json`
- `split_metadata.json`
- `config_snapshot.yaml`
- `config.sha256`
- `git_commit.txt`
- `command.txt`
- `train.log`
- `extract_cache.log`
- `eval_posthoc.log`
- `run_dir_check.log`
- `run.log`

Expected cache files:

- `cache/id_train.pt`
- `cache/id_val.pt`
- `cache/id_test.pt`
- `cache/ood_test_svhn.pt`
- `cache/classifier.pt`
- `cache/cache_metadata.json`

Expected evaluator files:

- `metrics_classification.json`
- `metrics_calibration.json`
- `metrics_ood_logit.json`
- `metrics_ood_feature.json`
- `metrics_ood_nc_hybrid.json`
- `metrics_geometry.json`
- `detector_params.json`
- `feature_stats.json`

## Result Import Checklist

Do not interpret metrics as confirmed local results until import and manifest registration are done.

After the server run, copy at least these files back into:

```text
results/raw/<run_id>/
```

Default copy set:

- `git_commit.txt`
- `config.sha256`
- `config_snapshot.yaml`
- `command.txt`
- `run.log`
- `train.log`
- `extract_cache.log`
- `eval_posthoc.log`
- `run_dir_check.log`
- all `metrics_*.json`
- `detector_params.json`
- `feature_stats.json`
- `training_summary.json`
- `split_metadata.json`

Checkpoint policy:

- `checkpoint_final.pt` is expected on the server.
- Copy it locally only if needed for reproduction, re-evaluation, or inspection.
- If not copied, record `checkpoint_copy_status` as `exists_on_server_not_copied`.

Create:

```text
results/manifests/<run_id>.json
```

Use:

```text
results/manifests/RUN_MANIFEST_TEMPLATE.json
```

Required manifest fields are defined in:

```text
ops/RUN_MANIFEST_RULES.md
```

## Reporting Format

When the `/goal` work finishes or stops, report:

1. server name: `101`
2. repo path
3. branch and commit
4. Python environment path and version
5. torch, torchvision, CUDA, GPU summary
6. config path and config hash
7. run ID and output path
8. exact commands run
9. files produced
10. checks passed or failed
11. remaining risks and next action

## Notes

This task is a pipeline smoke validation, not a paper-result run.

No experiment assumption should be changed unless the user explicitly asks for it.

The Windows/PowerShell snippets in older docs are only sync templates. For this task, use the Linux bash workflow above.

## Completion Record

Completed: 2026-05-13 KST

Confirmed M1A run:

- run ID: `20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0`
- server path: `/home/ghjin/iclr2027_runs/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0`
- manifest: `results/manifests/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0.json`
- checkpoint/cache policy: `checkpoint_final.pt` and cache `.pt` files remain in the run directory and were not copied into Git-tracked paths.

Checks completed:

- `code/tests/smoke_checks.py --config configs/smoke/cifar10_standard-cifar-resnet18_sgd_2ep_seed0.yaml --run-dir /home/ghjin/iclr2027_runs/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0 --check run-dir`
- manifest JSON parse
- metrics JSON parse
- copied-file existence check for the manifest copy set

Files recorded:

- `results/manifests/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0.json`
- raw metrics/log/config snapshots under `results/raw/20260512_2300_101_cifar10_standard-cifar-resnet18_sgd_seed0/` are intentionally ignored by Git.

Remaining risk:

- This is a 2 epoch smoke validation, not paper evidence.
- `metrics_ood_nc_hybrid.json` records that NC/prototype/hybrid detector scores are deferred; geometry diagnostics are emitted with revised metric names.
