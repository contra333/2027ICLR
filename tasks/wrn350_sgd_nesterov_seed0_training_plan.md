# Task: WRN350 SGD Nesterov Seed0 Training Plan

Created: 2026-05-19
Status: planned

## Goal

`standard_wrn_28_10_dropout03`의 CIFAR-10 350 epoch SGD Nesterov seed0 anchor run을 서버에서 완료한다.

이 task는 아래 네 단계가 모두 끝났을 때 완료로 본다.

1. 학습 전 서버/GPU/데이터 상태 확인
2. WRN350 SGD Nesterov seed0 학습 완료
3. `checkpoint_final.pt` 기준 shared cache 추출 및 post-hoc evaluator 실행
4. `results/manifests/<run_id>.json` 작성에 필요한 metrics/log/config snapshot 확보

## Context

서버:

```text
166.104.224.138
```

Repo path:

```bash
/home/ghjin/2027ICLR/2027ICLR
```

학습 config:

```bash
configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml
```

현재 확정된 주요 실험 축:

```text
ID dataset: CIFAR-10
OOD datasets: CIFAR-100, TinyImageNet, SVHN, MNIST
model: standard_wrn_28_10_dropout03
architecture_style: standard
spectral_norm: false
mod: false
optimizer: SGD
lr: 0.1
momentum: 0.9
nesterov: true
weight_decay: 5.0e-4
weight_decay_policy: weights_only_no_bias_norm
scheduler: MultiStepLR, milestones [150, 250], gamma 0.1
epochs: 350
seed: 0
main checkpoint for analysis: checkpoint_final.pt
```

현재 config hash:

```text
6e0120f8509cc71305217cbf7a8c632d2dea6ef85eb4d2502b09d38c72b75bdd
```

주의:

- `nesterov: true`는 definitive WRN SGD anchor의 실험 가정이다.
- 이후 SGD 계열 WRN 비교 config, 특히 `sam_sgd`, `asam_sgd`, `gsam_sgd`의 base SGD 설정에서도 이 선택을 고정한다.
- 이 run은 main CNN evidence의 첫 anchor이며, smoke metric이나 pipeline metric으로 해석하지 않는다.

## Plan

### 1. 서버 상태 확인

서버에 접속한 뒤 repo로 이동한다.

```bash
cd /home/ghjin/2027ICLR/2027ICLR
```

GPU, 디스크, 기존 run directory를 확인한다.

```bash
hostname
nvidia-smi
df -h "$HOME"
du -sh "$HOME/datasets" "$HOME/iclr2027_runs" 2>/dev/null
```

판단 기준:

- GPU memory가 거의 비어 있는 GPU 1장을 사용한다.
- WRN-28-10 CIFAR-10 단일 run은 1 GPU로 충분하므로, 처음부터 여러 GPU를 묶지 않는다.
- `$HOME/iclr2027_runs`에 충분한 여유가 있어야 한다. checkpoint는 50 epoch마다 저장되므로 run directory가 커질 수 있다.

### 2. Repo/config 상태 확인

서버 실행 전에 local에서 `nesterov: true` 변경이 push되어 있고, 서버가 같은 config를 보고 있어야 한다.

```bash
git checkout main
git pull --ff-only
git status -sb

CONFIG=configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml
sha256sum "$CONFIG"
```

성공 조건:

```text
sha256sum == 6e0120f8509cc71305217cbf7a8c632d2dea6ef85eb4d2502b09d38c72b75bdd
```

만약 hash가 다르면 학습을 시작하지 않는다. 이 경우 서버의 config가 local에서 결정한 `nesterov: true` anchor와 다르다는 뜻이다.

### 3. 데이터 확인 및 다운로드

현재 config는 학습 시작 전 OOD loader까지 구성한다. 따라서 아래 데이터가 필요하다.

```text
CIFAR-10
CIFAR-100
SVHN
MNIST
TinyImageNet
```

기본 root:

```bash
$HOME/datasets
```

확인 명령:

```bash
ls -lah "$HOME/datasets" 2>/dev/null
ls -lah "$HOME/datasets/tiny-imagenet-200/val" 2>/dev/null
```

처리 방침:

- CIFAR-10, CIFAR-100, SVHN, MNIST는 torchvision downloader가 처리할 수 있다.
- TinyImageNet은 현재 코드가 자동 다운로드하지 않는다.
- 따라서 definitive full config는 아래 경로가 존재해야 안정적으로 돈다.

```bash
$HOME/datasets/tiny-imagenet-200/val
```

데이터 smoke check:

```bash
python code/tests/smoke_checks.py --config "$CONFIG" --check data
```

판단 기준:

- 이 check가 통과하면 full config의 dataset registry, train/val split, OOD loader 구성이 가능하다는 뜻이다.
- TinyImageNet 경로 문제로 실패하면, TinyImageNet을 먼저 준비한 뒤 다시 `--check data`를 실행한다.
- OOD dataset을 임시로 빼고 학습만 먼저 돌리는 것은 가능하지만, 이 task의 definitive anchor로는 사용하지 않는다.

### 4. 학습 전 최소 train-step check

데이터 check가 통과하면 full 350 epoch 전에 한 batch forward/backward/update를 확인한다.

```bash
python code/tests/smoke_checks.py --config "$CONFIG" --check train-step
```

성공 조건:

- WRN forward가 성공한다.
- `return_features=True` 경로가 동작한다.
- SGD Nesterov optimizer step이 성공한다.
- dataloader, augmentation, label, loss 계산이 연결된다.

이 check가 실패하면 full training을 시작하지 않는다.

### 5. 학습 run 시작

run id 형식:

```text
YYYYMMDD_HHMM_138_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed0
```

예시:

```bash
CONFIG=configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml
RUN_ID=20260519_2300_138_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed0
OUT_ROOT="$HOME/iclr2027_runs"
OUT_DIR="$OUT_ROOT/$RUN_ID"

mkdir -p "$OUT_DIR"

git rev-parse HEAD > "$OUT_DIR/git_commit.txt"
sha256sum "$CONFIG" > "$OUT_DIR/config.sha256"
cp "$CONFIG" "$OUT_DIR/config_snapshot.yaml"

cat > "$OUT_DIR/command.txt" <<'EOF'
python code/train.py --config configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed0.yaml --out-dir $OUT_DIR
EOF

python code/train.py --config "$CONFIG" --out-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"
```

권장 실행 방식:

- 장시간 run이므로 `tmux` 또는 `screen` 안에서 실행한다.
- run 중에는 같은 GPU에 다른 heavy job을 추가하지 않는다.
- 첫 anchor가 완료되기 전에는 Adam/AdamW/SAM 계열을 동시에 벌리지 않는다.

모니터링:

```bash
tail -f "$OUT_DIR/run.log"
tail -n 5 "$OUT_DIR/train_metrics.jsonl"
tail -n 5 "$OUT_DIR/val_metrics.jsonl"
nvidia-smi
```

중간 checkpoint 정책:

```text
checkpoints/checkpoint_epoch_0050.pt
checkpoints/checkpoint_epoch_0100.pt
checkpoints/checkpoint_epoch_0150.pt
checkpoints/checkpoint_epoch_0200.pt
checkpoints/checkpoint_epoch_0250.pt
checkpoints/checkpoint_epoch_0300.pt
checkpoints/checkpoint_epoch_0350.pt
checkpoints/checkpoint_best_val.pt
checkpoints/checkpoint_final.pt
```

주의:

- main geometry/detector analysis는 `checkpoint_final.pt` 기준으로 먼저 수행한다.
- `checkpoint_best_val.pt`는 appendix 또는 deployment-style 비교용으로 따로 다룬다.

### 6. 학습 완료 후 final checkpoint 평가

학습이 끝나면 같은 run directory에서 shared cache를 추출한다.

```bash
python code/extract_cache.py \
  --config "$CONFIG" \
  --run-dir "$OUT_DIR" \
  --checkpoint-tag final
```

그 다음 post-hoc evaluator를 실행한다.

```bash
python code/eval_posthoc.py \
  --config "$CONFIG" \
  --run-dir "$OUT_DIR" \
  --checkpoint-tag final
```

run directory 검증:

```bash
python code/tests/smoke_checks.py \
  --config "$CONFIG" \
  --run-dir "$OUT_DIR" \
  --check run-dir
```

성공 조건:

- train/val metric jsonl이 존재한다.
- final checkpoint가 존재한다.
- final cache가 존재한다.
- evaluator JSON들이 모두 parse된다.
- geometry metric에서 legacy names `nc0`, `nc3`, `nc4`, `inter_dist`가 나오지 않는다.

### 7. 예상 output files

최소 보존 파일:

```text
$OUT_DIR/git_commit.txt
$OUT_DIR/config.sha256
$OUT_DIR/config_snapshot.yaml
$OUT_DIR/command.txt
$OUT_DIR/run.log
$OUT_DIR/train_metrics.jsonl
$OUT_DIR/val_metrics.jsonl
$OUT_DIR/training_summary.json
$OUT_DIR/split_metadata.json
$OUT_DIR/metrics_classification.json
$OUT_DIR/metrics_calibration.json
$OUT_DIR/metrics_ood_logit.json
$OUT_DIR/metrics_ood_feature.json
$OUT_DIR/metrics_ood_nc_hybrid.json
$OUT_DIR/metrics_geometry.json
$OUT_DIR/detector_params.json
$OUT_DIR/feature_stats.json
```

서버에 남길 large files:

```text
$OUT_DIR/checkpoints/*.pt
$OUT_DIR/cache/final/*.pt
```

기본 정책:

- local repo에는 metrics/log/config snapshot/manifest만 가져온다.
- checkpoint와 cache `.pt`는 필요할 때만 복사한다.
- raw run directory, checkpoint, feature dump는 Git에 넣지 않는다.

### 8. Local import checklist

학습 및 평가 완료 후 local로 복사할 파일:

```text
git_commit.txt
config.sha256
config_snapshot.yaml
command.txt
run.log
train_metrics.jsonl
val_metrics.jsonl
training_summary.json
split_metadata.json
metrics_classification.json
metrics_calibration.json
metrics_ood_logit.json
metrics_ood_feature.json
metrics_ood_nc_hybrid.json
metrics_geometry.json
detector_params.json
feature_stats.json
```

local destination:

```text
results/raw/<run_id>/
results/manifests/<run_id>.json
```

manifest에는 최소한 아래 항목을 기록한다.

```text
run_id
server_name: 138
server_path: /home/ghjin/iclr2027_runs/<run_id>
local_import_path: results/raw/<run_id>
git_branch
git_commit
config_path
config_hash
dataset
ood_datasets
model
optimizer
seed
detectors
geometry_metrics
run_command
metrics_files
log_files
config_snapshot_files
checkpoint_files
checkpoint_copy_status
verification_status
```

해석 규칙:

- manifest가 없으면 metric을 confirmed result로 말하지 않는다.
- metric JSON이 local에 없거나 server-only로 명시되지 않으면 confirmed result로 말하지 않는다.
- 이 seed0 anchor 하나만으로 optimizer 일반 결론을 내리지 않는다.

## Expansion After Success

이 run이 `train -> final cache -> posthoc eval -> run-dir check -> manifest`까지 완료된 뒤에만 다음 optimizer로 확장한다.

권장 순서:

1. `sgd` Nesterov seed0 WRN350 anchor 완료
2. `adam` seed0
3. `adamw` seed0
4. `adam_coupled_decoupled` ratio `0.0`, `0.5`, `1.0` seed0
5. ratio grid `0.25`, `0.75` 추가
6. `sam_sgd`, `asam_sgd`, `gsam_sgd` 추가
7. seed `1`, `2` 확장

확장 원칙:

- optimizer 이외의 dataset/model/schedule/seed policy/OOD/detector 설정은 고정한다.
- SGD-family는 base SGD 설정에서 `momentum: 0.9`, `nesterov: true`, `weight_decay_policy: weights_only_no_bias_norm`을 유지한다.
- OOD AUROC/AUPR/FPR95를 보고 optimizer 또는 detector hyperparameter를 고르지 않는다.

## Verification

학습 전 필수:

```bash
python code/tests/smoke_checks.py --config "$CONFIG" --check data
python code/tests/smoke_checks.py --config "$CONFIG" --check train-step
```

학습 후 필수:

```bash
python code/extract_cache.py --config "$CONFIG" --run-dir "$OUT_DIR" --checkpoint-tag final
python code/eval_posthoc.py --config "$CONFIG" --run-dir "$OUT_DIR" --checkpoint-tag final
python code/tests/smoke_checks.py --config "$CONFIG" --run-dir "$OUT_DIR" --check run-dir
```

완료 판정:

- `training_summary.json`에 `epochs: 350`이 기록되어 있다.
- `checkpoint_final.pt`가 존재한다.
- `metrics_*.json`, `detector_params.json`, `feature_stats.json`이 존재하고 JSON parse가 된다.
- `metrics_geometry.json`는 revised metric names를 사용한다.
- result manifest가 작성되어 local result registry에서 추적 가능하다.

## Notes

Confirmed facts:

- WRN350 SGD anchor config는 `nesterov: true`로 결정되었다.
- TinyImageNet은 현재 코드에서 자동 다운로드하지 않는다.
- full config는 OOD loader까지 구성하므로 데이터 check가 학습 전 gate 역할을 한다.

Interpretation:

- 이 run은 optimizer-induced geometry story의 첫 CNN anchor다.
- 이 run만으로 AdamW/SAM 대비 결론을 내리지 않고, 이후 matched-protocol optimizer 비교의 기준점으로 사용한다.

Open risks:

- TinyImageNet 경로가 없으면 `--check data` 또는 full run 시작 단계에서 실패한다.
- 350 epoch run 시간이 길기 때문에 GPU 점유 상태를 먼저 확인해야 한다.
- 현재 `condition_number_clipped`는 구현되어 있으나 metric contract의 주 설명은 상대적으로 약하므로, 논문에서 주요 해석 지표로 쓸 경우 문서 보강이 필요하다.
