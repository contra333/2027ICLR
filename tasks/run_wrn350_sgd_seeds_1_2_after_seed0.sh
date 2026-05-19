#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ghjin/2027ICLR/2027ICLR"
OUT_ROOT="${HOME}/iclr2027_runs"
HOST_TAG="hypatia"
SEED0_OUT_DIR="${OUT_ROOT}/20260519_1954_hypatia_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed0"
QUEUE_LOG="${OUT_ROOT}/wrn350_sgd_seed1_seed2_queue_$(date +%Y%m%d_%H%M).log"

cd "${REPO_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${QUEUE_LOG}"
}

latest_epoch() {
  local metrics_file="$1"
  if [[ -s "${metrics_file}" ]]; then
    wc -l < "${metrics_file}" | tr -d ' '
  else
    printf '0'
  fi
}

wait_for_seed0() {
  log "Waiting for seed0 to finish: ${SEED0_OUT_DIR}"
  while [[ ! -f "${SEED0_OUT_DIR}/checkpoints/checkpoint_final.pt" || ! -f "${SEED0_OUT_DIR}/training_summary.json" ]]; do
    local epoch
    epoch="$(latest_epoch "${SEED0_OUT_DIR}/train_metrics.jsonl")"
    log "seed0 still running or incomplete; train epochs recorded=${epoch}/350"
    sleep 60
  done
  log "seed0 finished; final checkpoint and training_summary.json detected."
}

run_seed() {
  local seed="$1"
  local config="configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed${seed}.yaml"
  local run_id
  local out_dir

  run_id="$(date +%Y%m%d_%H%M)_${HOST_TAG}_cifar10_standard-wrn-28-10-dropout03_sgd-nesterov_seed${seed}"
  out_dir="${OUT_ROOT}/${run_id}"
  mkdir -p "${out_dir}"

  log "Preparing seed${seed}: config=${config}, out_dir=${out_dir}"
  python code/tests/smoke_checks.py --config "${config}" --check data | tee -a "${QUEUE_LOG}"
  python code/tests/smoke_checks.py --config "${config}" --check model | tee -a "${QUEUE_LOG}"

  git rev-parse HEAD > "${out_dir}/git_commit.txt"
  sha256sum "${config}" > "${out_dir}/config.sha256"
  cp "${config}" "${out_dir}/config_snapshot.yaml"
  printf 'CUDA_VISIBLE_DEVICES=0 python code/train.py --config %s --out-dir %s\n' "${config}" "${out_dir}" > "${out_dir}/command.txt"

  log "Starting seed${seed} training: ${run_id}"
  CUDA_VISIBLE_DEVICES=0 python code/train.py --config "${config}" --out-dir "${out_dir}" 2>&1 | tee "${out_dir}/run.log"
  log "Finished seed${seed} training: ${run_id}"

  if [[ ! -f "${out_dir}/checkpoints/checkpoint_final.pt" ]]; then
    log "ERROR: seed${seed} finished without checkpoint_final.pt"
    return 1
  fi
  if [[ ! -f "${out_dir}/training_summary.json" ]]; then
    log "ERROR: seed${seed} finished without training_summary.json"
    return 1
  fi
  log "seed${seed} output verified: ${out_dir}"
}

log "WRN350 SGD seed1/seed2 queue started."
log "Repo HEAD: $(git rev-parse HEAD)"
log "Seed1 config hash: $(sha256sum configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed1.yaml)"
log "Seed2 config hash: $(sha256sum configs/wrn350/cifar10_standard-wrn-28-10-dropout03_sgd_350ep_seed2.yaml)"

wait_for_seed0
run_seed 1
run_seed 2

log "WRN350 SGD seed1/seed2 queue completed."
