# results/AGENTS.md

## Role

`results/` stores outputs copied back from GPU/server experiments and local derived analyses.

## Preservation rules

- Do not overwrite raw server outputs, checkpoints, logs, or metrics files.
- Put copied raw outputs under `results/raw/`.
- Put derived tables or normalized metrics under `results/processed/`.
- Put generated figures under `results/figures/`.
- Put one manifest per imported run or run group under `results/manifests/`.
- Copy metrics/logs/config snapshots by default. Copy checkpoints only when needed for reproduction, re-evaluation, or inspection.

## Required run manifest fields

Every `results/manifests/*.json` should include:

- `run_id`
- `server_name`
- `server_path`
- `local_import_path`
- `git_remote`
- `git_branch`
- `git_commit`
- `code_commit_or_snapshot`
- `config_path`
- `config_hash`
- `dataset`
- `ood_datasets`
- `model`
- `optimizer`
- `seed`
- `detectors`
- `geometry_metrics`
- `run_command`
- `metrics_files`
- `log_files`
- `config_snapshot_files`
- `checkpoint_files`
- `checkpoint_copy_status`
- `copied_files`
- `copied_at`
- `verification_status`

Use `results/manifests/RUN_MANIFEST_TEMPLATE.json` as the starting point.

## Interpretation rules

- A metric is confirmed only if it points to an imported metrics file or verified log.
- Keep raw geometry metrics separate from detector-side preprocessing controls such as PCA, shrinkage, tied covariance, diagonal covariance, and feature L2 normalization.
- Report missing seeds, failed checkpoints, and skipped evaluations explicitly.
- Do not average across runs unless the included run IDs and seeds are stated.
