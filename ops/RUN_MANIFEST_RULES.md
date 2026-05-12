# Run Manifest Rules

## Purpose

Run manifests make server results traceable from local analysis back to the exact server, code snapshot, config, seed, command, and copied files.

## Required fields

Use `results/manifests/RUN_MANIFEST_TEMPLATE.json`.

Required top-level identity:

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

Required experiment axes:

- `dataset`
- `ood_datasets`
- `model`
- `optimizer`
- `seed`
- `detectors`
- `geometry_metrics`

Required file tracking:

- `run_command`
- `metrics_files`
- `log_files`
- `config_snapshot_files`
- `checkpoint_files`
- `checkpoint_copy_status`
- `copied_files`
- `copied_at`
- `verification_status`

## Checkpoint copy status

Use one of:

- `not_expected`
- `exists_on_server_not_copied`
- `copied_best_only`
- `copied_final_only`
- `copied_best_and_final`
- `missing`
- `unknown`

## Interpretation rule

Metrics can be discussed as confirmed only when:

- the manifest JSON parses,
- the metrics file exists locally or is explicitly marked server-only,
- the config and Git snapshot are recorded,
- missing seeds or failed runs are named.

