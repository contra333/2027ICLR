# Result Sync Guide

## Default policy

Copy metrics and logs by default. Copy checkpoints only when a run must be reproduced, re-evaluated, or inspected.

Default copy set:

- metrics CSV/JSON,
- run logs,
- config snapshot,
- config hash,
- Git commit file,
- command file,
- any existing server-side manifest.

Optional copy set:

- best checkpoint,
- final checkpoint,
- feature dumps,
- logits/features arrays,
- large diagnostic artifacts.

## Local destination

Use this layout:

```text
results/raw/<run_id>/
results/manifests/<run_id>.json
results/processed/
results/figures/
```

## Copy templates

From Windows PowerShell, using `scp`:

```powershell
$RUN_ID = "YYYYMMDD_HHMM_<server>_<dataset>_<model>_<optimizer>_seed<seed>"
$SERVER = "<user>@<server-host>"
$REMOTE = "~/iclr2027_runs/$RUN_ID"
$LOCAL = "C:\Users\User\Desktop\2027ICLR\results\raw\$RUN_ID"

New-Item -ItemType Directory -Force -Path $LOCAL | Out-Null
scp "$SERVER:$REMOTE/git_commit.txt" $LOCAL
scp "$SERVER:$REMOTE/config.sha256" $LOCAL
scp "$SERVER:$REMOTE/config_snapshot.yaml" $LOCAL
scp "$SERVER:$REMOTE/command.txt" $LOCAL
scp "$SERVER:$REMOTE/run.log" $LOCAL
scp "$SERVER:$REMOTE/metrics.*" $LOCAL
```

Checkpoint copy, only when needed:

```powershell
scp "$SERVER:$REMOTE/checkpoints/best.*" $LOCAL
scp "$SERVER:$REMOTE/checkpoints/final.*" $LOCAL
```

## After copying

1. Create or update `results/manifests/<run_id>.json`.
2. Verify JSON parses.
3. Confirm the manifest points to copied metrics/log files.
4. Interpret metrics only after the manifest exists.

