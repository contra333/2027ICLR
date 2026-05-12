# Server Run Template

Use this as a checklist before launching an experiment on `101`, `175`, or `138`.

## Run identity

Run ID format:

```text
YYYYMMDD_HHMM_<server>_<dataset>_<model>_<optimizer>_seed<seed>
```

Example:

```text
20260512_2130_101_cifar10_resnet18_adamw_seed0
```

## Required metadata

Record these before launch:

- server name,
- Git branch,
- Git commit hash,
- config path,
- config hash,
- run command,
- output directory,
- expected metrics/log files,
- expected checkpoint policy.

## Command template

Replace placeholders before execution:

```bash
export RUN_ID="YYYYMMDD_HHMM_<server>_<dataset>_<model>_<optimizer>_seed<seed>"
export OUT_ROOT="$HOME/iclr2027_runs"
export OUT_DIR="$OUT_ROOT/$RUN_ID"

mkdir -p "$OUT_DIR"

git rev-parse HEAD > "$OUT_DIR/git_commit.txt"
sha256sum <CONFIG_PATH> > "$OUT_DIR/config.sha256"
cp <CONFIG_PATH> "$OUT_DIR/config_snapshot.yaml"

cat > "$OUT_DIR/command.txt" <<'EOF'
<TRAIN_OR_EVAL_COMMAND>
EOF

<TRAIN_OR_EVAL_COMMAND> 2>&1 | tee "$OUT_DIR/run.log"
```

## Minimum expected output

Each run should produce or preserve:

- `git_commit.txt`
- `config.sha256`
- `config_snapshot.yaml`
- `command.txt`
- `run.log`
- metrics file such as `metrics.json`, `metrics.csv`, or evaluator output

Checkpoints are optional by default and should be copied to local only when needed.

