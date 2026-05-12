---
name: result-import
description: Use when copying, registering, or validating server experiment results in this repo.
---

# Result Import

## Required behavior

- Preserve raw server outputs.
- Create a result manifest before interpreting metrics.
- Do not average or summarize runs until run IDs and seeds are known.

## Steps

1. Copy raw files into `results/raw/` or confirm they already exist.
2. Create a manifest under `results/manifests/` using `RUN_MANIFEST_TEMPLATE.json`.
3. Follow `ops/RESULT_SYNC_GUIDE.md` for the default metrics/log copy set and optional checkpoint copy set.
4. Verify metrics files are readable.
5. Verify checkpoints/logs exist when expected.
6. Only then create processed summaries or figures.

## Output

Report manifest path, copied files, verification status, and missing artifacts.
