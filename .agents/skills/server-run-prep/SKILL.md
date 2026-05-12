---
name: server-run-prep
description: Use when preparing GPU/server experiment commands, configs, transfer checklists, or run plans.
---

# Server Run Prep

## Required behavior

- Do not launch expensive local training unless explicitly requested.
- Treat configs as experiment assumptions.
- Make output paths and expected result files explicit.

## Steps

1. Read `AI_CONTEXT.md`, `code/AGENTS.md`, and `configs/AGENTS.md`.
2. Read `ops/MULTI_SERVER_GIT_WORKFLOW.md` and `ops/SERVER_RUN_TEMPLATE.md`.
3. Identify the experiment goal and whether it is main evidence, diagnostic, ViT extension, or pretrained regime.
4. Verify the config names and key axes: dataset, model, optimizer, seed, weight decay, feature layer, detectors.
5. Produce server-ready commands and expected output paths.
6. Include a result-import checklist for `results/manifests/*.json`.

## Output

Return commands, assumptions, expected files, and smoke-test recommendation.
