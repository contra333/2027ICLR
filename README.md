# 2027ICLR

This repository is the local operating hub for developing the user's NeurIPS 2026 submission into an ICLR 2027 project.

## Project focus

The project studies optimizer-induced penultimate feature geometry regimes and their effect on feature-based uncertainty / OOD detectors.

Core question:

How do optimizer choice and weight-decay coupling change Neural Collapse-like geometry, and when does that help, hurt, or mask Mahalanobis, DDU/GMM, kNN, and related feature-based detectors?

## Repository map

- `AGENTS.md`: operating rules for Codex and other AI agents.
- `AI_CONTEXT.md`: short hot-cache of current thesis, state, risks, and next actions.
- `소스/`: source literature, prior-paper context, manifests, and AI-readable derived artifacts.
- `code/`: experiment code or server code mirror.
- `configs/`: experiment configs.
- `results/`: raw/processed server results and result manifests.
- `reports/`: experiment reports, professor-facing summaries, and paper interpretation drafts.
- `tasks/`: one-off task tickets.
- `ops/`: multi-server Git, server run, and result-sync workflow notes.
- `.agents/skills/`: reusable AI workflows for this project.

## Operating model

Real training is expected to run on GPU servers such as `101`, `175`, and `138`. This local repo is for planning, source grounding, config/code preparation, result import, analysis, and reporting.

When results are copied back, preserve raw files and add a manifest under `results/manifests/` so every metric can be traced back to the server path, config, code snapshot, seed, and imported files.

Use Git for code, configs, docs, manifests, and small curated summaries. Keep raw server outputs, checkpoints, feature dumps, and large arrays out of Git.

## First files to read

For a new AI session, start with:

1. `AGENTS.md`
2. `AI_CONTEXT.md`
3. `소스/INDEX.md`

Then read only the nearest relevant folder-specific `AGENTS.md`.
