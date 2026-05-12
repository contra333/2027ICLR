# AGENTS.md

## Project purpose

This repository supports the ICLR 2027 extension of the user's NeurIPS 2026 paper.
The working thesis is:

Optimizer choice and weight-decay coupling induce different penultimate feature geometry / Neural Collapse regimes, and feature-based uncertainty detectors such as Mahalanobis, DDU/GMM, and kNN succeed or fail depending on detector-relevant geometry components.

## Startup protocol

Read only the smallest context needed for the task.

- General orientation: read `AI_CONTEXT.md`, then `소스/INDEX.md`, then the nearest relevant `AGENTS.md`.
- Literature/source work: read `소스/AGENTS.md`, `소스/INDEX.md`, and the relevant source card or manifest.
- Code work: read `code/AGENTS.md`, the relevant config, and the minimum local README or entrypoint needed.
- Config work: read `configs/AGENTS.md` before changing any experiment setting.
- Result analysis: read `results/AGENTS.md` and the relevant `results/manifests/*.json` before interpreting metrics.
- Reporting/writing: read `reports/AGENTS.md` and separate confirmed evidence from interpretation.
- Multi-server operations: read `ops/AGENTS.md` and the relevant `ops/*.md` workflow note.

Do not read full PDFs, huge extracted text files, or all result files by default. Use indexes, manifests, source cards, and targeted search first.

## Hard rules

- Think before coding, state important assumptions, and keep changes surgical.
- Write the minimum code or documentation that solves the requested problem.
- Do not refactor unrelated areas or reorganize the repository without explicit approval.
- Preserve original PDFs, raw results, checkpoints, server logs, and imported artifacts. Create derived files instead of overwriting originals.
- Distinguish NeurIPS-confirmed results, ICLR hypotheses, external-paper evidence, and new local experiment results.
- Do not claim a test, build, training run, or source read happened unless it actually happened.
- If confidence is low, say so directly and name the missing evidence.

## Research boundaries

- The NeurIPS 2026 paper supports the prior observation that optimizer-induced geometry shifts can decouple calibration/logit reliability from feature-based uncertainty.
- `Optimizer choice matters for the emergence of Neural Collapse` supports the optimizer -> Neural Collapse / representation geometry axis.
- Do not use that Neural Collapse paper alone as direct evidence for Energy, Mahalanobis, kNN, DDU, or downstream OOD detector behavior.
- The ICLR project claim must be supported by this repository's controlled experiments or by separately cited primary sources.

## Server/GPU boundary

- Real training is expected to run on a GPU server, not casually on this local workspace.
- Locally, prepare code, configs, run commands, manifests, import checklists, analysis scripts, and small smoke tests.
- Do not launch expensive local training unless the user explicitly asks for it.
- Every server result copied back into this repo needs a manifest under `results/manifests/`.
- Use Git for code, configs, docs, source indexes, run manifests, and small curated summaries.
- Do not use Git for raw server result folders, checkpoints, feature dumps, large arrays, or temporary logs.
- Expected GPU servers include `101`, `175`, and `138`; each should clone the same private remote when available.
- Use `main` for stable shared state and `exp/<short-name>` branches for experiment preparation.

## Verification and reporting

For multi-step work, report:

1. what changed,
2. what was checked,
3. what remains unverified or risky.

For experiment interpretation, always separate:

- confirmed metric,
- interpretation,
- hypothesis or next action.

For file transformations, verify that the original was preserved and that the derived artifact is readable and traceable.
