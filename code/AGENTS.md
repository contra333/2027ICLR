# code/AGENTS.md

## Role

`code/` contains experiment code or a mirror of server-side code used for the ICLR 2027 experiments.

## Rules

- Before implementing server training or evaluation code, read `code/IMPLEMENTATION_CONTRACT.md`, `code/models/ARCHITECTURE_CONTRACT.md`, and `reports/METRIC_DEFINITIONS.md`.
- Keep code changes minimal and local to the requested task.
- Do not silently change experiment assumptions such as dataset, model, optimizer, seed, weight decay, schedule, rho, or feature layer.
- Preserve existing CLI/config names unless a change is necessary and documented.
- Prefer standard architectures for main ICLR evidence. Treat DDU-specific SN/mod code as diagnostic unless explicitly requested.
- Do not run expensive local GPU training without explicit user instruction.
- For server jobs, prepare commands and configs; record expected outputs and import paths.

## Verification

- Run the smallest relevant smoke test when feasible.
- If full training is too expensive, do not run it; state that it was not run.
- Report exact commands used and whether they were local smoke tests or server-ready commands.

## Output discipline

Any code that writes experiment outputs should make it easy to populate `results/manifests/*.json` with config path, code snapshot, seed, metrics files, and checkpoints.
