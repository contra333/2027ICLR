# configs/AGENTS.md

## Role

`configs/` stores experiment settings. A config change is a research assumption change.

## Rules

- Before adding server training configs, read `ops/SERVER_CLONE_TO_FIRST_RUN.md`, `code/IMPLEMENTATION_CONTRACT.md`, and `code/models/ARCHITECTURE_CONTRACT.md`.
- Do not change dataset, architecture, optimizer, seed, weight decay, schedule, rho, feature layer, OOD dataset, detector, or preprocessing unless the task asks for it.
- When adding a config, make the experiment purpose clear in the filename or header.
- Keep matched-protocol and tuned-protocol configs separate.
- Keep main standard-architecture configs separate from DDU-style diagnostic configs.
- For ViT configs, record whether weight decay applies to all params or excludes LayerNorm/bias.

## Required metadata for new configs

Include enough information to recover:

- dataset and OOD datasets,
- model/backbone,
- optimizer and optimizer hyperparameters,
- weight-decay policy,
- seeds,
- training budget,
- feature layer,
- detectors and geometry metrics,
- expected output directory.

## Verification

Before a config is used for server training, check that it can be parsed and that output paths are explicit.
