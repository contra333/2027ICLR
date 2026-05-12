# ops/AGENTS.md

## Role

`ops/` contains operational instructions for multi-server Git use, GPU server runs, and local result import.

## Rules

- Before first server-side implementation, read `ops/SERVER_CLONE_TO_FIRST_RUN.md`.
- Prefer documenting commands before running them on shared servers.
- Treat Git as the channel for code, configs, docs, small summaries, and manifests.
- Do not put checkpoints, feature dumps, large logs, or raw run directories in Git.
- Every server run should be recoverable from a run ID, Git commit or snapshot, config, command, server name, and result manifest.
- If a server command is only a template, mark placeholders clearly instead of pretending it was executed.

## Reporting

When reporting server operations, include:

- server name,
- repo path,
- branch and commit,
- config path,
- run ID,
- output path,
- what was copied back locally.
