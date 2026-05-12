# Multi-Server Git Workflow

## Operating model

Local workspace:

- Path: `C:\Users\User\Desktop\2027ICLR`
- Role: canonical planning, source context, analysis, reporting, and result registry.

GPU servers:

- Expected servers: `101`, `175`, `138`
- Role: clone the same private Git repo, run experiments with different configs/seeds, and write outputs outside Git tracking.

Git remote:

- Use a private Git remote named `origin`.
- `main` is the stable shared branch.
- Use `exp/<short-name>` branches when preparing or validating experiment batches.
- The actual remote URL is not recorded here yet; add it when the user provides or creates the private repository.

## What goes through Git

Commit these:

- experiment code,
- configs,
- docs and operating instructions,
- source indexes and manifests,
- run manifests,
- small processed summaries,
- reports,
- figure scripts and small final figures.

Do not commit these:

- raw server run directories,
- checkpoints,
- feature dumps,
- large arrays,
- large temporary logs,
- downloaded datasets,
- local virtual environments.

## First-time local setup

If this folder has not been initialized:

```powershell
git init -b main
```

When a private remote exists:

```powershell
git remote add origin <PRIVATE_GIT_URL>
git status --short
git add .
git commit -m "Initialize ICLR 2027 research ops repo"
git push -u origin main
```

## First-time server setup

On each server:

```bash
git clone <PRIVATE_GIT_URL> 2027ICLR
cd 2027ICLR
git checkout main
```

Create a server-local output root outside Git if possible:

```bash
mkdir -p ~/iclr2027_runs
```

If outputs must live under the repo, use ignored paths such as:

```bash
mkdir -p results/raw results/checkpoints results/tmp
```

## Update cycle

Local to servers:

```powershell
git status --short
git add <changed_code_or_config_files>
git commit -m "<message>"
git push origin <branch>
```

On each server:

```bash
cd 2027ICLR
git fetch origin
git checkout <branch>
git pull --ff-only
```

Servers to local:

1. Copy metrics/logs/config snapshots/manifests back to `results/raw/<run_id>/`.
2. Copy checkpoints only when needed.
3. Add or update `results/manifests/<run_id>.json`.
4. Commit only manifests and small summaries/figures.

