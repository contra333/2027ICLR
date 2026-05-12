from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_run_metadata(config_path: str, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_snapshot = out / "config_snapshot.yaml"
    if not config_snapshot.exists():
        shutil.copy2(config_path, config_snapshot)
    git_commit = out / "git_commit.txt"
    if not git_commit.exists():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            commit = "unknown"
        git_commit.write_text(commit + "\n", encoding="utf-8")
    command = out / "command.txt"
    if not command.exists():
        command.write_text(" ".join(sys.argv) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
