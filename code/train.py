from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data import build_data_bundle
from models import build_model
from optimizers import build_optimizer
from train_utils import load_config, set_seed
from train_utils.metrics import classification_metrics, finalize_running, update_running
from train_utils.run_io import append_jsonl, ensure_run_metadata, write_json


def _unpack_batch(batch):
    if len(batch) == 3:
        images, labels, indices = batch
        return images, labels, indices
    images, labels = batch
    return images, labels, None


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running: dict[str, float] = {}
    for batch in loader:
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images, return_features=True)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            update_running(
                running,
                classification_metrics(logits.detach(), labels),
                count=labels.numel(),
            )
    return finalize_running(running)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running: dict[str, float] = {}
    for batch in loader:
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images, return_features=True)
        update_running(running, classification_metrics(logits, labels), count=labels.numel())
    return finalize_running(running)


def build_scheduler(optimizer, cfg):
    scheduler_cfg = cfg.get("scheduler", {})
    name = scheduler_cfg.get("name", "none")
    if name in ("none", None):
        return None
    if name != "multistep":
        raise ValueError(f"Milestone 1A supports only scheduler.name=multistep, got {name}")
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(m) for m in scheduler_cfg.get("milestones", [])],
        gamma=float(scheduler_cfg.get("gamma", 0.1)),
    )


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    cfg: dict[str, Any],
    epoch: int,
    seed: int,
    val_metrics: dict[str, float],
    split_metadata: dict[str, Any],
    checkpoint_tag: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg,
        "epoch": int(epoch),
        "seed": int(seed),
        "val_metrics": val_metrics,
        "split_metadata": split_metadata,
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(path),
    }
    torch.save(checkpoint, path)
    return path


def _is_better(value: float, best_value: float | None, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"Unsupported train.best_val_mode={mode!r}; expected 'max' or 'min'")


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 1A smoke trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_metadata(args.config, out_dir)

    seed = int(cfg.get("run", {}).get("seed", 0))
    set_seed(seed)
    data = build_data_bundle(cfg)
    write_json(out_dir / "split_metadata.json", data.split_metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 2))
    save_final = bool(train_cfg.get("save_final", True))
    save_best_val = bool(train_cfg.get("save_best_val", False))
    save_every_epochs = int(train_cfg.get("save_every_epochs", 0) or 0)
    best_val_metric = str(train_cfg.get("best_val_metric", "accuracy"))
    best_val_mode = str(train_cfg.get("best_val_mode", "max"))

    train_metrics_path = out_dir / "train_metrics.jsonl"
    val_metrics_path = out_dir / "val_metrics.jsonl"
    train_metrics_path.write_text("", encoding="utf-8")
    val_metrics_path.write_text("", encoding="utf-8")

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_files: list[str] = []
    best_value: float | None = None
    best_epoch: int | None = None
    final_val_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, data.train_aug_loader, optimizer, device)
        final_val_metrics = evaluate(model, data.id_val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        append_jsonl(
            train_metrics_path,
            {"epoch": epoch, "lr": lr, **train_metrics},
        )
        append_jsonl(
            val_metrics_path,
            {"epoch": epoch, "lr": lr, **final_val_metrics},
        )
        if scheduler is not None:
            scheduler.step()

        if save_every_epochs > 0 and epoch % save_every_epochs == 0:
            path = _save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                seed,
                final_val_metrics,
                data.split_metadata,
                checkpoint_tag=f"epoch_{epoch:04d}",
            )
            checkpoint_files.append(_rel(path, out_dir))

        if save_best_val:
            if best_val_metric not in final_val_metrics:
                raise ValueError(
                    f"train.best_val_metric={best_val_metric!r} missing from validation metrics "
                    f"{sorted(final_val_metrics)}"
                )
            value = float(final_val_metrics[best_val_metric])
            if _is_better(value, best_value, best_val_mode):
                best_value = value
                best_epoch = epoch
                path = _save_checkpoint(
                    checkpoint_dir / "checkpoint_best_val.pt",
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    epoch,
                    seed,
                    final_val_metrics,
                    data.split_metadata,
                    checkpoint_tag="best_val",
                )
                checkpoint_files.append(_rel(path, out_dir))

    final_checkpoint = None
    if save_final:
        path = _save_checkpoint(
            checkpoint_dir / "checkpoint_final.pt",
            model,
            optimizer,
            scheduler,
            cfg,
            epochs,
            seed,
            final_val_metrics,
            data.split_metadata,
            checkpoint_tag="final",
        )
        final_checkpoint = _rel(path, out_dir)
        checkpoint_files.append(final_checkpoint)

    summary = {
        "device": str(device),
        "epochs": epochs,
        "seed": seed,
        "final_val_metrics": final_val_metrics,
        "checkpoint": final_checkpoint,
        "checkpoint_dir": "checkpoints",
        "checkpoint_files": sorted(set(checkpoint_files)),
        "save_every_epochs": save_every_epochs,
        "save_best_val": save_best_val,
        "best_val_metric": best_val_metric if save_best_val else None,
        "best_val_mode": best_val_mode if save_best_val else None,
        "best_val_epoch": best_epoch,
        "best_val_value": best_value,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
