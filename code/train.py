from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    epochs = int(cfg.get("train", {}).get("epochs", 2))
    train_metrics_path = out_dir / "train_metrics.jsonl"
    val_metrics_path = out_dir / "val_metrics.jsonl"
    train_metrics_path.write_text("", encoding="utf-8")
    val_metrics_path.write_text("", encoding="utf-8")

    final_val_metrics = {}
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

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": epochs,
        "seed": seed,
        "val_metrics": final_val_metrics,
        "split_metadata": data.split_metadata,
    }
    torch.save(checkpoint, out_dir / "checkpoint_final.pt")
    summary = {
        "device": str(device),
        "epochs": epochs,
        "seed": seed,
        "final_val_metrics": final_val_metrics,
        "checkpoint": "checkpoint_final.pt",
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
