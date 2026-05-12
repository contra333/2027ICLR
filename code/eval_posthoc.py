from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from eval.aggregate_metrics import aggregate_ood
from eval.classification import accuracy, ece_15bin, fit_temperature, nll
from eval.feature_detectors import (
    fit_gmm_diag,
    fit_gmm_shrinkage,
    fit_gmm_tied,
    fit_knn,
    fit_mahalanobis,
    l2_normalize,
    score_gmm_diag,
    score_gmm_shrinkage,
    score_gmm_tied,
    score_knn,
    score_mahalanobis,
)
from eval.geometry import compute_geometry, feature_stats_by_split
from eval.logit_detectors import energy_id_score, msp
from train_utils import load_config
from train_utils.run_io import write_json


def _load_cache(cache_dir: Path, name: str) -> dict[str, Any]:
    return torch.load(cache_dir / f"{name}.pt", map_location="cpu")


def _load_all_caches(cache_dir: Path) -> dict[str, dict[str, Any]]:
    caches = {
        "id_train": _load_cache(cache_dir, "id_train"),
        "id_val": _load_cache(cache_dir, "id_val"),
        "id_test": _load_cache(cache_dir, "id_test"),
    }
    for path in sorted(cache_dir.glob("ood_test_*.pt")):
        caches[path.stem] = torch.load(path, map_location="cpu")
    return caches


def _aggregate_for_ood(
    id_scores: torch.Tensor, ood_scores_by_name: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    out = {}
    id_np = id_scores.detach().cpu().numpy()
    for ood_name, ood_scores in ood_scores_by_name.items():
        out[ood_name] = aggregate_ood(id_np, ood_scores.detach().cpu().numpy())
    return out


def _run_logit_detectors(caches: dict[str, dict], detector_params: dict) -> dict:
    scores = {}
    for split, cache in caches.items():
        logits = cache["logits"]
        scores[split] = {
            "msp": msp(logits),
            "energy_id_score": energy_id_score(logits, temperature=1.0),
        }

    detector_params["msp"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "none",
    }
    detector_params["energy_id_score"] = {
        "score_direction": "higher_is_id_like",
        "temperature": 1.0,
        "native_sign_note": "project stores T*logsumexp(z/T), not negative energy",
        "fitting_split": "none",
    }

    result = {}
    for detector in ["msp", "energy_id_score"]:
        result[detector] = _aggregate_for_ood(
            scores["id_test"][detector],
            {
                split.replace("ood_test_", ""): split_scores[detector]
                for split, split_scores in scores.items()
                if split.startswith("ood_test_")
            },
        )
    return result


def _run_feature_detectors(cfg: dict, caches: dict[str, dict], detector_params: dict) -> dict:
    id_train = caches["id_train"]
    id_val = caches["id_val"]
    id_test = caches["id_test"]
    x_train = id_train["features"].float()
    y_train = id_train["labels"].long()
    x_val = id_val["features"].float()
    y_val = id_val["labels"].long()
    x_test = id_test["features"].float()
    ood_features = {
        split.replace("ood_test_", ""): cache["features"].float()
        for split, cache in caches.items()
        if split.startswith("ood_test_")
    }

    eval_cfg = cfg.get("eval", {})
    knn_k = int(eval_cfg.get("knn", {}).get("k", 50))
    alpha_grid = [float(a) for a in eval_cfg.get("gmm_shrinkage", {}).get(
        "alpha_grid", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    )]

    result = {}

    maha_model = fit_mahalanobis(x_train, y_train)
    id_scores = score_mahalanobis(maha_model, x_test)
    ood_scores = {name: score_mahalanobis(maha_model, feats) for name, feats in ood_features.items()}
    result["mahalanobis"] = _aggregate_for_ood(id_scores, ood_scores)
    detector_params["mahalanobis"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "id_train",
        "covariance": "tied",
        "jitter": 1.0e-5,
    }

    x_train_l2 = l2_normalize(x_train)
    x_test_l2 = l2_normalize(x_test)
    ood_l2 = {name: l2_normalize(feats) for name, feats in ood_features.items()}
    maha_l2_model = fit_mahalanobis(x_train_l2, y_train)
    id_scores = score_mahalanobis(maha_l2_model, x_test_l2)
    ood_scores = {name: score_mahalanobis(maha_l2_model, feats) for name, feats in ood_l2.items()}
    result["mahalanobis_l2"] = _aggregate_for_ood(id_scores, ood_scores)
    detector_params["mahalanobis_l2"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "id_train",
        "normalization": "l2_feature_before_fit_and_score",
        "note": "Mahalanobis++-motivated control, not full Mahalanobis++ reproduction",
        "jitter": 1.0e-5,
    }

    knn_model = fit_knn(x_train, k=knn_k)
    id_scores = score_knn(knn_model, x_test)
    ood_scores = {name: score_knn(knn_model, feats) for name, feats in ood_features.items()}
    result["knn"] = _aggregate_for_ood(id_scores, ood_scores)
    detector_params["knn"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "id_train",
        "k": knn_k,
        "score": "negative_kth_nearest_neighbor_distance",
        "self_neighbor_rule": "exclude self-neighbor only when scoring id_train itself",
    }

    gmm_tied_model = fit_gmm_tied(x_train, y_train)
    id_scores = score_gmm_tied(gmm_tied_model, x_test)
    ood_scores = {name: score_gmm_tied(gmm_tied_model, feats) for name, feats in ood_features.items()}
    result["gmm_ddu_tied"] = _aggregate_for_ood(id_scores, ood_scores)
    detector_params["gmm_ddu_tied"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "id_train",
        "covariance": "tied",
        "score": "logsumexp_class_log_density_with_equal_class_prior",
        "jitter": 1.0e-5,
    }

    gmm_diag_model = fit_gmm_diag(x_train, y_train)
    id_scores = score_gmm_diag(gmm_diag_model, x_test)
    ood_scores = {name: score_gmm_diag(gmm_diag_model, feats) for name, feats in ood_features.items()}
    result["gmm_ddu_diag"] = _aggregate_for_ood(id_scores, ood_scores)
    detector_params["gmm_ddu_diag"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "id_train",
        "covariance": "classwise_diagonal",
        "score": "logsumexp_class_log_density_with_equal_class_prior",
        "variance_floor": 1.0e-5,
    }

    gmm_shrink_model = fit_gmm_shrinkage(x_train, y_train, x_val, y_val, alpha_grid=alpha_grid)
    id_scores = score_gmm_shrinkage(gmm_shrink_model, x_test)
    ood_scores = {
        name: score_gmm_shrinkage(gmm_shrink_model, feats)
        for name, feats in ood_features.items()
    }
    result["gmm_ddu_shrinkage"] = _aggregate_for_ood(id_scores, ood_scores)
    detector_params["gmm_ddu_shrinkage"] = {
        "score_direction": "higher_is_id_like",
        "fitting_split": "id_train",
        "selection_split": "id_val",
        "covariance": "classwise_shrinkage",
        "shrinkage_target": "T_c = trace(Sigma_c) / d * I",
        "alpha_grid": alpha_grid,
        "alpha": gmm_shrink_model["alpha"],
        "alpha_selection": gmm_shrink_model["alpha_selection"],
        "score": "logsumexp_class_log_density_with_equal_class_prior",
        "jitter": 1.0e-5,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M1A post-hoc evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    cache_dir = run_dir / "cache"
    caches = _load_all_caches(cache_dir)
    classifier = torch.load(cache_dir / "classifier.pt", map_location="cpu")
    detector_params: dict[str, Any] = {
        "global_ood_convention": {
            "higher_score": "more_id_like",
            "id_label": 1,
            "ood_label": 0,
        },
        "aggregate_metrics": {
            "auroc_tie_handling": "rank_average",
            "aupr_in_tie_handling": "stable_descending_score_order",
            "fpr95_threshold_rule": "ID-score 5th percentile; accept score >= threshold",
        },
    }

    id_val_logits = caches["id_val"]["logits"].float()
    id_val_labels = caches["id_val"]["labels"].long()
    id_test_logits = caches["id_test"]["logits"].float()
    id_test_labels = caches["id_test"]["labels"].long()
    temperature = fit_temperature(id_val_logits, id_val_labels)

    classification = {
        "id_test": {
            "accuracy": accuracy(id_test_logits, id_test_labels),
            "nll": nll(id_test_logits, id_test_labels),
            "num_samples": int(id_test_labels.numel()),
        }
    }
    calibration = {
        "id_test": {
            "ece_15bin": ece_15bin(id_test_logits, id_test_labels),
            "temperature_scaled_ece_15bin": ece_15bin(
                id_test_logits / temperature, id_test_labels
            ),
            "temperature": float(temperature),
            "temperature_fitting_split": "id_val",
        }
    }
    detector_params["temperature_scaled_ece_15bin"] = {
        "temperature": float(temperature),
        "fitting_split": "id_val",
    }

    logit_ood = _run_logit_detectors(caches, detector_params)
    feature_ood = _run_feature_detectors(cfg, caches, detector_params)
    geometry = {
        "id_train": compute_geometry(
            caches["id_train"]["features"].float(),
            caches["id_train"]["labels"].long(),
            classifier["weight"].float(),
        ),
        "metadata": {
            "strict_revised_names": True,
            "legacy_names_not_emitted": ["nc0", "nc3", "nc4", "inter_dist"],
            "feature_layer": cfg["model"].get("feature_layer", "pre_classifier_flattened_pool_output"),
        },
    }
    nc_hybrid = {
        "implemented_detectors": [],
        "note": "M1A emits geometry diagnostics only; NC/prototype/hybrid detector scores are deferred.",
    }

    write_json(run_dir / "metrics_classification.json", classification)
    write_json(run_dir / "metrics_calibration.json", calibration)
    write_json(run_dir / "metrics_ood_logit.json", logit_ood)
    write_json(run_dir / "metrics_ood_feature.json", feature_ood)
    write_json(run_dir / "metrics_ood_nc_hybrid.json", nc_hybrid)
    write_json(run_dir / "metrics_geometry.json", geometry)
    write_json(run_dir / "detector_params.json", detector_params)
    write_json(run_dir / "feature_stats.json", feature_stats_by_split(caches))


if __name__ == "__main__":
    main()
