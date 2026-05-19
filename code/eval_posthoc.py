from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from eval.aggregate_metrics import aggregate_ood
from eval.classification import accuracy, ece_15bin, fit_temperature, nll
from eval.feature_detectors import (
    FEATURE_DETECTOR_REGISTRY,
    NC_HYBRID_DETECTOR_REGISTRY,
    fit_gmm_diag,
    fit_gmm_full,
    fit_gmm_pca,
    fit_gmm_shrinkage,
    fit_gmm_tied,
    fit_knn,
    fit_mahalanobis,
    fit_ncc,
    fit_vim,
    l2_normalize,
    ncc_accuracy,
    score_gmm_diag,
    score_gmm_full,
    score_gmm_pca,
    score_gmm_shrinkage,
    score_gmm_tied,
    score_knn,
    score_mahalanobis,
    score_ncc_distance,
    score_nc_prototype_cosine,
    score_vim_id_score,
)
from eval.geometry import GEOMETRY_METRIC_REGISTRY, compute_geometry, feature_stats_by_split, filter_geometry_metrics
from eval.logit_detectors import LOGIT_DETECTOR_REGISTRY
from train_utils import load_config
from train_utils.run_io import write_json


DEFAULT_LOGIT_DETECTORS = ["msp", "maxlogit", "energy_id_score", "neg_entropy"]
DEFAULT_FEATURE_DETECTORS = [
    "mahalanobis",
    "mahalanobis_l2",
    "knn",
    "knn_l2",
    "gmm_ddu_tied",
    "gmm_ddu_diag",
    "gmm_ddu_shrinkage",
]
DEFAULT_GEOMETRY_METRICS = [
    "within_var",
    "inter_dist_l2",
    "inter_dist_sq",
    "nc0_width_norm",
    "nc0_by_K",
    "nc1",
    "nc2_mean_etf",
    "nc2_mean_cos",
    "nc2_weight_etf",
    "nc2_product_etf",
    "nc3_self_duality",
    "nc3_self_duality_raw",
    "nc3_cos_alignment",
    "nc4_agreement",
    "anisotropy_lambda1_trace",
    "effective_rank",
    "condition_number_clipped",
    "covariance_eigenspectrum",
]


def _load_cache(cache_dir: Path, name: str) -> dict[str, Any]:
    return torch.load(cache_dir / f"{name}.pt", map_location="cpu")


def _resolve_cache_dir(
    run_dir: Path,
    cache_dir_arg: str | None,
    checkpoint_tag: str | None,
    cache_subdir: str | None,
) -> Path:
    if cache_dir_arg:
        return Path(cache_dir_arg).expanduser()
    base = run_dir / "cache"
    if checkpoint_tag:
        return base / checkpoint_tag
    if cache_subdir:
        return base / cache_subdir
    tagged_final = base / "final"
    if (tagged_final / "id_train.pt").exists():
        return tagged_final
    return base


def _load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "cache_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Cache metadata must be a JSON object: {path}")
    return loaded


def _load_all_caches(cache_dir: Path) -> dict[str, dict[str, Any]]:
    caches = {
        "id_train": _load_cache(cache_dir, "id_train"),
        "id_val": _load_cache(cache_dir, "id_val"),
        "id_test": _load_cache(cache_dir, "id_test"),
    }
    for path in sorted(cache_dir.glob("ood_test_*.pt")):
        caches[path.stem] = torch.load(path, map_location="cpu")
    return caches


def _requested_detectors(cfg: dict, family: str, default: list[str]) -> list[str]:
    detector_cfg = cfg.get("eval", {}).get("detectors", {})
    if family not in detector_cfg:
        return list(default)
    value = detector_cfg.get(family)
    if value is None:
        return []
    return [str(name) for name in value]


def _validate_names(requested: list[str], registry: dict[str, Any], family: str) -> None:
    unknown = sorted(set(requested) - set(registry))
    if unknown:
        raise ValueError(f"Unknown {family} detector/metric name(s): {unknown}")


def _aggregate_for_ood(
    id_scores: torch.Tensor, ood_scores_by_name: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    out = {}
    id_np = id_scores.detach().cpu().numpy()
    for ood_name, ood_scores in ood_scores_by_name.items():
        out[ood_name] = aggregate_ood(id_np, ood_scores.detach().cpu().numpy())
    return out


def _ood_feature_splits(caches: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        split.replace("ood_test_", ""): cache["features"].float()
        for split, cache in caches.items()
        if split.startswith("ood_test_")
    }


def _ood_logit_splits(caches: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        split.replace("ood_test_", ""): cache["logits"].float()
        for split, cache in caches.items()
        if split.startswith("ood_test_")
    }


def _run_logit_detectors(cfg: dict, caches: dict[str, dict], detector_params: dict) -> dict:
    requested = _requested_detectors(cfg, "logit", DEFAULT_LOGIT_DETECTORS)
    _validate_names(requested, LOGIT_DETECTOR_REGISTRY, "logit")
    eval_cfg = cfg.get("eval", {})
    energy_temperature = float(eval_cfg.get("energy", {}).get("temperature", 1.0))

    scores: dict[str, dict[str, torch.Tensor]] = {}
    for split, cache in caches.items():
        logits = cache["logits"].float()
        split_scores = {}
        for detector in requested:
            if detector == "energy_id_score":
                split_scores[detector] = LOGIT_DETECTOR_REGISTRY[detector](
                    logits, temperature=energy_temperature
                )
            else:
                split_scores[detector] = LOGIT_DETECTOR_REGISTRY[detector](logits)
        scores[split] = split_scores

    for detector in requested:
        detector_params[detector] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "none",
        }
    if "energy_id_score" in requested:
        detector_params["energy_id_score"].update(
            {
                "temperature": energy_temperature,
                "native_sign_note": "project stores T*logsumexp(z/T), not negative energy",
            }
        )
    if "neg_entropy" in requested:
        detector_params["neg_entropy"].update(
            {"native_sign_note": "raw entropy is OOD-like; project stores negative entropy"}
        )

    result = {}
    for detector in requested:
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
    requested = _requested_detectors(cfg, "feature", DEFAULT_FEATURE_DETECTORS)
    _validate_names(requested, FEATURE_DETECTOR_REGISTRY, "feature")

    id_train = caches["id_train"]
    id_val = caches["id_val"]
    id_test = caches["id_test"]
    x_train = id_train["features"].float()
    y_train = id_train["labels"].long()
    x_val = id_val["features"].float()
    y_val = id_val["labels"].long()
    x_test = id_test["features"].float()
    ood_features = _ood_feature_splits(caches)

    eval_cfg = cfg.get("eval", {})
    knn_k = int(eval_cfg.get("knn", {}).get("k", 50))
    alpha_grid = [
        float(a)
        for a in eval_cfg.get("gmm_shrinkage", {}).get(
            "alpha_grid", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        )
    ]
    result = {}

    if "mahalanobis" in requested:
        model = fit_mahalanobis(x_train, y_train)
        id_scores = score_mahalanobis(model, x_test)
        ood_scores = {name: score_mahalanobis(model, feats) for name, feats in ood_features.items()}
        result["mahalanobis"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["mahalanobis"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "covariance": "tied",
            "covariance_convention": "pooled within-class scatter divided by N-K",
            "jitter": 1.0e-5,
        }

    x_train_l2 = x_test_l2 = None
    ood_l2 = None
    if "mahalanobis_l2" in requested or "knn_l2" in requested:
        x_train_l2 = l2_normalize(x_train)
        x_test_l2 = l2_normalize(x_test)
        ood_l2 = {name: l2_normalize(feats) for name, feats in ood_features.items()}

    if "mahalanobis_l2" in requested:
        model = fit_mahalanobis(x_train_l2, y_train)
        id_scores = score_mahalanobis(model, x_test_l2)
        ood_scores = {name: score_mahalanobis(model, feats) for name, feats in ood_l2.items()}
        result["mahalanobis_l2"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["mahalanobis_l2"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "normalization": "l2_feature_before_fit_and_score",
            "note": "Mahalanobis++-motivated control, not full Mahalanobis++ reproduction",
            "covariance_convention": "pooled within-class scatter divided by N-K after L2 normalization",
            "jitter": 1.0e-5,
        }

    if "knn" in requested:
        model = fit_knn(x_train, k=knn_k)
        id_scores = score_knn(model, x_test)
        ood_scores = {name: score_knn(model, feats) for name, feats in ood_features.items()}
        result["knn"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["knn"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "k": knn_k,
            "score": "negative_kth_nearest_neighbor_distance",
            "self_neighbor_rule": "exclude self-neighbor only when scoring id_train itself",
        }

    if "knn_l2" in requested:
        model = fit_knn(x_train_l2, k=knn_k)
        id_scores = score_knn(model, x_test_l2)
        ood_scores = {name: score_knn(model, feats) for name, feats in ood_l2.items()}
        result["knn_l2"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["knn_l2"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "normalization": "l2_feature_before_fit_and_score",
            "k": knn_k,
            "score": "negative_kth_nearest_neighbor_distance",
            "self_neighbor_rule": "exclude self-neighbor only when scoring id_train itself",
        }

    if "gmm_ddu_tied" in requested:
        model = fit_gmm_tied(x_train, y_train)
        id_scores = score_gmm_tied(model, x_test)
        ood_scores = {name: score_gmm_tied(model, feats) for name, feats in ood_features.items()}
        result["gmm_ddu_tied"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["gmm_ddu_tied"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "covariance": "tied",
            "covariance_convention": "pooled within-class scatter divided by N-K",
            "class_prior": "equal",
            "score": "logsumexp_class_log_density_with_equal_class_prior",
            "jitter": 1.0e-5,
            "note": "DDU-style GMM diagnostic on standard SN-off feature space, not faithful DDU reproduction",
        }

    if "gmm_ddu_diag" in requested:
        model = fit_gmm_diag(x_train, y_train)
        id_scores = score_gmm_diag(model, x_test)
        ood_scores = {name: score_gmm_diag(model, feats) for name, feats in ood_features.items()}
        result["gmm_ddu_diag"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["gmm_ddu_diag"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "covariance": "classwise_diagonal",
            "covariance_convention": "classwise diagonal variance divided by n_c",
            "class_prior": "equal",
            "score": "logsumexp_class_log_density_with_equal_class_prior",
            "variance_floor": 1.0e-5,
            "note": "DDU-style GMM diagnostic on standard SN-off feature space, not faithful DDU reproduction",
        }

    if "gmm_ddu_shrinkage" in requested:
        model = fit_gmm_shrinkage(x_train, y_train, x_val, y_val, alpha_grid=alpha_grid)
        id_scores = score_gmm_shrinkage(model, x_test)
        ood_scores = {name: score_gmm_shrinkage(model, feats) for name, feats in ood_features.items()}
        result["gmm_ddu_shrinkage"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["gmm_ddu_shrinkage"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "selection_split": "id_val",
            "covariance": "classwise_shrinkage",
            "covariance_convention": "classwise covariance divided by n_c-1 before shrinkage",
            "shrinkage_target": "T_c = trace(Sigma_c) / d * I",
            "alpha_grid": alpha_grid,
            "alpha": model["alpha"],
            "alpha_selection": model["alpha_selection"],
            "class_prior": "equal",
            "score": "logsumexp_class_log_density_with_equal_class_prior",
            "jitter": 1.0e-5,
            "note": "DDU-style shrinkage GMM / covariance-estimation control, not faithful DDU reproduction",
        }

    if "gmm_ddu_full" in requested:
        model = fit_gmm_full(x_train, y_train)
        id_scores = score_gmm_full(model, x_test)
        ood_scores = {name: score_gmm_full(model, feats) for name, feats in ood_features.items()}
        result["gmm_ddu_full"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["gmm_ddu_full"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "covariance": "classwise_full",
            "covariance_convention": "classwise covariance divided by n_c-1",
            "class_prior": "equal",
            "score": "logsumexp_class_log_density_with_equal_class_prior",
            "jitter": 1.0e-5,
            "diagnostic_only": True,
            "instability_note": "full covariance may be singular/ill-conditioned in high dimensions",
        }

    if "gmm_ddu_pca" in requested:
        pca_cfg = eval_cfg.get("gmm_pca", {})
        dim = int(pca_cfg.get("dim", 128))
        covariance = str(pca_cfg.get("covariance", "shrinkage"))
        model = fit_gmm_pca(x_train, y_train, x_val, y_val, dim, covariance, alpha_grid)
        id_scores = score_gmm_pca(model, x_test)
        ood_scores = {name: score_gmm_pca(model, feats) for name, feats in ood_features.items()}
        result["gmm_ddu_pca"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["gmm_ddu_pca"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "pca_fit_split": "id_train",
            "pca_requested_dim": dim,
            "pca_effective_dim": model["pca"]["effective_dim"],
            "select_dim_by": str(pca_cfg.get("select_dim_by", "fixed")),
            "covariance": covariance,
            "covariance_convention": "inherited from selected PCA-space GMM covariance",
            "class_prior": "equal",
            "diagnostic_only": True,
            "note": "DDU-style PCA GMM diagnostic on standard SN-off feature space, not faithful DDU reproduction",
        }
        if covariance == "shrinkage":
            detector_params["gmm_ddu_pca"].update(
                {
                    "alpha": model["gmm"]["alpha"],
                    "alpha_selection": model["gmm"]["alpha_selection"],
                    "alpha_grid": alpha_grid,
                }
            )

    return result


def _run_nc_hybrid_detectors(cfg: dict, caches: dict[str, dict], detector_params: dict) -> dict:
    requested = _requested_detectors(cfg, "nc_hybrid", [])
    _validate_names(requested, NC_HYBRID_DETECTOR_REGISTRY, "nc_hybrid")
    if not requested:
        return {
            "implemented_detectors": [],
            "note": "NC/prototype/hybrid detector scores were not requested in eval.detectors.nc_hybrid.",
        }

    x_train = caches["id_train"]["features"].float()
    y_train = caches["id_train"]["labels"].long()
    train_logits = caches["id_train"]["logits"].float()
    x_test = caches["id_test"]["features"].float()
    y_test = caches["id_test"]["labels"].long()
    test_logits = caches["id_test"]["logits"].float()
    ood_features = _ood_feature_splits(caches)
    ood_logits = _ood_logit_splits(caches)
    ncc_model = fit_ncc(x_train, y_train)
    result: dict[str, Any] = {"implemented_detectors": requested}

    if "ncc_accuracy" in requested:
        result["ncc_accuracy"] = {"id_test": ncc_accuracy(ncc_model, x_test, y_test)}
        detector_params["ncc_accuracy"] = {
            "fitting_split": "id_train",
            "reporting_split": "id_test",
            "score_direction": "higher_is_better",
            "note": "label accuracy of nearest-class-center classifier, not NC4 agreement",
        }

    if "ncc_distance" in requested:
        id_scores = score_ncc_distance(ncc_model, x_test)
        ood_scores = {name: score_ncc_distance(ncc_model, feats) for name, feats in ood_features.items()}
        result["ncc_distance"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["ncc_distance"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "score": "negative_min_euclidean_distance_to_id_train_class_mean",
        }

    if "nc_prototype_cosine" in requested:
        id_scores = score_nc_prototype_cosine(ncc_model, x_test)
        ood_scores = {
            name: score_nc_prototype_cosine(ncc_model, feats) for name, feats in ood_features.items()
        }
        result["nc_prototype_cosine"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["nc_prototype_cosine"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "score": "max_cosine_between_l2_feature_and_l2_class_mean",
        }

    if "vim_id_score" in requested:
        vim_cfg = cfg.get("eval", {}).get("vim", {})
        principal_dim = int(vim_cfg.get("principal_dim", 128))
        model = fit_vim(x_train, train_logits, principal_dim=principal_dim)
        id_scores = score_vim_id_score(model, x_test, test_logits)
        ood_scores = {
            name: score_vim_id_score(model, ood_features[name], ood_logits[name])
            for name in ood_features
        }
        result["vim_id_score"] = _aggregate_for_ood(id_scores, ood_scores)
        detector_params["vim_id_score"] = {
            "score_direction": "higher_is_id_like",
            "fitting_split": "id_train",
            "principal_dim": principal_dim,
            "principal_dim_effective": model["pca"]["effective_dim"],
            "centering": "id_train_feature_mean",
            "alpha": model["alpha"],
            "alpha_fit_split": model["alpha_fit_split"],
            "alpha_rule": model["alpha_rule"],
            "native_direction": "higher_is_ood_like",
            "project_transform": "1 - p_virtual_ood",
            "diagnostic_only": True,
        }
        detector_params["vim_native_ood_score"] = {
            "score_direction": "higher_is_ood_like",
            "aggregate_metric_input": False,
            "note": "native diagnostic recorded in params only; project aggregate uses vim_id_score",
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-subdir", help="Backward-compatible alias for cache/<subdir>")
    parser.add_argument("--checkpoint-tag", help="Cache tag such as final, best_val, or epoch_0050")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    cache_dir = _resolve_cache_dir(run_dir, args.cache_dir, args.checkpoint_tag, args.cache_subdir)
    caches = _load_all_caches(cache_dir)
    cache_metadata = _load_cache_metadata(cache_dir)
    classifier = torch.load(cache_dir / "classifier.pt", map_location="cpu")
    detector_params: dict[str, Any] = {
        "metric_contract_version": "2026-05-19",
        "global_ood_convention": {
            "higher_score": "more_id_like",
            "id_label": 1,
            "ood_label": 0,
        },
        "aggregate_metrics": {
            "auroc_tie_handling": "rank_average",
            "aupr_in_tie_handling": "stable_descending_score_order",
            "fpr95_threshold_rule": "ID-score 5th percentile quantile; accept score >= threshold",
        },
        "cache": {
            "path": str(cache_dir),
            "checkpoint_tag": cache_metadata.get("checkpoint_tag", args.checkpoint_tag),
            "checkpoint": cache_metadata.get("checkpoint"),
            "checkpoint_epoch": cache_metadata.get("checkpoint_epoch"),
            "metadata_present": bool(cache_metadata),
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

    logit_ood = _run_logit_detectors(cfg, caches, detector_params)
    feature_ood = _run_feature_detectors(cfg, caches, detector_params)
    nc_hybrid = _run_nc_hybrid_detectors(cfg, caches, detector_params)

    requested_geometry = _requested_detectors(cfg, "geometry", DEFAULT_GEOMETRY_METRICS)
    _validate_names(requested_geometry, GEOMETRY_METRIC_REGISTRY, "geometry")
    all_geometry = compute_geometry(
        caches["id_train"]["features"].float(),
        caches["id_train"]["labels"].long(),
        classifier["weight"].float(),
        id_eval_features=caches["id_test"]["features"].float(),
        id_eval_logits=caches["id_test"]["logits"].float(),
    )
    geometry = {
        "id_train": filter_geometry_metrics(all_geometry, requested_geometry),
        "metadata": {
            "metric_contract_version": "2026-05-19",
            "checkpoint_tag": cache_metadata.get("checkpoint_tag", args.checkpoint_tag),
            "checkpoint": cache_metadata.get("checkpoint"),
            "strict_revised_names": True,
            "legacy_names_not_emitted": ["nc0", "nc3", "nc4", "inter_dist"],
            "feature_layer": cfg["model"].get("feature_layer", "pre_classifier_flattened_pool_output"),
            "global_mean_convention": "class_mean_average_for_nc_metrics",
            "sample_global_mean_shift_metric": "global_mean_l2_shift_sample_vs_class_mean",
            "inter_dist_pair_rule": "off_diagonal_class_pairs_only",
            "nc3_self_duality_output": "paper_normalized_raw_frobenius_distance_divided_by_K_times_d",
            "nc3_self_duality_raw_output": "raw_frobenius_distance_between_frobenius_normalized_W_and_centered_M_transpose",
            "classifier_bias_ignored_for_geometry": True,
            "covariance_eigenspectrum": "sorted_descending_eigenvalues_of_id_train_within_class_covariance",
            "covariance_convention": "Sigma_W uses ID train within-class scatter divided by N",
        },
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
