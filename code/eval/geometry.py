from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as F

from .feature_detectors import class_stats


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _simplex_etf_reference(k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    eye = torch.eye(k, dtype=dtype, device=device)
    centered = eye - torch.ones((k, k), dtype=dtype, device=device) / max(k, 1)
    return centered / torch.norm(centered, p="fro").clamp_min(1.0e-12)


def _normalized_gram(matrix: torch.Tensor) -> torch.Tensor:
    gram = matrix @ matrix.T
    return gram / torch.norm(gram, p="fro").clamp_min(1.0e-12)


def _etf_distance(matrix: torch.Tensor, etf_ref: torch.Tensor) -> torch.Tensor:
    k = matrix.shape[0]
    return torch.norm(_normalized_gram(matrix) - etf_ref, p="fro") / max(k * k, 1)


def _effective_rank(eigenvalues: torch.Tensor) -> torch.Tensor:
    eig = eigenvalues.clamp_min(0.0)
    total = eig.sum()
    if total <= 0:
        return torch.zeros((), dtype=eigenvalues.dtype, device=eigenvalues.device)
    probs = eig / total
    entropy = -(probs * torch.log(probs.clamp_min(1.0e-30))).sum()
    return torch.exp(entropy)


def compute_geometry(
    id_train_features: torch.Tensor,
    id_train_labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    id_eval_features: torch.Tensor | None = None,
    id_eval_logits: torch.Tensor | None = None,
) -> dict[str, Any]:
    features = id_train_features.double()
    labels = id_train_labels.long()
    weight = classifier_weight.double()
    stats = class_stats(features, labels)
    means = stats.means.double()
    classes = stats.classes
    sample_global_mean = features.mean(dim=0)
    class_global_mean = means.mean(dim=0)
    centered_means = means - class_global_mean
    d = features.shape[1]
    k = len(classes)

    within_accum = torch.zeros((d, d), dtype=torch.float64, device=features.device)
    within_trace = torch.zeros((), dtype=torch.float64, device=features.device)
    for idx, cls in enumerate(classes):
        cls_features = features[labels == cls]
        diff = cls_features - means[idx]
        within_accum += diff.T @ diff
        within_trace += diff.pow(2).sum()
    sigma_w = within_accum / max(features.shape[0], 1)
    within_var = within_trace / max(features.shape[0], 1)

    sigma_b = centered_means.T @ centered_means / max(k, 1)
    nc1 = torch.trace(sigma_w @ torch.linalg.pinv(sigma_b)).real / max(k, 1)

    pair_l2 = []
    pair_sq = []
    for i, j in itertools.combinations(range(k), 2):
        diff = means[i] - means[j]
        pair_l2.append(torch.norm(diff, p=2))
        pair_sq.append(diff.pow(2).sum())
    inter_dist_l2 = torch.stack(pair_l2).mean() if pair_l2 else torch.zeros((), dtype=torch.float64)
    inter_dist_sq = torch.stack(pair_sq).mean() if pair_sq else torch.zeros((), dtype=torch.float64)

    ones = torch.ones((weight.shape[0], 1), dtype=weight.dtype, device=weight.device)
    row_sum_norm_sq = torch.norm(weight.T @ ones, p=2).pow(2)
    nc0_width_norm = row_sum_norm_sq / max(weight.shape[1], 1)
    nc0_by_K = row_sum_norm_sq / max(weight.shape[0], 1)

    mean_dirs = F.normalize(centered_means, p=2, dim=1, eps=1.0e-12)
    cos = mean_dirs @ mean_dirs.T
    off_diag = cos[~torch.eye(k, dtype=torch.bool, device=cos.device)]
    nc2_mean_cos = off_diag.mean() if off_diag.numel() else torch.zeros((), dtype=torch.float64)

    etf_ref = _simplex_etf_reference(k, dtype=torch.float64, device=features.device)
    nc2_mean_etf = _etf_distance(centered_means, etf_ref)
    nc2_weight_etf = _etf_distance(weight, etf_ref)
    product = weight @ centered_means.T
    product_norm = product / torch.norm(product, p="fro").clamp_min(1.0e-12)
    nc2_product_etf = torch.norm(product_norm - etf_ref, p="fro") / max(k * k, 1)

    w_norm = weight / torch.norm(weight, p="fro").clamp_min(1.0e-12)
    m_t_norm = centered_means / torch.norm(centered_means, p="fro").clamp_min(1.0e-12)
    nc3_self_duality_raw = torch.norm(w_norm - m_t_norm, p="fro")
    nc3_self_duality = nc3_self_duality_raw / max(k * d, 1)
    nc3_cos_alignment = F.cosine_similarity(weight, centered_means, dim=1, eps=1.0e-12).mean()

    result: dict[str, Any] = {
        "within_var": _scalar(within_var),
        "inter_dist_l2": _scalar(inter_dist_l2),
        "inter_dist_sq": _scalar(inter_dist_sq),
        "nc0_width_norm": _scalar(nc0_width_norm),
        "nc0_by_K": _scalar(nc0_by_K),
        "nc1": _scalar(nc1),
        "nc2_mean_etf": _scalar(nc2_mean_etf),
        "nc2_mean_cos": _scalar(nc2_mean_cos),
        "nc2_weight_etf": _scalar(nc2_weight_etf),
        "nc2_product_etf": _scalar(nc2_product_etf),
        "nc3_self_duality": _scalar(nc3_self_duality),
        "nc3_self_duality_raw": _scalar(nc3_self_duality_raw),
        "nc3_cos_alignment": _scalar(nc3_cos_alignment),
        "global_mean_l2_shift_sample_vs_class_mean": _scalar(torch.norm(sample_global_mean - class_global_mean, p=2)),
    }

    if id_eval_features is not None and id_eval_logits is not None:
        eval_features = id_eval_features.double()
        distances = torch.cdist(eval_features, means)
        ncc_idx = distances.argmin(dim=1)
        ncc_pred = classes[ncc_idx].to(id_eval_logits.device)
        learned_pred = id_eval_logits.argmax(dim=1).long()
        result["nc4_agreement"] = float((learned_pred.cpu() == ncc_pred.cpu()).float().mean().item())

    eigenvalues = torch.linalg.eigvalsh(sigma_w).real.clamp_min(0.0)
    eigenvalues_desc = torch.flip(torch.sort(eigenvalues).values, dims=[0])
    trace = eigenvalues_desc.sum()
    lambda_max = eigenvalues_desc[0] if eigenvalues_desc.numel() else torch.zeros((), dtype=torch.float64)
    positive = eigenvalues_desc[eigenvalues_desc > 1.0e-12]
    condition_number = (positive[0] / positive[-1]) if positive.numel() else torch.zeros((), dtype=torch.float64)
    result.update(
        {
            "anisotropy_lambda1_trace": _scalar(lambda_max / trace.clamp_min(1.0e-30)),
            "effective_rank": _scalar(_effective_rank(eigenvalues_desc)),
            "condition_number_clipped": _scalar(condition_number),
            "covariance_eigenspectrum": [float(v) for v in eigenvalues_desc.detach().cpu().tolist()],
        }
    )
    return result


def filter_geometry_metrics(metrics: dict[str, Any], requested: list[str]) -> dict[str, Any]:
    unknown = sorted(set(requested) - set(GEOMETRY_METRIC_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown geometry metric(s): {unknown}")
    return {name: metrics[name] for name in requested if name in metrics}


def feature_stats_by_split(caches: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for split, cache in caches.items():
        features = cache["features"].double()
        norms = torch.norm(features, p=2, dim=1)
        quantiles = torch.quantile(
            norms, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64, device=features.device)
        )
        out[split] = {
            "num_samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "feature_norm_mean": float(norms.mean().item()),
            "feature_norm_std": float(norms.std(unbiased=False).item()),
            "feature_norm_quantiles": {
                "min": float(quantiles[0].item()),
                "q25": float(quantiles[1].item()),
                "median": float(quantiles[2].item()),
                "q75": float(quantiles[3].item()),
                "max": float(quantiles[4].item()),
            },
        }
    return out


GEOMETRY_METRIC_REGISTRY = {
    "within_var": {},
    "inter_dist_l2": {},
    "inter_dist_sq": {},
    "nc0_width_norm": {},
    "nc0_by_K": {},
    "nc1": {},
    "nc2_mean_etf": {},
    "nc2_mean_cos": {},
    "nc2_weight_etf": {},
    "nc2_product_etf": {},
    "nc3_self_duality": {},
    "nc3_self_duality_raw": {},
    "nc3_cos_alignment": {},
    "nc4_agreement": {},
    "anisotropy_lambda1_trace": {},
    "effective_rank": {},
    "condition_number_clipped": {},
    "covariance_eigenspectrum": {},
    "global_mean_l2_shift_sample_vs_class_mean": {},
}
