from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from .feature_detectors import class_stats


def compute_geometry(
    id_train_features: torch.Tensor,
    id_train_labels: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> dict[str, float]:
    features = id_train_features.double()
    labels = id_train_labels.long()
    weight = classifier_weight.double()
    stats = class_stats(features, labels)
    means = stats.means.double()
    classes = stats.classes
    global_mean = features.mean(dim=0)
    centered_means = means - global_mean
    d = features.shape[1]
    k = len(classes)

    within_accum = torch.zeros((d, d), dtype=torch.float64)
    within_trace = torch.zeros((), dtype=torch.float64)
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
    inter_dist_l2 = torch.stack(pair_l2).mean() if pair_l2 else torch.zeros(())
    inter_dist_sq = torch.stack(pair_sq).mean() if pair_sq else torch.zeros(())

    ones = torch.ones((weight.shape[0], 1), dtype=weight.dtype)
    nc0_width_norm = torch.norm(weight.T @ ones, p=2).pow(2) / max(weight.shape[1], 1)

    mean_dirs = F.normalize(centered_means, p=2, dim=1, eps=1.0e-12)
    cos = mean_dirs @ mean_dirs.T
    off_diag = cos[~torch.eye(k, dtype=torch.bool)]
    nc2_mean_cos = off_diag.mean() if off_diag.numel() else torch.zeros(())

    w_norm = weight / torch.norm(weight, p="fro").clamp_min(1.0e-12)
    m_norm = centered_means / torch.norm(centered_means, p="fro").clamp_min(1.0e-12)
    nc3_self_duality = torch.norm(w_norm - m_norm, p="fro")

    return {
        "within_var": float(within_var.item()),
        "inter_dist_l2": float(inter_dist_l2.item()),
        "inter_dist_sq": float(inter_dist_sq.item()),
        "nc0_width_norm": float(nc0_width_norm.item()),
        "nc1": float(nc1.item()),
        "nc2_mean_cos": float(nc2_mean_cos.item()),
        "nc3_self_duality": float(nc3_self_duality.item()),
    }


def feature_stats_by_split(caches: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for split, cache in caches.items():
        features = cache["features"].double()
        norms = torch.norm(features, p=2, dim=1)
        quantiles = torch.quantile(norms, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
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
