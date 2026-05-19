from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


JITTER = 1.0e-5


@dataclass
class ClassStats:
    classes: torch.Tensor
    means: torch.Tensor
    counts: torch.Tensor


def class_stats(features: torch.Tensor, labels: torch.Tensor) -> ClassStats:
    classes = torch.unique(labels).sort().values.long()
    means = []
    counts = []
    for cls in classes:
        mask = labels == cls
        cls_features = features[mask]
        means.append(cls_features.mean(dim=0))
        counts.append(torch.tensor(cls_features.shape[0], dtype=torch.long, device=features.device))
    return ClassStats(classes=classes, means=torch.stack(means, dim=0), counts=torch.stack(counts))


def tied_covariance(features: torch.Tensor, labels: torch.Tensor, stats: ClassStats) -> torch.Tensor:
    centered = []
    for idx, cls in enumerate(stats.classes):
        cls_features = features[labels == cls]
        centered.append(cls_features - stats.means[idx])
    diffs = torch.cat(centered, dim=0)
    denom = max(int(features.shape[0] - len(stats.classes)), 1)
    cov = diffs.T @ diffs / denom
    return cov + JITTER * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)


def _maha_scores(features: torch.Tensor, means: torch.Tensor, cov: torch.Tensor, batch_size: int = 512):
    scores = []
    precision = torch.linalg.pinv(cov.double()).to(features.dtype)
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        diff = batch[:, None, :] - means[None, :, :]
        solved = torch.einsum("nkd,df->nkf", diff, precision)
        dist = (diff * solved).sum(dim=2)
        scores.append(-dist.min(dim=1).values)
    return torch.cat(scores, dim=0)


def fit_mahalanobis(id_train_features: torch.Tensor, id_train_labels: torch.Tensor):
    stats = class_stats(id_train_features, id_train_labels)
    cov = tied_covariance(id_train_features, id_train_labels, stats)
    return {"stats": stats, "covariance": cov}


def score_mahalanobis(model, features: torch.Tensor) -> torch.Tensor:
    return _maha_scores(features, model["stats"].means, model["covariance"])


def fit_knn(id_train_features: torch.Tensor, k: int):
    return {"features": id_train_features.contiguous(), "k": int(k)}


def score_knn(model, features: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    train_features = model["features"]
    k = min(int(model["k"]), train_features.shape[0])
    scores = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        distances = torch.cdist(batch, train_features)
        kth = distances.topk(k, largest=False, dim=1).values[:, -1]
        scores.append(-kth)
    return torch.cat(scores, dim=0)


def _log_prob_full(features: torch.Tensor, means: torch.Tensor, covs: list[torch.Tensor]) -> torch.Tensor:
    out = []
    d = features.shape[1]
    const = d * math.log(2.0 * math.pi)
    eye = torch.eye(d, dtype=torch.float64, device=features.device)
    for mean, cov in zip(means, covs):
        cov = cov.double()
        precision = torch.linalg.pinv(cov)
        sign, logdet = torch.linalg.slogdet(cov)
        if sign.item() <= 0:
            logdet = torch.logdet(cov + JITTER * eye)
        diff = (features.double() - mean.double()).to(torch.float64)
        quad = (diff @ precision * diff).sum(dim=1)
        out.append((-0.5 * (quad + logdet + const)).to(features.dtype))
    return torch.stack(out, dim=1)


def _equal_prior_logsumexp(log_probs: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(log_probs, dim=1) - math.log(log_probs.shape[1])


def fit_gmm_tied(id_train_features: torch.Tensor, id_train_labels: torch.Tensor):
    stats = class_stats(id_train_features, id_train_labels)
    cov = tied_covariance(id_train_features, id_train_labels, stats)
    return {"stats": stats, "covariance": cov}


def score_gmm_tied(model, features: torch.Tensor) -> torch.Tensor:
    stats = model["stats"]
    log_probs = _log_prob_full(features, stats.means, [model["covariance"]] * len(stats.classes))
    return _equal_prior_logsumexp(log_probs)


def fit_gmm_diag(id_train_features: torch.Tensor, id_train_labels: torch.Tensor):
    stats = class_stats(id_train_features, id_train_labels)
    variances = []
    for idx, cls in enumerate(stats.classes):
        cls_features = id_train_features[id_train_labels == cls]
        variances.append(cls_features.var(dim=0, unbiased=False).clamp_min(JITTER))
    return {"stats": stats, "variances": torch.stack(variances, dim=0)}


def score_gmm_diag(model, features: torch.Tensor) -> torch.Tensor:
    stats = model["stats"]
    means = stats.means
    variances = model["variances"]
    diff = features[:, None, :] - means[None, :, :]
    log_probs = -0.5 * (
        (diff.pow(2) / variances[None, :, :]).sum(dim=2)
        + variances.log().sum(dim=1)[None, :]
        + features.shape[1] * math.log(2.0 * math.pi)
    )
    return _equal_prior_logsumexp(log_probs)


def _class_covariances(features: torch.Tensor, labels: torch.Tensor, stats: ClassStats) -> list[torch.Tensor]:
    covs = []
    d = features.shape[1]
    eye = torch.eye(d, dtype=features.dtype, device=features.device)
    for idx, cls in enumerate(stats.classes):
        cls_features = features[labels == cls]
        diff = cls_features - stats.means[idx]
        denom = max(cls_features.shape[0] - 1, 1)
        covs.append(diff.T @ diff / denom + JITTER * eye)
    return covs


def fit_gmm_full(id_train_features: torch.Tensor, id_train_labels: torch.Tensor):
    stats = class_stats(id_train_features, id_train_labels)
    covs = _class_covariances(id_train_features, id_train_labels, stats)
    return {"stats": stats, "covariances": covs}


def score_gmm_full(model, features: torch.Tensor) -> torch.Tensor:
    log_probs = _log_prob_full(features, model["stats"].means, model["covariances"])
    return _equal_prior_logsumexp(log_probs)


def _shrink_cov(cov: torch.Tensor, alpha: float) -> torch.Tensor:
    d = cov.shape[0]
    target = torch.trace(cov) / d * torch.eye(d, dtype=cov.dtype, device=cov.device)
    return (1.0 - alpha) * cov + alpha * target + JITTER * torch.eye(d, dtype=cov.dtype, device=cov.device)


def fit_gmm_shrinkage(
    id_train_features: torch.Tensor,
    id_train_labels: torch.Tensor,
    id_val_features: torch.Tensor,
    id_val_labels: torch.Tensor,
    alpha_grid: list[float],
):
    stats = class_stats(id_train_features, id_train_labels)
    base_covs = _class_covariances(id_train_features, id_train_labels, stats)
    class_to_idx = {int(cls.item()): idx for idx, cls in enumerate(stats.classes)}
    best_alpha = alpha_grid[0]
    best_ll = -float("inf")
    for alpha in alpha_grid:
        covs = [_shrink_cov(cov, alpha) for cov in base_covs]
        log_probs = _log_prob_full(id_val_features, stats.means, covs)
        selected = []
        for row_idx, label in enumerate(id_val_labels):
            selected.append(log_probs[row_idx, class_to_idx[int(label.item())]])
        ll = torch.stack(selected).mean().item()
        if ll > best_ll:
            best_ll = ll
            best_alpha = float(alpha)
    return {
        "stats": stats,
        "covariances": [_shrink_cov(cov, best_alpha) for cov in base_covs],
        "alpha": best_alpha,
        "alpha_selection": "id_val_class_conditional_likelihood",
    }


def score_gmm_shrinkage(model, features: torch.Tensor) -> torch.Tensor:
    log_probs = _log_prob_full(features, model["stats"].means, model["covariances"])
    return _equal_prior_logsumexp(log_probs)


def fit_pca(features: torch.Tensor, dim: int) -> dict[str, Any]:
    mean = features.mean(dim=0)
    centered = (features - mean).double()
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    q = min(int(dim), vh.shape[0], features.shape[1])
    return {
        "mean": mean,
        "components": vh[:q].to(features.dtype),
        "requested_dim": int(dim),
        "effective_dim": int(q),
        "fit_split": "id_train",
    }


def transform_pca(model: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    return (features - model["mean"]) @ model["components"].T


def fit_gmm_pca(
    id_train_features: torch.Tensor,
    id_train_labels: torch.Tensor,
    id_val_features: torch.Tensor,
    id_val_labels: torch.Tensor,
    dim: int,
    covariance: str,
    alpha_grid: list[float],
):
    pca = fit_pca(id_train_features, dim)
    train_pca = transform_pca(pca, id_train_features)
    val_pca = transform_pca(pca, id_val_features)
    if covariance == "full":
        gmm = fit_gmm_full(train_pca, id_train_labels)
    elif covariance == "diag":
        gmm = fit_gmm_diag(train_pca, id_train_labels)
    elif covariance == "tied":
        gmm = fit_gmm_tied(train_pca, id_train_labels)
    elif covariance == "shrinkage":
        gmm = fit_gmm_shrinkage(train_pca, id_train_labels, val_pca, id_val_labels, alpha_grid)
    else:
        raise ValueError(f"Unsupported gmm_ddu_pca covariance={covariance}")
    return {"pca": pca, "gmm": gmm, "covariance": covariance}


def score_gmm_pca(model, features: torch.Tensor) -> torch.Tensor:
    projected = transform_pca(model["pca"], features)
    covariance = model["covariance"]
    if covariance == "full":
        return score_gmm_full(model["gmm"], projected)
    if covariance == "diag":
        return score_gmm_diag(model["gmm"], projected)
    if covariance == "tied":
        return score_gmm_tied(model["gmm"], projected)
    if covariance == "shrinkage":
        return score_gmm_shrinkage(model["gmm"], projected)
    raise ValueError(f"Unsupported gmm_ddu_pca covariance={covariance}")


def fit_ncc(id_train_features: torch.Tensor, id_train_labels: torch.Tensor):
    return {"stats": class_stats(id_train_features, id_train_labels)}


def score_ncc_distance(model, features: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(features, model["stats"].means)
    return -distances.min(dim=1).values


def predict_ncc(model, features: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(features, model["stats"].means)
    pred_idx = distances.argmin(dim=1)
    return model["stats"].classes[pred_idx]


def ncc_accuracy(model, features: torch.Tensor, labels: torch.Tensor) -> float:
    pred = predict_ncc(model, features)
    return float((pred.cpu() == labels.cpu()).float().mean().item())


def score_nc_prototype_cosine(model, features: torch.Tensor) -> torch.Tensor:
    feature_dirs = F.normalize(features, p=2, dim=1, eps=1.0e-12)
    proto_dirs = F.normalize(model["stats"].means, p=2, dim=1, eps=1.0e-12)
    return (feature_dirs @ proto_dirs.T).max(dim=1).values


def fit_vim(
    id_train_features: torch.Tensor,
    id_train_logits: torch.Tensor,
    principal_dim: int,
):
    pca = fit_pca(id_train_features, principal_dim)
    residuals = vim_residual_norm({"pca": pca}, id_train_features)
    mean_residual = residuals.mean().clamp_min(1.0e-12)
    mean_maxlogit = id_train_logits.max(dim=1).values.mean().clamp_min(1.0e-12)
    alpha = float((mean_maxlogit / mean_residual).item())
    return {
        "pca": pca,
        "alpha": alpha,
        "alpha_fit_split": "id_train",
        "alpha_rule": "mean_id_train_maxlogit_div_mean_id_train_residual_norm",
    }


def vim_residual_norm(model, features: torch.Tensor) -> torch.Tensor:
    pca = model["pca"]
    centered = features - pca["mean"]
    projected = centered @ pca["components"].T @ pca["components"]
    return torch.norm(centered - projected, p=2, dim=1)


def score_vim_native_ood_score(model, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    residual = vim_residual_norm(model, features)
    virtual_logit = residual * float(model["alpha"])
    augmented = torch.cat([logits, virtual_logit[:, None]], dim=1)
    return F.softmax(augmented, dim=1)[:, -1]


def score_vim_id_score(model, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - score_vim_native_ood_score(model, features, logits)


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, p=2, dim=1, eps=1.0e-12)


FEATURE_DETECTOR_REGISTRY = {
    "mahalanobis": {"family": "feature"},
    "mahalanobis_l2": {"family": "feature"},
    "knn": {"family": "feature"},
    "knn_l2": {"family": "feature"},
    "gmm_ddu_tied": {"family": "feature_density"},
    "gmm_ddu_diag": {"family": "feature_density"},
    "gmm_ddu_shrinkage": {"family": "feature_density"},
    "gmm_ddu_full": {"family": "feature_density_diagnostic"},
    "gmm_ddu_pca": {"family": "feature_density_diagnostic"},
}


NC_HYBRID_DETECTOR_REGISTRY = {
    "ncc_distance": {"family": "nearest_class_center"},
    "ncc_accuracy": {"family": "nearest_class_center_label_metric"},
    "nc_prototype_cosine": {"family": "prototype_cosine"},
    "vim_id_score": {"family": "vim_hybrid"},
}
