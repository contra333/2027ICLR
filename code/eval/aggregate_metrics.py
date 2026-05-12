from __future__ import annotations

import numpy as np


def _as_numpy(scores) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def auroc(id_scores, ood_scores) -> float:
    pos = _as_numpy(id_scores)
    neg = _as_numpy(ood_scores)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - labels.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aupr_in(id_scores, ood_scores) -> float:
    pos = _as_numpy(id_scores)
    neg = _as_numpy(ood_scores)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    total_pos = tp[-1]
    if total_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / total_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def fpr95(id_scores, ood_scores) -> float:
    pos = _as_numpy(id_scores)
    neg = _as_numpy(ood_scores)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    threshold = float(np.quantile(pos, 0.05))
    return float(np.mean(neg >= threshold))


def aggregate_ood(id_scores, ood_scores) -> dict[str, float]:
    return {
        "auroc": auroc(id_scores, ood_scores),
        "aupr_in": aupr_in(id_scores, ood_scores),
        "fpr95": fpr95(id_scores, ood_scores),
    }
