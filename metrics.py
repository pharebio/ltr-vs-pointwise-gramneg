"""
Rank-aware and early-recognition metrics for learn-to-rank (LTR) evaluation.

These metrics are deliberately model-agnostic: they take a vector of predicted
scores and a vector of ground-truth labels/values and return a scalar. They can
be dropped into any evaluator alongside RMSE / R2, or used standalone in the LODO
harness (``evaluate.py``).

Why these metrics:
    * Spearman / Kendall  -> global ordering quality (assay-invariant).
    * NDCG@k              -> graded, position-weighted ordering; rewards putting
                            the strongest actives at the very top.
    * EF@k                -> "enrichment factor": how many more hits appear in the
                            top k% than random. The number a screening chemist
                            actually cares about.
    * BEDROC / RIE        -> smooth, tunable early-recognition metrics that fix
                            the arbitrary-cutoff problem of EF@k. Recommended
                            primary metric for extreme-sparsity settings.

References:
    Truchon & Bayly, J. Chem. Inf. Model. 2007 (BEDROC / RIE, alpha=20).
    Jain & Nicholls, J. Comput. Aided Mol. Des. 2008 (early-recognition metrics).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Correlation-style (whole-list ordering)
# --------------------------------------------------------------------------- #
def spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman rank correlation. NaNs in y_true are dropped pairwise."""
    y_true, y_score = _drop_nan(y_true, y_score)
    if len(y_true) < 3:
        return float("nan")
    return float(stats.spearmanr(y_true, y_score).correlation)


def kendall_tau(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kendall's tau-b. Robust to ties; slower than Spearman on big lists."""
    y_true, y_score = _drop_nan(y_true, y_score)
    if len(y_true) < 3:
        return float("nan")
    return float(stats.kendalltau(y_true, y_score).correlation)


# --------------------------------------------------------------------------- #
# Graded ranking
# --------------------------------------------------------------------------- #
def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    """
    Normalized Discounted Cumulative Gain. Uses ``y_true`` as the (non-negative)
    graded relevance, so it works directly on continuous growth-inhibition values
    (higher = more active). If your labels can be negative, shift them first.
    """
    y_true, y_score = _drop_nan(y_true, y_score)
    if len(y_true) == 0:
        return float("nan")
    gains = np.clip(y_true, a_min=0.0, a_max=None)
    order = np.argsort(-y_score)
    ideal = np.argsort(-gains)
    k = k or len(y_true)

    def _dcg(idx):
        g = gains[idx][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(g) + 2))
        return float(np.sum(g * discounts))

    idcg = _dcg(ideal)
    return _dcg(order) / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Early recognition (require binary actives)
# --------------------------------------------------------------------------- #
def enrichment_factor(
    y_true_binary: np.ndarray, y_score: np.ndarray, fraction: float = 0.01
) -> float:
    """
    EF at the top ``fraction`` of the ranked list.

    EF = (hits_in_top / n_top) / (total_hits / n_total).
    EF=1 is random; the theoretical max is 1/hit_rate.
    """
    y_true_binary, y_score = _drop_nan(y_true_binary, y_score)
    n = len(y_true_binary)
    n_actives = int(y_true_binary.sum())
    if n == 0 or n_actives == 0:
        return float("nan")
    n_top = max(1, int(round(fraction * n)))
    top_idx = np.argsort(-y_score)[:n_top]
    hits_top = float(y_true_binary[top_idx].sum())
    return (hits_top / n_top) / (n_actives / n)


def rie(y_true_binary: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    Robust Initial Enhancement: mean early-weighted score over actives divided by
    the value expected under a random ranking. RIE=1 is random; higher is better.
    """
    y_true_binary, y_score = _drop_nan(y_true_binary, y_score)
    n = len(y_true_binary)
    n_actives = int(y_true_binary.sum())
    if n == 0 or n_actives == 0:
        return float("nan")
    order = np.argsort(-y_score)
    ranks = np.where(y_true_binary[order] == 1)[0] + 1  # 1-indexed active ranks
    num = np.sum(np.exp(-alpha * ranks / n)) / n_actives
    all_ranks = np.arange(1, n + 1)
    den = np.sum(np.exp(-alpha * all_ranks / n)) / n  # random expectation per active
    return float(num / den)


def auroc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area under the ROC curve, using the predicted scores to rank binarized hits.

    Included so the pointwise baseline and the LTR model are scored with identical
    metric code. Returns NaN if a list is all-active
    or all-inactive (AUROC undefined).
    """
    from sklearn.metrics import roc_auc_score

    y_true_binary, y_score = _drop_nan(y_true_binary, y_score)
    n_pos = int(y_true_binary.sum())
    if n_pos == 0 or n_pos == len(y_true_binary):
        return float("nan")
    return float(roc_auc_score(y_true_binary, y_score))


def bedroc(y_true_binary: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    Boltzmann-Enhanced Discrimination of ROC (Truchon & Bayly 2007).

    Implemented as the min-max normalization of RIE: BEDROC = (RIE - RIE_min) /
    (RIE_max - RIE_min), where RIE_max/min correspond to all actives placed at the
    very top / very bottom. Guarantees BEDROC in [0, 1] with perfect ranking -> 1,
    worst -> 0, random -> ~ n_actives/n. alpha=20 weights ~the top 8% of the list.
    Recommended primary metric for the extreme-sparsity setting.
    """
    y_true_binary, y_score = _drop_nan(y_true_binary, y_score)
    n = len(y_true_binary)
    n_actives = int(y_true_binary.sum())
    if n == 0 or n_actives == 0 or n_actives == n:
        return float("nan")
    all_ranks = np.arange(1, n + 1)
    den = np.sum(np.exp(-alpha * all_ranks / n)) / n

    def _rie_for(ranks):
        return (np.sum(np.exp(-alpha * ranks / n)) / n_actives) / den

    r = rie(y_true_binary, y_score, alpha)
    r_max = _rie_for(np.arange(1, n_actives + 1))
    r_min = _rie_for(np.arange(n - n_actives + 1, n + 1))
    return float((r - r_min) / (r_max - r_min))


# --------------------------------------------------------------------------- #
# Convenience: full report for one ranked list
# --------------------------------------------------------------------------- #
def ranking_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    active_threshold: float | None = None,
    ef_fractions: tuple[float, ...] = (0.01, 0.05),
    bedroc_alpha: float = 20.0,
) -> dict[str, float]:
    """
    Compute the full battery for a single assay / held-out dataset.

    active_threshold: if given, y_true is binarized (y_true >= threshold -> active)
    for the early-recognition metrics. If None, only correlation + NDCG returned.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    out = {
        "n": int(np.sum(~np.isnan(y_true))),
        "spearman": spearman(y_true, y_score),
        "kendall": kendall_tau(y_true, y_score),
        "ndcg": ndcg_at_k(y_true, y_score),
    }
    if active_threshold is not None:
        yb = (y_true >= active_threshold).astype(float)
        out["n_actives"] = int(np.nansum(yb))
        out["bedroc"] = bedroc(yb, y_score, alpha=bedroc_alpha)
        out["auroc"] = auroc(yb, y_score)
        out["rie"] = rie(yb, y_score, alpha=bedroc_alpha)
        for f in ef_fractions:
            out[f"ef@{f:g}"] = enrichment_factor(yb, y_score, fraction=f)
    return out


# --------------------------------------------------------------------------- #
def _drop_nan(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]
