"""
Leave-One-Dataset-Out (LODO) evaluation harness for LTR models.

Ties together splits.py + pairs.py + model.py + metrics.py into the evaluation
protocol recommended in the report:

    for each held-out assay/campaign D:
        train a ranker on all OTHER datasets (pairs formed within their groups)
        score the held-out D
        report Spearman / NDCG / BEDROC / EF within D

Reporting *within* the held-out dataset is essential: it measures whether the
model transfers an ordering to unseen assay conditions, and it never requires
cross-assay value comparability at evaluation time either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import ranking_report
from model import MLPScorer, RankingTrainer
from pairs import build_within_group_pairs, sample_pairs_imbalanced
from splits import leave_one_dataset_out
from thresholds import ThresholdSpec, resolve_threshold


def run_lodo(
    X: np.ndarray,
    labels: np.ndarray,
    dataset_ids: np.ndarray,
    groups: np.ndarray | None = None,
    active_threshold: ThresholdSpec = None,
    objective: str = "pairwise",
    sparse_sampling: bool = False,
    epochs: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """
    X            : [n, d] feature matrix (RDKit descriptors / Morgan FP)
    labels       : [n] continuous activity (higher = more active); NaN allowed
    dataset_ids  : [n] which assay/campaign each row belongs to (LODO unit)
    groups       : [n] within-which comparisons are valid; defaults to dataset_ids
    active_threshold: label cutoff defining a "hit" for BEDROC/EF. A scalar applies
                      to every dataset; a {dataset_id: threshold} map or a callable
                      lets each held-out assay use its own cutoff (e.g. 80% GI for
                      %GI tasks, pMIC 7 for the pMIC task -- see thresholds.py).
    objective    : "pairwise" | "listwise"
    sparse_sampling: use active-anchored + hard-negative pair sampling
    """
    rng = np.random.default_rng(seed)
    groups = dataset_ids if groups is None else groups
    rows = []

    for held, train_idx, test_idx in leave_one_dataset_out(dataset_ids):
        Xtr, ytr, gtr = X[train_idx], labels[train_idx], groups[train_idx]
        Xte, yte = X[test_idx], labels[test_idx]

        scorer = MLPScorer(in_dim=X.shape[1])
        trainer = RankingTrainer(scorer, objective=objective)

        if objective == "listwise":
            trainer.fit_listwise(Xtr, ytr, gtr, epochs=epochs, verbose=False)
        else:
            if sparse_sampling and active_threshold is not None:
                pairs = sample_pairs_imbalanced(
                    ytr, gtr, active_threshold=active_threshold, rng=rng
                )
            else:
                pairs = build_within_group_pairs(ytr, gtr, rng=rng)
            if not pairs:
                continue
            trainer.fit_pairwise(Xtr, pairs, epochs=epochs, verbose=False)

        scores = trainer.score(Xte)
        thr = resolve_threshold(active_threshold, held)
        rep = ranking_report(yte, scores, active_threshold=thr)
        rep["held_out"] = held
        rows.append(rep)
        print(f"held-out={held:>24}  spearman={rep['spearman']:.3f}  "
              f"ndcg={rep['ndcg']:.3f}"
              + (f"  bedroc={rep.get('bedroc', float('nan')):.3f}"
                 if thr is not None else ""))

    df = pd.DataFrame(rows).set_index("held_out")
    print("\n=== LODO summary (mean across held-out datasets) ===")
    print(df.mean(numeric_only=True).to_string())
    return df


def compare_to_pointwise_baseline(
    X, labels, dataset_ids, groups=None, active_threshold: ThresholdSpec = None,
    epochs=30, seed=0,
) -> pd.DataFrame:
    """
    Head-to-head: LTR (pairwise) vs a pointwise regression baseline (RidgeCV),
    both evaluated with the SAME rank-aware metrics under LODO. This is the
    apples-to-apples experiment to justify (or refute) adopting LTR.

    active_threshold accepts a scalar, a {dataset_id: threshold} map, or a callable
    (see thresholds.py) so each held-out assay is scored on its own hit cutoff.
    """
    from sklearn.linear_model import RidgeCV

    groups = dataset_ids if groups is None else groups
    rows = []
    for held, tr, te in leave_one_dataset_out(dataset_ids):
        m = ~np.isnan(labels[tr])
        reg = RidgeCV().fit(X[tr][m], labels[tr][m])
        base_scores = reg.predict(X[te])
        thr = resolve_threshold(active_threshold, held)
        rep_b = ranking_report(labels[te], base_scores, active_threshold=thr)
        rep_b.update({"held_out": held, "method": "pointwise_ridge"})
        rows.append(rep_b)

    ltr = run_lodo(X, labels, dataset_ids, groups, active_threshold,
                   objective="pairwise", epochs=epochs, seed=seed)
    for held, r in ltr.iterrows():
        d = r.to_dict()
        d.update({"held_out": held, "method": "ltr_pairwise"})
        rows.append(d)

    return pd.DataFrame(rows)
