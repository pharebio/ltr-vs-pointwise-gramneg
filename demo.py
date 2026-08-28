"""
Self-contained smoke test / demo of the LTR scaffold on synthetic multi-assay
data. Verifies the pipeline runs end-to-end and that a pairwise ranker recovers a
latent activity ordering that transfers across held-out assays.

Run:  python demo.py
(Requires: numpy, scipy, scikit-learn, pandas, torch)
"""
from __future__ import annotations

import numpy as np

from evaluate import compare_to_pointwise_baseline, run_lodo


def make_synthetic(n_per_assay=400, n_assays=4, d=50, hit_rate=0.03, seed=0):
    """
    Each assay shares a common latent 'true activity' = w . x, but applies its own
    monotone-but-nonlinear distortion + offset + noise (mimicking heterogeneous,
    non-normalized phenotypic readouts). A good ranker should recover the shared
    ordering even though raw values are not comparable across assays.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    Xs, ys, ds = [], [], []
    for a in range(n_assays):
        X = rng.normal(size=(n_per_assay, d))
        latent = X @ w
        # assay-specific monotone distortion + scale + offset + noise
        gain = rng.uniform(0.5, 2.0)
        offset = rng.uniform(-3, 3)
        distort = np.tanh(gain * (latent - latent.mean()))
        y = distort + offset + rng.normal(scale=0.3, size=n_per_assay)
        Xs.append(X)
        ys.append(y)
        ds.append(np.array([f"assay_{a}"] * n_per_assay))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    dataset_ids = np.concatenate(ds)
    labels = y.copy()
    # Per-assay active threshold: top `hit_rate` quantile computed WITHIN each assay.
    # A single global cutoff would leave some assays (with lower offsets) with zero
    # actives -> NaN BEDROC/EF; a per-assay map mirrors how real endpoints each carry
    # their own hit cutoff (80% GI vs pMIC 7). See thresholds.py.
    thresholds = {
        f"assay_{a}": float(np.quantile(ys[a], 1 - hit_rate)) for a in range(n_assays)
    }
    return X, labels, dataset_ids, thresholds


if __name__ == "__main__":
    X, labels, dataset_ids, thresholds = make_synthetic()
    thr_str = ", ".join(f"{k}={v:.2f}" for k, v in thresholds.items())
    print(f"data: X={X.shape}  assays={np.unique(dataset_ids).tolist()}\n"
          f"per-assay active_threshold: {thr_str}\n")

    print(">>> Pairwise LTR under Leave-One-Dataset-Out:")
    run_lodo(X, labels, dataset_ids, active_threshold=thresholds,
             objective="pairwise", sparse_sampling=True, epochs=20)

    print("\n>>> LTR vs pointwise-regression baseline (same rank metrics):")
    cmp = compare_to_pointwise_baseline(
        X, labels, dataset_ids, active_threshold=thresholds, epochs=20
    )
    summary = cmp.groupby("method")[["spearman", "ndcg", "bedroc", "ef@0.01"]].mean()
    print(summary.to_string())
