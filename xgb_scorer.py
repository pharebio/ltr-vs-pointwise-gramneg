"""XGBoost scorers for the public benchmark: pointwise regression + LambdaMART.

Adds the one architecture the benchmark matrix was missing (issue #3): a gradient-boosted
tree model evaluated under BOTH objectives on identical features, so the MSE-vs-ranking
contrast is tested on GBDTs the same way `model.py` tests it on MLPs and
`ranking_chemprop.py` tests it end-to-end on the D-MPNN.

  pointwise   -- XGBRegressor, reg:squarederror. Ranks by predicted activity.
  lambdamart  -- XGBRanker, rank:ndcg. The classical learn-to-rank GBDT (Burges 2010);
                 a listwise-ish objective optimising a bound on NDCG via LambdaRank
                 gradients, i.e. the tree analogue of our LambdaLoss-NDCG.

Torch-free on purpose (numpy + xgboost only) so it stays importable in CI, which installs
no torch. The relevance mapping is therefore duplicated from ``ranking_chemprop`` rather
than imported -- the same trade-off that module already documents for the gate.

GROUPING: LambdaMART needs a "query" per comparison set. Here a query == an ASSAY, which
keeps every comparison within-assay and never puts %GI and pMIC on one axis -- the same
invariant the pairwise/listwise losses enforce.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5)


def xgb_relevance(y: np.ndarray, groups: np.ndarray | None = None, levels: int = 4,
                  mode: str = "continuous_gi", rel_max: float = 100.0,
                  threshold: float | dict | None = None) -> np.ndarray:
    """Map continuous activity to INTEGER relevance grades 0..levels for rank:ndcg.

    XGBoost's NDCG objective uses the label as a gain exponent (2^rel - 1), so labels must
    be small non-negative integers -- unlike our torch LambdaLoss, which accepts floats.

    modes:
      continuous_gi    -- global scale: levels * clip(y / rel_max, 0, 1). Use when every
                          assay shares one scale (%GI). This is the magnitude-aware
                          relabelling that beat binarised relevance in Phase 2 (E10).
      continuous_local -- per-assay min-max, so a log-scale endpoint (pMIC) is not crushed
                          by the dominant %GI range. Requires `groups`.
      bins             -- binary actives/inactives (0 or `levels`); needs `threshold`.
    """
    y = np.asarray(y, dtype=float)
    if mode == "bins":
        if threshold is None:
            raise ValueError("mode='bins' requires `threshold`")
        if isinstance(threshold, dict):
            if groups is None:
                raise ValueError("per-assay thresholds require `groups`")
            thr = np.array([threshold[g] for g in groups], dtype=float)
        else:
            thr = np.full(len(y), float(threshold))
        return np.where(y >= thr, levels, 0).astype(int)

    if mode == "continuous_gi":
        frac = np.clip(y / rel_max, 0.0, 1.0)
    elif mode == "continuous_local":
        if groups is None:
            raise ValueError("mode='continuous_local' requires `groups`")
        frac = np.zeros(len(y), dtype=float)
        for g in np.unique(groups):
            m = groups == g
            lo, hi = float(np.min(y[m])), float(np.max(y[m]))
            frac[m] = 0.0 if hi <= lo else np.clip((y[m] - lo) / (hi - lo), 0.0, 1.0)
    else:
        raise ValueError(f"unknown relevance mode {mode!r}")
    return np.rint(levels * frac).astype(int)


def fit_score_xgb_pointwise(Xtr, ytr, Xte, seed: int = 0, device: str = "cpu",
                            n_jobs: int = -1, **params) -> np.ndarray:
    """XGBoost regression (reg:squarederror). Ranks by predicted activity."""
    from xgboost import XGBRegressor

    p = {**DEFAULT_PARAMS, **params}
    model = XGBRegressor(objective="reg:squarederror", random_state=seed,
                         device=device, n_jobs=n_jobs, **p)
    model.fit(np.asarray(Xtr), np.asarray(ytr, dtype=float))
    return model.predict(np.asarray(Xte))


def fit_score_xgb_lambdamart(Xtr, ytr, groups_tr, Xte, seed: int = 0, device: str = "cpu",
                             n_jobs: int = -1, relevance_mode: str = "continuous_gi",
                             levels: int = 4, rel_max: float = 100.0,
                             threshold: float | dict | None = None,
                             **params) -> np.ndarray:
    """LambdaMART (XGBRanker, rank:ndcg) with one query group per ASSAY.

    XGBRanker requires rows sorted by qid, so we sort internally and never rely on the
    caller's ordering. Test scores are per-molecule and group-independent (the ranker
    outputs a single latent score), so `Xte` needs no qid.
    """
    from xgboost import XGBRanker

    Xtr, groups_tr = np.asarray(Xtr), np.asarray(groups_tr)
    rel = xgb_relevance(ytr, groups_tr, levels=levels, mode=relevance_mode,
                        rel_max=rel_max, threshold=threshold)

    # qid must be integral and, per XGBoost's API, rows must be grouped/sorted by it
    _, qid = np.unique(groups_tr, return_inverse=True)
    order = np.argsort(qid, kind="stable")

    p = {**DEFAULT_PARAMS, **params}
    model = XGBRanker(objective="rank:ndcg", random_state=seed, device=device,
                      n_jobs=n_jobs, **p)
    model.fit(Xtr[order], rel[order], qid=qid[order])
    return model.predict(np.asarray(Xte))
