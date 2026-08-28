"""Bootstrap confidence intervals for the benchmark's scope-aggregate BEDROC, computed
from the persisted out-of-fold vectors (no retraining).

The reported bar is the UNWEIGHTED mean over a scope's endpoints of each endpoint's
within-assay BEDROC. The CI must be a CI on exactly that quantity, so:

  * resample COMPOUNDS within each assay, with replacement (n_a -> n_a);
  * recompute that assay's BEDROC on the resample;
  * combine with the SAME unweighted mean over the endpoints used for the point estimate;
  * repeat B times; the 2.5/97.5 percentiles are the 95% CI.

The endpoint SET is held FIXED across resamples -- the endpoints are the population we
report on, not a random sample, so between-endpoint heterogeneity (the old "±1 SD across
assays" bars) is NOT folded in. Only within-endpoint compound-sampling noise is. Because
the mean is unweighted, sparse heads (few actives -> wide per-assay bootstrap) dominate
the interval; that is honest given we report the unweighted mean.

For a GAP between two methods, per-bar CIs overlap even when the difference is real,
because both are scored on the SAME compounds (correlated). paired_gap() resamples the
same compound indices for both arms and bootstraps the DIFFERENCE -- the correct, more
powerful test. It requires row-aligned OOF (same scope/split/features).

    micromamba run -n learn2rank python bootstrap_ci.py <oof.npz> [<oof_b.npz>]
"""
from __future__ import annotations

import numpy as np

from metrics import bedroc

ALPHA_BEDROC = 20.0


def _load(path: str):
    d = np.load(path, allow_pickle=True)
    oof, y, task, thr = d["oof"], d["y"], d["task"].astype(str), d["threshold"]
    m = ~np.isnan(oof)
    return oof[m], y[m], task[m], thr[m]


def _scored_assays(ybin: np.ndarray, task: np.ndarray) -> tuple[list, dict]:
    idx = {a: np.flatnonzero(task == a) for a in np.unique(task)}
    assays = [a for a in idx if ybin[idx[a]].sum() > 0]   # skip assays with no actives
    return assays, idx


def _aggregate(scores, ybin, assays, idx, rng=None) -> float:
    """Unweighted mean over `assays` of per-assay BEDROC; resample within assay if rng."""
    vals = []
    for a in assays:
        i = idx[a]
        if rng is not None:
            i = rng.choice(i, size=len(i), replace=True)
        yb = ybin[i]
        if yb.sum() == 0:               # degenerate resample (rare on sparse heads)
            continue
        vals.append(bedroc(yb, scores[i], alpha=ALPHA_BEDROC))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_aggregate(path: str, B: int = 1000, seed: int = 0,
                        alpha: float = 0.05) -> dict:
    """Point estimate + percentile CI of the scope-aggregate BEDROC for one config."""
    oof, y, task, thr = _load(path)
    ybin = (y >= thr).astype(int)
    assays, idx = _scored_assays(ybin, task)
    point = _aggregate(oof, ybin, assays, idx)
    rng = np.random.default_rng(seed)
    boots = np.array([_aggregate(oof, ybin, assays, idx, rng) for _ in range(B)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_assays": len(assays), "B": B}


def paired_gap(path_a: str, path_b: str, B: int = 1000, seed: int = 0,
               alpha: float = 0.05) -> dict:
    """Paired-difference bootstrap of aggregate BEDROC(a) - BEDROC(b).

    a and b must be row-aligned (identical task/threshold vectors -> same scope, split,
    representation). Shares resample indices between the two arms each iteration."""
    oa, ya, ta, tha = _load(path_a)
    ob, yb_, tb, thb = _load(path_b)
    if not (np.array_equal(ta, tb) and np.array_equal(tha, thb)):
        raise ValueError("OOF not row-aligned (need same scope/split/features)")
    ybin = (ya >= tha).astype(int)
    assays, idx = _scored_assays(ybin, ta)
    gap = _aggregate(oa, ybin, assays, idx) - _aggregate(ob, ybin, assays, idx)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        va, vb = [], []
        for a in assays:
            i = rng.choice(idx[a], size=len(idx[a]), replace=True)
            yb = ybin[i]
            if yb.sum() == 0:
                continue
            va.append(bedroc(yb, oa[i], alpha=ALPHA_BEDROC))
            vb.append(bedroc(yb, ob[i], alpha=ALPHA_BEDROC))
        diffs.append(np.mean(va) - np.mean(vb))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"gap": gap, "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "B": B}


def bootstrap_assay_draws(oof_path: str, assay: str, B: int = 1000,
                          seed: int = 0) -> np.ndarray:
    """Raw bootstrap draws of ONE assay's BEDROC (within-assay compound resampling).

    Returned so callers can pool draws across runs -- e.g. the pMIC ablation pools the
    5 subsample seeds into one CI that captures both the which-108-actives draw and the
    within-assay compound resampling.
    """
    oof, y, task, thr = _load(oof_path)
    m = task == assay
    yb = (y[m] >= thr[m]).astype(int)
    s = oof[m]
    idx = np.arange(len(yb))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        i = rng.choice(idx, size=len(idx), replace=True)
        if yb[i].sum() == 0:
            continue
        draws.append(bedroc(yb[i], s[i], alpha=ALPHA_BEDROC))
    return np.asarray(draws)


def bootstrap_aggregate_cached(oof_path: str, cache: dict, B: int = 1000,
                               seed: int = 0, alpha: float = 0.05) -> dict:
    """bootstrap_aggregate with a per-OOF cache so re-renders don't recompute.

    Keyed by the OOF filename; invalidated when the file's mtime or B changes (so the
    5 CV re-runs, which overwrite their OOF, recompute rather than serve a stale CI).
    `cache` is a plain dict the caller loads/saves as JSON.
    """
    import os
    from pathlib import Path

    key = Path(oof_path).name
    mtime = os.path.getmtime(oof_path)
    ent = cache.get(key)
    if ent and ent.get("mtime") == mtime and ent.get("B") == B and ent.get("seed") == seed:
        return ent
    r = bootstrap_aggregate(oof_path, B=B, seed=seed, alpha=alpha)
    r.update({"mtime": mtime, "B": B, "seed": seed})
    cache[key] = r
    return r


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print(bootstrap_aggregate(sys.argv[1]))
    elif len(sys.argv) == 3:
        print(paired_gap(sys.argv[1], sys.argv[2]))
    else:
        print(__doc__)
