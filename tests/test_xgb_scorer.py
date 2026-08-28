"""
Correctness tests for xgb_scorer's relevance mapping (the LambdaMART label encoding).

xgb_scorer is torch-free by design, so the relevance tests run in the torch-free CI env.
The two fit functions need xgboost, which CI does not install, so those are importorskip'd.

The relevance mapping is the part most worth pinning: XGBoost's rank:ndcg uses the label
as a gain exponent (2^rel - 1), so a mis-scaled label silently destroys the ranking signal
rather than erroring -- which is exactly the cross-scale failure we measured in E6/E13,
where a global scale crushed the pMIC endpoint.
"""
import numpy as np
import pytest

from xgb_scorer import xgb_relevance


def test_continuous_gi_is_integral_bounded_and_monotone():
    y = np.array([0.0, 10.0, 50.0, 90.0, 100.0])
    rel = xgb_relevance(y, levels=4, mode="continuous_gi")
    assert rel.dtype.kind == "i", "rank:ndcg needs integer grades"
    assert rel.min() >= 0 and rel.max() <= 4
    assert np.all(np.diff(rel) >= 0), "higher %GI must never get a lower grade"


def test_continuous_gi_clips_out_of_range_values():
    """%GI can exceed 100 or go negative in real screens; grades must stay in [0, levels]."""
    rel = xgb_relevance(np.array([-30.0, 130.0]), levels=4, mode="continuous_gi")
    assert list(rel) == [0, 4]


def test_continuous_local_rescues_a_second_scale():
    """The cross-scale invariant: a global scale collapses pMIC to a single grade, so the
    ranker gets no within-assay signal; per-assay min-max must spread it."""
    y = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 4.0, 5.0, 6.0, 7.0, 9.0])
    g = np.array(["gi"] * 5 + ["pmic"] * 5)

    glob = xgb_relevance(y, g, levels=4, mode="continuous_gi")
    assert len(set(glob[5:])) == 1, "global scale is expected to crush the pMIC assay"

    loc = xgb_relevance(y, g, levels=4, mode="continuous_local")
    assert loc[5:].min() == 0 and loc[5:].max() == 4, "pMIC must span the full grade range"
    assert loc[:5].min() == 0 and loc[:5].max() == 4, "%GI must be unaffected"


def test_continuous_local_handles_a_constant_assay():
    """A single-valued assay has no spread; it must yield grade 0, not divide by zero."""
    y = np.array([7.0, 7.0, 7.0])
    rel = xgb_relevance(y, np.array(["x"] * 3), mode="continuous_local")
    assert list(rel) == [0, 0, 0]


def test_bins_uses_per_assay_thresholds():
    """Per-endpoint cutoffs: 80 %GI vs pMIC 7 -- a scalar threshold would mislabel pMIC."""
    y = np.array([79.0, 81.0, 6.0, 7.5])
    g = np.array(["gi", "gi", "pmic", "pmic"])
    rel = xgb_relevance(y, g, levels=4, mode="bins", threshold={"gi": 80.0, "pmic": 7.0})
    assert list(rel) == [0, 4, 0, 4]


def test_bins_requires_a_threshold():
    with pytest.raises(ValueError):
        xgb_relevance(np.array([1.0]), mode="bins")


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        xgb_relevance(np.array([1.0]), mode="nope")


def test_lambdamart_accepts_interleaved_assay_rows():
    """The internal qid sort is load-bearing, not cosmetic.

    XGBoost hard-fails on ungrouped qid ("`qid` must be sorted in non-decreasing order
    along with data"), and our long-format data arrives with assay rows interleaved, so
    without the sort every LambdaMART run would crash. Passing interleaved groups must
    just work, and must produce a usable (non-constant) score.
    """
    pytest.importorskip("xgboost")
    from xgb_scorer import fit_score_xgb_lambdamart

    rng = np.random.default_rng(0)
    n, d = 400, 8
    X = rng.normal(size=(n, d))
    y = np.clip(50 + 30 * X[:, 0], 0, 100)
    g = rng.choice(["a", "b", "c"], size=n)          # interleaved assays, NOT grouped
    Xte = rng.normal(size=(50, d))

    s = fit_score_xgb_lambdamart(X, y, g, Xte, seed=0, n_estimators=20, max_depth=3)
    assert s.shape == (50,)
    assert np.isfinite(s).all()
    assert s.std() > 0, "a constant score would mean the ranker learned nothing"


def test_lambdamart_is_stable_but_not_bitwise_reproducible_across_row_order():
    """Documents a real XGBoost property so it is not mistaken for a bug later.

    Row order changes tree construction (hist binning tie-breaks / float accumulation
    order), so permuting the input yields near-identical but NOT identical scores -- this
    holds even with subsample=colsample=1.0, i.e. it is not just stochastic sampling.
    Rankings agree to ~0.98+ correlation, which is what actually matters for our metrics.
    """
    pytest.importorskip("xgboost")
    from xgb_scorer import fit_score_xgb_lambdamart

    rng = np.random.default_rng(0)
    n, d = 400, 8
    X = rng.normal(size=(n, d))
    y = np.clip(50 + 30 * X[:, 0], 0, 100)
    g = rng.choice(["a", "b", "c"], size=n)
    Xte = rng.normal(size=(50, d))

    kw = dict(seed=0, n_estimators=20, max_depth=3)
    s1 = fit_score_xgb_lambdamart(X, y, g, Xte, **kw)
    p = rng.permutation(n)
    s2 = fit_score_xgb_lambdamart(X[p], y[p], g[p], Xte, **kw)
    assert np.corrcoef(s1, s2)[0, 1] > 0.95
