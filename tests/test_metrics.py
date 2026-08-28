"""Numeric checks for the rank-aware / early-recognition metrics (no torch needed)."""
import numpy as np

import metrics


def test_bedroc_bounds():
    rng = np.random.default_rng(0)
    y = rng.random(2000)
    yb = (y > np.quantile(y, 0.98)).astype(float)  # 2% actives
    assert abs(metrics.bedroc(yb, y) - 1.0) < 1e-6      # perfect ranking
    assert abs(metrics.bedroc(yb, -y) - 0.0) < 1e-6     # worst ranking


def test_bedroc_random_near_hitrate():
    rng = np.random.default_rng(1)
    n, hit = 2000, 0.02
    y = rng.random(n)
    yb = (y > np.quantile(y, 1 - hit)).astype(float)
    vals = [metrics.bedroc(yb, rng.random(n)) for _ in range(200)]
    assert 0.0 < np.mean(vals) < 0.15  # random ~ hit_rate, well below a real ranker


def test_enrichment_factor():
    rng = np.random.default_rng(2)
    n, hit = 1000, 0.02
    y = rng.random(n)
    yb = (y > np.quantile(y, 1 - hit)).astype(float)
    ef_perfect = metrics.enrichment_factor(yb, y, fraction=0.01)
    assert ef_perfect > 40  # near theoretical max 1/hit = 50
    assert np.isnan(metrics.enrichment_factor(np.zeros(n), y))  # no actives -> nan


def test_rie_random_is_one():
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.random(n)
    yb = (y > np.quantile(y, 0.98)).astype(float)
    vals = [metrics.rie(yb, rng.random(n)) for _ in range(50)]
    assert abs(np.mean(vals) - 1.0) < 0.2  # RIE=1 under random


def test_spearman_ndcg_perfect():
    y = np.linspace(0, 1, 100)
    assert abs(metrics.spearman(y, y) - 1.0) < 1e-9
    assert abs(metrics.ndcg_at_k(y, y) - 1.0) < 1e-9
    assert metrics.ndcg_at_k(y, -y) < metrics.ndcg_at_k(y, y)


def test_ranking_report_keys():
    rng = np.random.default_rng(4)
    y = rng.random(500)
    rep = metrics.ranking_report(y, y + rng.normal(scale=0.1, size=500),
                                 active_threshold=np.quantile(y, 0.95))
    for k in ("spearman", "kendall", "ndcg", "bedroc", "rie", "ef@0.01", "n_actives"):
        assert k in rep


def test_nan_handling():
    y = np.array([1.0, np.nan, 3.0, 2.0])
    s = np.array([1.0, 2.0, 3.0, 2.0])
    assert not np.isnan(metrics.spearman(y, s))  # NaN dropped pairwise
