"""Multi-task head scoring, including the sibling fallback used under LODO.

The fallback exists because LODO deletes the head for the very assay being tested (every
task belongs to exactly one source), which left gnn_mt with every assay unscored. These
are pure-numpy so they run in CI, where torch is not installed.
"""
import numpy as np
import pytest

from run_benchmark import sibling_percentile_score


def test_rows_with_a_head_use_that_head_verbatim():
    P = np.array([[1.0, 90.0], [2.0, 80.0], [3.0, 70.0]])
    out = sibling_percentile_score(P, te_inv=np.array([0, 1, 2]), cols=np.array([0, 0, 1]))
    assert out.tolist() == [1.0, 2.0, 70.0]


def test_missing_head_falls_back_to_sibling_mean_percentile():
    # compound 2 ranks top on both heads, compound 0 bottom on both -> ordering preserved
    P = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    out = sibling_percentile_score(P, te_inv=np.array([0, 1, 2]), cols=np.array([-1, -1, -1]))
    assert not np.isnan(out).any()
    assert out[0] < out[1] < out[2]
    assert out.max() <= 1.0 and out.min() > 0.0


def test_disagreeing_heads_average_rather_than_follow_one():
    # head 0 and head 1 rank the compounds in OPPOSITE orders, so consensus is flat
    P = np.array([[1.0, 30.0], [2.0, 20.0], [3.0, 10.0]])
    out = sibling_percentile_score(P, te_inv=np.arange(3), cols=np.full(3, -1))
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2])


def test_the_lodo_case_leaves_no_row_unscored():
    """Every test row's assay is absent from training -- the actual LODO situation."""
    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 4))
    te_inv = rng.integers(0, 50, size=200)
    out = sibling_percentile_score(P, te_inv, cols=np.full(200, -1))
    assert not np.isnan(out).any()
    # rows sharing a compound must get the same score -- scoring is per-compound
    for c in set(te_inv.tolist()):
        assert len(set(out[te_inv == c].round(12))) == 1


def test_mixed_case_scores_both_kinds_of_row():
    P = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    out = sibling_percentile_score(P, te_inv=np.array([0, 1, 2]), cols=np.array([1, -1, 0]))
    assert out[0] == 10.0          # head 1, raw
    assert out[2] == 3.0           # head 0, raw
    assert 0.0 < out[1] <= 1.0     # sibling percentile, a different scale by construction


def test_heads_on_wildly_different_scales_do_not_dominate_consensus():
    """pMIC (~4-10) next to %GI (0-100): raw averaging would let %GI decide the order."""
    P = np.array([[100.0, 4.0], [50.0, 9.0], [1.0, 7.0]])
    out = sibling_percentile_score(P, te_inv=np.arange(3), cols=np.full(3, -1))
    # percentile ranks: head0 -> 3,2,1 ; head1 -> 1,3,2  => means 2.0, 2.5, 1.5
    assert out[1] > out[0] > out[2]
