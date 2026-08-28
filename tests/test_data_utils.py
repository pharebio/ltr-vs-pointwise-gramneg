"""Checks for pair construction and evaluation splits (no torch needed)."""
import json

import numpy as np

import pairs
import splits
import thresholds


def test_within_group_pairs():
    labels = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    groups = np.array(["a", "a", "a", "b", "b"])
    p = pairs.build_within_group_pairs(labels, groups, min_gap=0.0)
    # group a (3 items) -> 3 ordered "greater" pairs; group b (2 items) -> 1; total 4
    assert len(p) == 4
    # no pair ever crosses a group boundary
    for pr in p:
        assert groups[pr.i] == groups[pr.j]
        assert labels[pr.i] > labels[pr.j]


def test_min_gap_filters_small_differences():
    labels = np.array([1.0, 1.05, 3.0])
    groups = np.array(["a", "a", "a"])
    p = pairs.build_within_group_pairs(labels, groups, min_gap=0.1)
    # the 1.0 vs 1.05 comparison is inside the DMZ and must be dropped
    assert all(abs(labels[pr.i] - labels[pr.j]) > 0.1 for pr in p)


def test_censored_pairs_only_when_certain():
    # a: ">64" (lower-bounded high), b: "=8", c: "<2"
    vals = np.array([64.0, 8.0, 2.0])
    qual = np.array([">", "=", "<"])
    groups = np.array(["g", "g", "g"])
    p = pairs.handle_censored(vals, qual, groups)
    got = {(pr.i, pr.j) for pr in p}
    # a>64 certainly beats b=8 and c<2; b=8 certainly beats c<2
    assert (0, 1) in got and (0, 2) in got and (1, 2) in got
    # never the reverse
    assert (1, 0) not in got and (2, 0) not in got


def test_imbalanced_sampling_anchors_on_actives():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.ones(3), np.zeros(97)])  # 3% actives
    groups = np.array(["g"] * 100)
    p = pairs.sample_pairs_imbalanced(labels, groups, active_threshold=0.5,
                                      n_neg_per_active=10, rng=rng)
    assert len(p) == 30  # 3 actives x 10 negatives
    for pr in p:
        assert labels[pr.i] == 1 and labels[pr.j] == 0


def test_imbalanced_sampling_active_active_ranked_by_value():
    # one group: 3 actives with distinct %GI + 4 inactives
    labels = np.array([90.0, 85.0, 82.0, 10.0, 20.0, 5.0, 0.0])
    groups = np.array(["g"] * 7)
    # isolate active-active pairs (no negatives sampled) -> ordered by value
    p = pairs.sample_pairs_imbalanced(labels, groups, active_threshold=80.0,
                                      n_neg_per_active=0, n_pos_per_active=10)
    # 90>{85,82}, 85>{82}, 82>{} = 3 ordered active-over-active pairs
    assert len(p) == 3
    for pr in p:
        assert labels[pr.i] >= 80.0 and labels[pr.j] >= 80.0   # both actives
        assert labels[pr.i] > labels[pr.j]                     # higher %GI outranks
    # pos_weight is carried onto active-active pairs for the ranking loss
    pw = pairs.sample_pairs_imbalanced(labels, groups, active_threshold=80.0,
                                       n_neg_per_active=0, n_pos_per_active=10,
                                       pos_weight=3.0)
    assert all(pr.weight == 3.0 for pr in pw)
    # default (n_pos=0) forms NO active-active pairs -> backward compatible
    p0 = pairs.sample_pairs_imbalanced(labels, groups, active_threshold=80.0,
                                       n_neg_per_active=2)
    assert all(not (labels[pr.i] >= 80 and labels[pr.j] >= 80) for pr in p0)


def test_lodo_partition():
    ds = np.array(["x", "x", "y", "y", "z"])
    seen = []
    for held, tr, te in splits.leave_one_dataset_out(ds):
        assert set(ds[te]) == {held}
        assert held not in set(ds[tr])
        assert len(tr) + len(te) == len(ds)  # exact partition
        seen.append(held)
    assert sorted(seen) == ["x", "y", "z"]


def test_scaffold_group_holdout_no_leakage():
    rng = np.random.default_rng(1)
    scaf = np.array([i // 4 for i in range(40)])  # 10 scaffold clusters
    for _, tr, te in splits.leave_one_scaffold_group_out(scaf, n_folds=5, rng=rng):
        assert set(scaf[tr]).isdisjoint(set(scaf[te]))  # whole clusters held out


def test_load_endpoint_thresholds_both_shapes(tmp_path):
    # nested metadata shape: {task: {"threshold": x, "dataset": ...}}
    meta = {"ecoli_gi": {"threshold": 80, "dataset": "coadd_os"},
            "gadiya_pmic": {"threshold": 7.0, "dataset": "gadiya_pmic"}}
    p = tmp_path / "dataset_metadata.json"
    p.write_text(json.dumps(meta))
    thr = thresholds.load_endpoint_thresholds(p)
    assert thr == {"ecoli_gi": 80.0, "gadiya_pmic": 7.0}
    # flat {task: threshold} shape and an already-loaded dict both work
    assert thresholds.load_endpoint_thresholds({"a": 5}) == {"a": 5.0}


def test_resolve_threshold_scalar_map_callable():
    assert thresholds.resolve_threshold(None, "x") is None
    assert thresholds.resolve_threshold(80.0, "anything") == 80.0        # scalar broadcasts
    m = {"ecoli_gi": 80.0, "gadiya_pmic": 7.0}
    assert thresholds.resolve_threshold(m, "gadiya_pmic") == 7.0
    assert thresholds.resolve_threshold(m, "unlisted") is None           # absent -> None
    assert thresholds.resolve_threshold(lambda k: 42.0, "x") == 42.0


def test_imbalanced_sampling_per_group_threshold():
    # Two groups with DIFFERENT hit cutoffs: %GI-like (>=80) and pMIC-like (>=7).
    labels = np.array([90.0, 85.0, 10.0, 20.0,   # group gi: 2 actives (>=80)
                       8.0, 9.0, 2.0, 3.0])       # group pmic: 2 actives (>=7)
    groups = np.array(["gi", "gi", "gi", "gi", "pmic", "pmic", "pmic", "pmic"])
    thr_map = {"gi": 80.0, "pmic": 7.0}
    p = pairs.sample_pairs_imbalanced(labels, groups, active_threshold=thr_map,
                                      n_neg_per_active=5)
    # 2 actives x 2 negatives per group x 2 groups = 8 pairs, each active-over-inactive
    assert len(p) == 8
    for pr in p:
        assert labels[pr.i] > labels[pr.j]
        assert groups[pr.i] == groups[pr.j]
    # A single global scalar of 80 would find NO actives in the pmic group:
    p_scalar = pairs.sample_pairs_imbalanced(labels, groups, active_threshold=80.0,
                                             n_neg_per_active=5)
    assert {pr.group for pr in p_scalar} == {"gi"}  # pmic group silently contributes nothing
