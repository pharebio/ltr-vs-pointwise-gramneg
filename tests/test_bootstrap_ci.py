"""Bootstrap-CI invariants (pure numpy; runs in CI without torch)."""
import numpy as np
import pytest

from bootstrap_ci import bootstrap_aggregate, paired_gap


def _write_oof(tmp_path, oof, y, task, thr, name="oof.npz"):
    p = tmp_path / name
    np.savez(p, oof=oof.astype(np.float32), y=y.astype(np.float32),
             task=np.asarray(task).astype("U"), threshold=thr.astype(np.float32),
             fold=np.zeros(len(y), dtype=np.float32), run_id=np.array("t"))
    return str(p)


def _synthetic(n_per=400, n_assays=3, sep=0.0, seed=0):
    """n_assays endpoints; scores correlate with a binary label by strength `sep`."""
    rng = np.random.default_rng(seed)
    oof, y, task, thr = [], [], [], []
    for a in range(n_assays):
        yb = (rng.random(n_per) < 0.1).astype(float)   # ~10% actives
        s = sep * yb + rng.normal(size=n_per)           # sep=0 -> pure noise
        oof.append(s)
        y.append(yb)
        thr.append(np.full(n_per, 0.5))
        task += [f"assay{a}"] * n_per
    return (np.concatenate(oof), np.concatenate(y), np.array(task),
            np.concatenate(thr))


def test_point_estimate_inside_ci(tmp_path):
    oof, y, task, thr = _synthetic(sep=3.0)
    p = _write_oof(tmp_path, oof, y, task, thr)
    r = bootstrap_aggregate(p, B=200)
    assert r["lo"] <= r["point"] <= r["hi"]
    assert r["n_assays"] == 3


def test_paired_self_gap_is_zero(tmp_path):
    """A config against itself: gap exactly 0, CI must include 0."""
    oof, y, task, thr = _synthetic(sep=2.0)
    p = _write_oof(tmp_path, oof, y, task, thr)
    r = paired_gap(p, p, B=200)
    assert r["gap"] == pytest.approx(0.0, abs=1e-9)
    assert not r["excludes_zero"]


def test_paired_gap_detects_real_separation(tmp_path):
    """Strong ranker vs pure-noise ranker on the SAME labels: gap > 0, CI excludes 0."""
    oof_s, y, task, thr = _synthetic(sep=4.0, seed=1)
    oof_n, _, _, _ = _synthetic(sep=0.0, seed=1)          # same labels, noise scores
    good = _write_oof(tmp_path, oof_s, y, task, thr, "good.npz")
    bad = _write_oof(tmp_path, oof_n, y, task, thr, "bad.npz")
    r = paired_gap(good, bad, B=300)
    assert r["gap"] > 0
    assert r["excludes_zero"]


def test_paired_gap_rejects_misaligned(tmp_path):
    a = _write_oof(tmp_path, *_synthetic(n_assays=3), name="a.npz")
    b = _write_oof(tmp_path, *_synthetic(n_assays=2), name="b.npz")
    with pytest.raises(ValueError, match="row-aligned"):
        paired_gap(a, b, B=50)
