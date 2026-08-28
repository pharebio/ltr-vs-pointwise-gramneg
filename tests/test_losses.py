"""
Correctness tests for the LambdaLoss-NDCG surrogate (losses.lambdaloss_ndcg).

Guarded by importorskip: losses.py needs torch, which the torch-free CI env does
not install, so these are skipped there and run locally
(`micromamba run -n learn2rank python -m pytest -q tests/test_losses.py`).

They pin the two things we verified when auditing why LambdaLoss underperformed:
  1. our ΔNDCG weighting matches the canonical Wang et al. 2018 (Eqs 5-6) formula
     (== allRank's lambdaRank_scheme) with `reduction="sum"`;
  2. `reduction="sum"` (the paper/allRank default) gives an ~n_pairs larger gradient
     than `reduction="mean"` -- averaging over the huge pair count is what
     undertrained the listwise model at a fixed learning rate.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from losses import lambdaloss_ndcg  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402
from ranking_chemprop import list_relevance  # noqa: E402


def _canonical_sum(scores, labels, sigma=1.0):
    """Wang et al. 2018 Eqs 5-6 (lambdaRank scheme), log2, sum reduction."""
    n = scores.shape[0]
    gains = 2.0 ** labels - 1.0
    ideal, _ = torch.sort(gains, descending=True)
    pos_idx = torch.arange(1, n + 1, dtype=scores.dtype)
    max_dcg = (ideal / torch.log2(1.0 + pos_idx)).sum().clamp(min=1e-10)
    g = gains / max_dcg
    rank = torch.argsort(torch.argsort(scores, descending=True)).to(scores.dtype) + 1
    inv_d = 1.0 / torch.log2(1.0 + rank)
    d_ndcg = (g[:, None] - g[None, :]).abs() * (inv_d[:, None] - inv_d[None, :]).abs()
    s_diff = scores[:, None] - scores[None, :]
    pos = (labels[:, None] > labels[None, :]).to(scores.dtype)
    logistic = torch.log1p(torch.exp(-sigma * s_diff)) / np.log(2.0)
    return (d_ndcg * pos * logistic).sum()


def test_lambdaloss_matches_canonical_sum():
    torch.manual_seed(0)
    labels = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    scores = torch.randn(8)
    ours = float(lambdaloss_ndcg(scores.clone(), labels, reduction="sum"))
    ref = float(_canonical_sum(scores.clone(), labels))
    assert np.isclose(ours, ref, rtol=1e-4), f"ours={ours} vs canonical={ref}"


def test_lambdaloss_recovers_ideal_order():
    labels = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    s = torch.zeros(8, requires_grad=True)
    opt = torch.optim.Adam([s], lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        lambdaloss_ndcg(s, labels, reduction="sum").backward()
        opt.step()
    assert ndcg_at_k(labels.numpy(), s.detach().numpy()) > 0.99


def test_sum_reduction_gradient_larger_than_mean():
    # averaging over the many ordered pairs shrinks the gradient by ~n_pairs
    rng = np.random.default_rng(0)
    labels = torch.tensor(
        np.concatenate([rng.integers(1, 5, 40), np.zeros(200)]).astype(np.float32))
    n_pos = int((labels[:, None] > labels[None, :]).sum())

    def grad_norm(reduction):
        s = torch.zeros(len(labels), requires_grad=True)
        g = torch.autograd.grad(lambdaloss_ndcg(s, labels, reduction=reduction), s)[0]
        return float(g.norm())

    gn_sum, gn_mean = grad_norm("sum"), grad_norm("mean")
    assert gn_sum > gn_mean
    # ratio tracks the pair count (within an order of magnitude)
    assert gn_sum / (gn_mean + 1e-12) > 0.1 * n_pos


# --- non-binarized whole-list relevance (continuous_gi / rank_gi) + gain toggle --- #
def test_list_relevance_continuous_gi():
    y = np.array([-5.0, 0.0, 2.0, 40.0, 60.0, 85.0, 98.0, 130.0])
    r = list_relevance(y, mode="continuous_gi", levels=4, rel_max=100.0)
    assert r.min() >= 0.0 and r.max() <= 4.0          # bounded so 2**rel stays finite
    assert r[0] == 0.0 and r[1] == 0.0                # negatives / zero -> 0 (floor, no cliff)
    assert np.all(np.diff(r) >= 0)                    # monotonic in %GI
    assert np.isclose(r[-1], 4.0)                     # > rel_max clipped to L
    assert np.isclose(r[5], 4.0 * 85 / 100)           # scaled magnitude


def test_list_relevance_rank_gi_spacing():
    y = np.array([2.0, 40.0, 10.0, 95.0])             # unsorted
    r = list_relevance(y, mode="rank_gi", levels=3)
    assert np.allclose(np.sort(r), [0.0, 1.0, 2.0, 3.0])   # evenly spaced percentile * L
    assert r[np.argmax(y)] == 3.0 and r[np.argmin(y)] == 0.0


def test_lambdaloss_gain_linear_vs_exp():
    labels = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0])
    scores = torch.randn(5)
    lin = float(lambdaloss_ndcg(scores.clone(), labels, gain="linear"))
    exp = float(lambdaloss_ndcg(scores.clone(), labels, gain="exp"))
    assert np.isfinite(lin) and np.isfinite(exp)
    assert not np.isclose(lin, exp)                   # different gain -> different weighting
    # linear stays finite where exp overflows on a large (accidentally-unscaled) relevance
    big = torch.tensor([0.0, 2000.0])
    assert np.isfinite(float(lambdaloss_ndcg(torch.tensor([0.1, 0.2]), big, gain="linear")))
    assert not np.isfinite(float(lambdaloss_ndcg(torch.tensor([0.1, 0.2]), big, gain="exp")))
