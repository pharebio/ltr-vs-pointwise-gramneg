"""
Learn-to-rank and contrastive loss functions (PyTorch).

All losses operate on a batch of *scores* produced by any scoring network
f: molecule -> R (e.g. a D-MPNN encoder + linear head, or an MLP over RDKit
descriptors). They make no assumption about the encoder, so they slot on top of
either the ChemProp encoder (via its Python API) or a precomputed
RDKit feature pipeline.

Three families are provided, mapping to the three strategies in the report:

    1. Pairwise   : RankNet / margin. Learns from *relative* order within an
                    assay -> no cross-assay normalization needed. Naturally
                    supports censored (bounded) labels: you only emit a pair
                    when the order is known.
    2. Listwise   : ListNet (top-1) and a LambdaLoss-style NDCG-weighted pairwise
                    loss that directly optimizes early recognition.
    3. Contrastive: Supervised NT-Xent (InfoNCE) and an activity-cliff-weighted
                    variant that pulls same-activity molecules together and pushes
                    activity-cliff pairs apart in embedding space.

References:
    Burges et al. 2005 (RankNet); Cao et al. 2007 (ListNet);
    Wang et al. 2018 (LambdaLoss); Khosla et al. 2020 (SupCon);
    Shen et al. 2024/2026 (activity-cliff contrastive).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Pairwise
# --------------------------------------------------------------------------- #
def ranknet_loss(
    scores_i: torch.Tensor,
    scores_j: torch.Tensor,
    labels_i: torch.Tensor,
    labels_j: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RankNet pairwise cross-entropy. For each pair (i, j) the target probability
    that i outranks j is 1 if labels_i > labels_j else 0 (0.5 for ties).

    Because the target only uses the *sign* of (labels_i - labels_j), values from
    different assays never need to be on the same scale -- you just need pairs to
    be drawn from within the same assay/context (see pairs.build_within_group_pairs).

    weight: optional per-pair weight, e.g. larger for cross-assay-agreeing pairs
            or for pairs spanning an activity cliff (LambdaLoss-style).
    """
    s_diff = scores_i - scores_j
    with torch.no_grad():
        target = torch.sign(labels_i - labels_j).clamp(min=-1, max=1)
        target = (target + 1.0) / 2.0  # {-1,0,1} -> {0,0.5,1}
    loss = F.binary_cross_entropy_with_logits(s_diff, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def margin_ranking_loss(
    scores_i: torch.Tensor,
    scores_j: torch.Tensor,
    labels_i: torch.Tensor,
    labels_j: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Hinge pairwise loss: max(0, margin - (s_i - s_j) * sign(y_i - y_j))."""
    y = torch.sign(labels_i - labels_j)
    return F.margin_ranking_loss(scores_i, scores_j, y, margin=margin)


# --------------------------------------------------------------------------- #
# 2. Listwise
# --------------------------------------------------------------------------- #
def listnet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ListNet top-1 loss: cross-entropy between the softmax over predicted scores and
    the softmax over ground-truth labels for a single ranked list (one assay batch).
    Operate on one group (assay) at a time; average across groups outside.
    """
    p_true = F.softmax(labels, dim=0)
    log_p_pred = F.log_softmax(scores, dim=0)
    return -(p_true * log_p_pred).sum()


_LN2 = 0.6931471805599453


def lambdaloss_ndcg(scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-10,
                    reduction: str = "sum", gain: str = "exp",
                    k: int | None = None) -> torch.Tensor:
    """
    LambdaRank / LambdaLoss NDCG surrogate for a single list (Wang et al. 2018,
    Eqs 5-6; == allRank's ``lambdaRank_scheme``). Each pair (i, j) with y_i > y_j is
    weighted by the NDCG change a swap would cause,

        ΔNDCG(i,j) = |G_i - G_j| · |1/D_i - 1/D_j| / maxDCG,   G = 2^y - 1,
        D = log2(1 + rank),

    and contributes  ΔNDCG(i,j) · log2(1 + e^-(s_i - s_j)),  so gradient concentrates
    on the top of the list -- what early-recognition (BEDROC/EF) rewards.

    reduction: **"sum"** (default) matches the paper and allRank. "mean" divides by
    the number of ordered pairs; note that averaging over the (often huge) pair count
    shrinks the gradient by ~n_pairs, which badly undertrains the model at a fixed
    learning rate -- prefer "sum".

    gain: **"exp"** (default) uses G = 2^y - 1 (standard NDCG, top-heavy) -- relevance MUST
    be small (<= ~10) or 2^y overflows. "linear" uses G = y, gentler and safe over a wider
    relevance range; the natural choice with continuous whole-list %GI relevance.
    """
    n = scores.shape[0]
    if n < 2:
        return scores.sum() * 0.0
    gains = (2.0 ** labels - 1.0) if gain == "exp" else labels
    # ideal DCG for normalization
    ideal_gains, _ = torch.sort(gains, descending=True)
    discounts = 1.0 / torch.log2(torch.arange(2, n + 2, device=scores.device, dtype=scores.dtype))
    if k is not None and k < n:
        # NDCG@k: truncate the IDCG to the top k as well, or the normaliser would credit
        # gain the truncated DCG can never reach and the loss would not sum to 1 at the
        # optimum.
        idcg = (ideal_gains[:k] * discounts[:k]).sum().clamp(min=eps)
    else:
        idcg = (ideal_gains * discounts).sum().clamp(min=eps)

    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)          # [n, n]
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    pos = (label_diff > 0).float()                              # i should outrank j

    # rank-based discount delta (approximate |DCG swap|) using current predicted order
    rank = torch.argsort(torch.argsort(scores, descending=True)).float() + 1
    disc = 1.0 / torch.log2(rank + 1.0)
    if k is not None and k < n:
        # Positions past k contribute no discount, so a swap wholly below the cutoff gets
        # zero weight and only swaps touching the top k drive the gradient. This is the
        # point of the variant: full-list NDCG's log tail decays so slowly (weight 1.00 at
        # rank 1 vs 0.06 at rank 100,000) that the objective is dominated by ordering the
        # irrelevant bulk -- measured here as NDCG moving 0.74->0.84 across a BEDROC range
        # of 0.25->0.74, i.e. nearly flat with respect to what we actually score.
        disc = torch.where(rank <= k, disc, torch.zeros_like(disc))
    delta = torch.abs(disc.unsqueeze(1) - disc.unsqueeze(0))
    gain_delta = torch.abs(gains.unsqueeze(1) - gains.unsqueeze(0))
    weight = (gain_delta * delta) / idcg

    # log2(1 + e^-(s_i - s_j)); BCE-with-logits(s_diff, 1) is the natural-log form
    ln_loss = F.binary_cross_entropy_with_logits(s_diff, torch.ones_like(s_diff), reduction="none")
    weighted = (ln_loss / _LN2) * pos * weight
    if reduction == "mean":
        return weighted.sum() / (pos.sum() + eps)
    return weighted.sum()


# --------------------------------------------------------------------------- #
# 3. Contrastive (representation learning)
# --------------------------------------------------------------------------- #
def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive (NT-Xent) loss. ``labels`` here should be a *coarse*
    class (e.g. active / inactive per assay, or activity-decile bucket) so that
    molecules with similar activity are pulled together in embedding space and
    dissimilar ones pushed apart. Pre-train the encoder with this, then attach a
    ranking head -- a strong recipe when true positives are extremely sparse.
    """
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.t() / temperature
    n = z.shape[0]
    logits_mask = ~torch.eye(n, dtype=torch.bool, device=device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & logits_mask

    sim = sim.masked_fill(~logits_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = pos_mask.sum(1).clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / denom
    valid = pos_mask.sum(1) > 0
    if valid.sum() == 0:
        return embeddings.sum() * 0.0
    return -mean_log_prob_pos[valid].mean()


def activity_cliff_contrastive(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    struct_sim: torch.Tensor,
    activity_gap_thresh: float = 1.0,
    sim_thresh: float = 0.7,
    temperature: float = 0.1,
    cliff_weight: float = 2.0,
) -> torch.Tensor:
    """
    Activity-cliff-aware contrastive loss (after Shen et al.).

    struct_sim: [n, n] pairwise Tanimoto similarity of the batch.
    A pair is an "activity cliff" if structurally similar (sim > sim_thresh) but
    with a large activity gap (>activity_gap_thresh). Such pairs are up-weighted
    as hard negatives, forcing the encoder to separate look-alike molecules that
    behave differently -- exactly the discrimination antibiotic SAR demands.
    """
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.t() / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)

    label_gap = (labels.view(-1, 1) - labels.view(1, -1)).abs()
    same_activity = (label_gap <= activity_gap_thresh) & (~eye)
    cliff = (struct_sim > sim_thresh) & (label_gap > activity_gap_thresh) & (~eye)

    weights = torch.ones_like(sim)
    weights[cliff] = cliff_weight  # hard negatives

    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim + weights.log(), dim=1, keepdim=True)
    denom = same_activity.sum(1).clamp(min=1)
    mean_log_prob_pos = (same_activity * log_prob).sum(1) / denom
    valid = same_activity.sum(1) > 0
    if valid.sum() == 0:
        return embeddings.sum() * 0.0
    return -mean_log_prob_pos[valid].mean()
