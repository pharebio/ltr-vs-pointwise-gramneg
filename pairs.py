"""
Pair / group construction for pairwise and listwise learning-to-rank.

The central idea (see report, "Cross-assay integration"): NEVER compare raw
activity values across assays. Instead, only form training comparisons *within*
a homogeneous context -- a single assay, a single plate/batch, or a single
congeneric series. The model then learns a shared latent "activity" scale that
is consistent with every within-context ordering simultaneously. This is exactly
the surrogate-ordering trick of Sun et al. (DPDI) and sidesteps normalization and
assay-aggregation entirely.

This module handles three job types:
    * build_within_group_pairs   -> all ordered pairs inside each group (assay)
    * sample_pairs_imbalanced    -> active-anchored sampling + hard negatives for
                                    extreme sparsity (don't drown in inactive-inactive pairs)
    * handle_censored            -> emit pairs only when the order is unambiguous
                                    given bounded/qualified values ("<", ">")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thresholds import ThresholdSpec, resolve_threshold


@dataclass
class Pair:
    i: int          # row index of first molecule
    j: int          # row index of second molecule (i should outrank j: label_i > label_j)
    group: str      # assay / batch / series identifier
    weight: float = 1.0


def build_within_group_pairs(
    labels: np.ndarray,
    groups: np.ndarray,
    min_gap: float = 0.0,
    max_pairs_per_group: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Pair]:
    """
    Enumerate ordered pairs within each group where labels differ by > min_gap.

    min_gap acts as a "demilitarized zone" (Fralish et al. 2024): skip pairs whose
    activity difference is smaller than assay noise, so the model isn't trained to
    resolve differences that aren't real. Set it to ~1-2x the assay's std of
    replicate error.
    """
    rng = rng or np.random.default_rng(0)
    pairs: list[Pair] = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y = labels[idx]
        valid = ~np.isnan(y)
        idx, y = idx[valid], y[valid]
        gp: list[Pair] = []
        for a in range(len(idx)):
            for b in range(len(idx)):
                if a == b:
                    continue
                if y[a] - y[b] > min_gap:
                    gp.append(Pair(int(idx[a]), int(idx[b]), str(g)))
        if max_pairs_per_group and len(gp) > max_pairs_per_group:
            gp = list(rng.choice(gp, size=max_pairs_per_group, replace=False))
        pairs.extend(gp)
    return pairs


def sample_pairs_imbalanced(
    labels: np.ndarray,
    groups: np.ndarray,
    active_threshold: ThresholdSpec,
    n_neg_per_active: int = 20,
    n_pos_per_active: int = 0,
    pos_weight: float = 1.0,
    hard_negative_scores: np.ndarray | None = None,
    min_gap: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[Pair]:
    """
    Sparsity-aware pair sampling. Two kinds of within-group pairs are formed:

    * active-over-inactive (always): for each *active* molecule (label >= threshold)
      sample ``n_neg_per_active`` negatives (label < threshold) from the SAME group.
      This keeps every batch informative even when the hit rate is <1%, instead of
      being dominated by uninformative inactive-inactive comparisons. Raise
      ``n_neg_per_active`` for more training volume.
    * active-over-active (opt-in, ``n_pos_per_active > 0``): for each active, sample
      up to ``n_pos_per_active`` *other actives with strictly lower label* and order
      them by their continuous value (e.g. %GI). This adds top-of-list ordering
      signal -- distinguishing strong hits from weak hits -- which pure
      active-vs-random-inactive pairs never provide (those saturate/collapse the loss
      because random inactives are trivially separable). Only the sign of the label
      difference is used, so no cross-assay scale is needed. ``pos_weight`` scales the
      RankNet contribution of these harder pairs (see ``ranknet_loss(weight=...)``).

    active_threshold: a scalar applied to every group, OR a per-group map / callable
    (see thresholds.resolve_threshold) so mixed endpoints -- e.g. %GI groups at 80
    and a pMIC group at 7 -- each anchor on their own hit cutoff. Groups whose
    threshold resolves to None are skipped.

    min_gap: skip pairs whose label difference is <= min_gap (a "demilitarized zone"
    against assay noise). Applies to both pair kinds; ties (gap 0) are always skipped.

    hard_negative_scores: optional current model scores. If given, negatives are
    drawn preferentially from the highest-scoring inactives (the model's current
    false positives) -- hard-negative mining, which sharpens early recognition.
    """
    rng = rng or np.random.default_rng(0)
    pairs: list[Pair] = []
    for g in np.unique(groups):
        thr = resolve_threshold(active_threshold, g)
        if thr is None:
            continue
        idx = np.where(groups == g)[0]
        y = labels[idx]
        valid = ~np.isnan(y)
        idx, y = idx[valid], y[valid]
        act = idx[y >= thr]
        neg = idx[y < thr]

        # --- active-over-inactive ---
        if len(act) and len(neg):
            if hard_negative_scores is not None:
                order = np.argsort(-hard_negative_scores[neg])
                neg_ranked = neg[order]
            for a in act:
                if hard_negative_scores is not None:
                    chosen = neg_ranked[:n_neg_per_active]
                else:
                    chosen = rng.choice(neg, size=min(n_neg_per_active, len(neg)), replace=False)
                for b in chosen:
                    if labels[a] - labels[b] > min_gap:
                        pairs.append(Pair(int(a), int(b), str(g)))

        # --- active-over-active (rank hits by their continuous value) ---
        if n_pos_per_active > 0 and len(act) >= 2:
            for a in act:
                lower = act[labels[act] < labels[a] - min_gap]  # a strictly outranks these
                if len(lower) == 0:
                    continue
                chosen = rng.choice(lower, size=min(n_pos_per_active, len(lower)), replace=False)
                for b in chosen:
                    pairs.append(Pair(int(a), int(b), str(g), weight=pos_weight))
    return pairs


def handle_censored(
    values: np.ndarray,
    qualifiers: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator | None = None,
) -> list[Pair]:
    """
    Build pairs from bounded / qualified measurements, following the DeltaClassifier
    logic (Fralish et al. 2024, RSC Med Chem): a lot of screening data is censored
    ("MIC > 64", "no inhibition at max dose"). Regression discards these, but their
    *order* relative to a measured active is often certain and highly informative.

    qualifiers: array of {"=", "<", ">"} aligned to values.
    Emits a pair (i over j) only when the inequality is guaranteed:
        - i="=" or ">" larger, j="=" or "<" smaller, and bounds don't overlap.
    """
    rng = rng or np.random.default_rng(0)
    pairs: list[Pair] = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for a in idx:
            for b in idx:
                if a == b:
                    continue
                if _certain_greater(values[a], qualifiers[a], values[b], qualifiers[b]):
                    pairs.append(Pair(int(a), int(b), str(g)))
    return pairs


def _certain_greater(va, qa, vb, qb) -> bool:
    """True iff a is certainly more active than b given qualified values."""
    if np.isnan(va) or np.isnan(vb):
        return False
    # lower bound of a
    lo_a = va if qa in ("=", ">") else -np.inf
    # upper bound of b
    hi_b = vb if qb in ("=", "<") else np.inf
    return lo_a > hi_b


def pairs_to_arrays(pairs: list[Pair]):
    """Convert a list of Pair to parallel numpy arrays for batching."""
    i = np.array([p.i for p in pairs], dtype=np.int64)
    j = np.array([p.j for p in pairs], dtype=np.int64)
    w = np.array([p.weight for p in pairs], dtype=np.float32)
    return i, j, w
