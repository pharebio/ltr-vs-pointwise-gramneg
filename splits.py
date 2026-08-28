"""
Evaluation splits for LTR models, with a focus on Leave-One-Dataset-Out (LODO).

Motivation (see report, "Evaluation design"):
    Random and even scaffold splits within a pooled dataset overstate how well a
    model will generalize to the *next* screening campaign, because train and test
    share assay conditions, chemical series, and time. LODO holds out an entire
    assay / campaign / library, trains on the rest, and asks: "does the learned
    ordering transfer to a dataset the model has never seen?" That is the question
    that matters for prospective antibiotic screening.

These functions return train/test index arrays; plug them into evaluate.py.
They complement standard random / scaffold (Butina) splits.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def leave_one_dataset_out(dataset_ids: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """
    Yield (held_out_id, train_idx, test_idx) for each unique dataset id.

    dataset_ids: per-row label of which assay / campaign / library the row came
    from, e.g. "assayA", "assayB", "external_validation".
    """
    for held in np.unique(dataset_ids):
        test_idx = np.where(dataset_ids == held)[0]
        train_idx = np.where(dataset_ids != held)[0]
        yield str(held), train_idx, test_idx


def leave_one_scaffold_group_out(
    scaffold_ids: np.ndarray,
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Grouped k-fold where whole Murcko-scaffold clusters are held out together.
    Prevents near-duplicate analogs leaking across the train/test boundary --
    a stricter, more honest test of chemical-series generalization.

    scaffold_ids: per-row scaffold / Butina-cluster id.
    """
    rng = rng or np.random.default_rng(0)
    groups = np.unique(scaffold_ids)
    rng.shuffle(groups)
    folds = np.array_split(groups, n_folds)
    for k, held_groups in enumerate(folds):
        test_mask = np.isin(scaffold_ids, held_groups)
        yield k, np.where(~test_mask)[0], np.where(test_mask)[0]


def temporal_split(
    dates: np.ndarray,
    cutoff,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-based split: train on everything measured before ``cutoff``, test after.
    The most realistic proxy for prospective performance if you have registration
    dates. dates should be comparable (e.g. numpy datetime64 or ordinal ints).
    """
    train_idx = np.where(dates < cutoff)[0]
    test_idx = np.where(dates >= cutoff)[0]
    return train_idx, test_idx
