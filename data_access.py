"""Local-first access to the public benchmark's data files.

All inputs live under ``data/``. The combined public dataframe and the fold assignments
are shipped with the repo; the feature tables are large and are regenerated locally:

  data/public_gram_negative_wide_df.csv.gz  combined public activity matrix   (shipped, gzip)
  data/public_1x5_split.csv.gz              5-fold chemistry-holdout split    (shipped, gzip)
  data/public_5x5_split.csv.gz              5x5 repeated-CV splits            (shipped, gzip)
  data/public_lodo_splits_meta.json         leave-one-source-out fold order   (shipped)
  data/rdkit_features.parquet               200-dim RDKit descriptors  -> compute_rdkit_features.py
  data/chemeleon_embeddings.parquet         2048-dim CheMeleon embeddings -> extract_embeddings.py

CSV inputs are gzip-compressed to stay well under repo file-size limits; pandas reads
``.csv.gz`` transparently. The LODO folds are defined by held-out source, so only the
small meta (fold order per scope) is needed at run time.

Each accessor returns a path and raises a clear error naming the file and how to produce
it, so a missing feature table points the user at the regeneration script.
"""
from __future__ import annotations

import os

_REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_REPO, "data")

LOCAL_WIDE = os.path.join(DATA, "public_gram_negative_wide_df.csv.gz")
LOCAL_SPLIT_1X5 = os.path.join(DATA, "public_1x5_split.csv.gz")
LOCAL_SPLIT_5X5 = os.path.join(DATA, "public_5x5_split.csv.gz")
LOCAL_LODO = os.path.join(DATA, "public_lodo_splits_meta.json")
LOCAL_FEATURES = os.path.join(DATA, "rdkit_features.parquet")
LOCAL_CHEMELEON_EMB = os.path.join(DATA, "chemeleon_embeddings.parquet")


def _require(path: str, how: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. {how}")
    return path


def wide_csv() -> str:
    return _require(LOCAL_WIDE, "It ships with the repo under data/.")


def split_1x5_csv() -> str:
    return _require(LOCAL_SPLIT_1X5, "It ships with the repo under data/.")


def features_parquet() -> str:
    return _require(LOCAL_FEATURES,
                    "Regenerate it with:  python compute_rdkit_features.py")


def chemeleon_parquet() -> str:
    return _require(LOCAL_CHEMELEON_EMB,
                    "Regenerate it with:  python extract_embeddings.py")
