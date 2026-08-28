#!/usr/bin/env python
"""Compute the 200-dim CDF-normalized RDKit 2D descriptor table for the benchmark.

Regenerates the ``rdkit`` feature table the harness consumes (see features.py), keyed by
``canonicalized_smiles``. Uses descriptastorus' RDKit2DNormalized generator: the standard
200 RDKit 2D descriptors, CDF-normalized to [0, 1]. Compounds RDKit cannot process are
dropped (features.align_features handles their absence via its ``found`` mask).

    python compute_rdkit_features.py                 # -> data/rdkit_features.parquet
    python compute_rdkit_features.py --n-workers 16
    python compute_rdkit_features.py --limit 8       # quick sanity check
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

SMILES_COL = "canonicalized_smiles"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WIDE = os.path.join(_HERE, "data", "public_gram_negative_wide_df.csv.gz")
DEFAULT_OUT = os.path.join(_HERE, "data", "rdkit_features.parquet")

_GEN = None


def _init() -> None:
    global _GEN
    from descriptastorus.descriptors import rdNormalizedDescriptors
    _GEN = rdNormalizedDescriptors.RDKit2DNormalized()


def _process(smiles: str):
    res = _GEN.process(smiles)              # [ok, f1, ..., f200]
    if not res or not res[0]:
        return None
    v = np.asarray(res[1:], dtype=np.float32)
    return None if not np.all(np.isfinite(v)) else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wide-csv", default=DEFAULT_WIDE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=None, help="process only first N SMILES")
    a = ap.parse_args()

    from descriptastorus.descriptors import rdNormalizedDescriptors
    cols = ["rdkit_" + n for n, _ in rdNormalizedDescriptors.RDKit2DNormalized().columns]

    smiles = pd.read_csv(a.wide_csv, usecols=[SMILES_COL])[SMILES_COL].dropna().unique().tolist()
    if a.limit:
        smiles = smiles[:a.limit]
    print(f"computing {len(cols)} RDKit descriptors for {len(smiles):,} unique SMILES "
          f"on {a.n_workers} worker(s)...", flush=True)

    with Pool(a.n_workers, initializer=_init) as pool:
        results = pool.map(_process, smiles, chunksize=256)

    keep = [(s, r) for s, r in zip(smiles, results, strict=True) if r is not None]
    n_fail = len(smiles) - len(keep)
    X = np.vstack([r for _, r in keep]).astype(np.float32)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, SMILES_COL, [s for s, _ in keep])
    df.set_index(SMILES_COL).to_parquet(a.out)
    print(f"wrote {a.out}  ({X.shape[0]:,} x {X.shape[1]}; {n_fail} RDKit failures dropped)",
          flush=True)


if __name__ == "__main__":
    main()
