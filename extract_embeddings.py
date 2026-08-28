#!/usr/bin/env python
"""
Extract frozen **CheMeleon** foundation embeddings (2048-dim) for the benchmark
compounds, keyed by canonicalized_smiles -> parquet. This is the `chemeleon` feature
table the harness consumes (a drop-in alternative to the RDKit descriptors), and it
tests whether a *learned* molecular representation lifts ranking over RDKit.

CheMeleon is a pretrained, self-supervised foundation encoder -- it has never seen
bioactivity labels, so its embeddings carry no label leakage under LODO/CV.

Loading follows chemprop's `chemeleon_foundation_finetuning` example: weights at
``~/.chemprop/chemeleon_mp.pt`` (Zenodo 15460715), a BondMessagePassing +
MeanAggregation encoder; the frozen embedding is ``MPNN.fingerprint(bmg, V_d, X_d)``.
Runs via the ChemProp Python API (needs chemprop + rdkit + torch installed):

    # full run (GPU if available)
    python extract_embeddings.py
    # sanity check (few molecules, CPU)
    python extract_embeddings.py --limit 8 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CHEMELEON_CKPT = os.path.expanduser("~/.chemprop/chemeleon_mp.pt")
SMILES_COL = "canonicalized_smiles"
LEARN2RANK_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=CHEMELEON_CKPT, help="CheMeleon MP checkpoint")
    p.add_argument("--smiles-csv", default=None,
                   help="CSV with a canonicalized_smiles column (default: the public wide df)")
    p.add_argument("--out",
                   default=os.path.join(LEARN2RANK_DIR, "data", "chemeleon_embeddings.parquet"))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="embed only first N SMILES (sanity)")
    return p.parse_args()


def load_chemeleon(ckpt_path: str, device):
    """Build a frozen CheMeleon encoder (MPNN with a dummy head; only fingerprint used)."""
    import torch
    from chemprop import models, nn

    ck = torch.load(ckpt_path, weights_only=True)
    mp = nn.BondMessagePassing(**ck["hyper_parameters"])
    mp.load_state_dict(ck["state_dict"])
    ffn = nn.RegressionFFN(input_dim=mp.output_dim)  # unused; fingerprint() stops before it
    mpnn = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    mpnn = mpnn.to(device).eval()
    for prm in mpnn.parameters():
        prm.requires_grad_(False)
    return mpnn, int(mp.output_dim)


def main():
    a = parse_args()
    if LEARN2RANK_DIR not in sys.path:
        sys.path.insert(0, LEARN2RANK_DIR)

    import torch
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from rdkit import Chem

    device = torch.device(a.device if a.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- SMILES ----
    if a.smiles_csv:
        smiles = pd.read_csv(a.smiles_csv, usecols=[SMILES_COL])[SMILES_COL].tolist()
    else:
        import data_access
        smiles = pd.read_csv(data_access.wide_csv(),
                             usecols=[SMILES_COL])[SMILES_COL].tolist()
    smiles = list(dict.fromkeys(smiles))  # unique, order-preserving
    if a.limit:
        smiles = smiles[:a.limit]
    print(f"embedding {len(smiles):,} unique SMILES on {device}", flush=True)

    mpnn, dim = load_chemeleon(a.ckpt, device)
    print(f"CheMeleon loaded from {a.ckpt} (embedding dim={dim})", flush=True)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [i for i, m in enumerate(mols) if m is not None]
    datapoints = [MoleculeDatapoint(mol=mols[i], x_d=None) for i in valid]
    dset = MoleculeDataset(datapoints)
    loader = build_dataloader(dset, batch_size=min(a.batch_size, max(1, len(datapoints))),
                              shuffle=False, num_workers=0)

    embs, t0, done = [], time.time(), 0
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            for attr in ("V", "E", "edge_index", "rev_edge_index", "batch"):
                t = getattr(bmg, attr, None)
                if t is not None:
                    setattr(bmg, attr, t.to(device))
            V_d = batch.V_d.to(device) if batch.V_d is not None else None
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            Z = mpnn.fingerprint(bmg, V_d=V_d, X_d=X_d).detach().cpu().numpy()
            embs.append(np.atleast_2d(Z))
            done += len(Z)
            if done % (a.batch_size * 20) < a.batch_size:
                print(f"  {done:,}/{len(valid):,}  ({done / (time.time() - t0):.0f}/s)", flush=True)

    X = np.vstack(embs).astype(np.float32)
    valid_smiles = [smiles[i] for i in valid]
    print(f"embedded {X.shape[0]:,} x {X.shape[1]} "
          f"({len(smiles) - len(valid)} RDKit failures) in {time.time()-t0:.0f}s", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=[f"chemeleon_{i}" for i in range(X.shape[1])])
    df.insert(0, SMILES_COL, valid_smiles)
    df.set_index(SMILES_COL).to_parquet(out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
