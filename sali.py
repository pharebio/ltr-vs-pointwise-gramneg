"""
SALI (Structure-Activity Landscape Index) pair selection for the D-MPNN RankNet ranker.

SALI (Guha & Van Drie 2008, J. Chem. Inf. Model. 48:646-658): for a pair (i, j),

    SALI_ij = |A_i - A_j| / (1 - sim(i, j))

with A the activity and sim a structural similarity in [0, 1]. HIGH SALI = similar
structure but very different activity = an *activity cliff* -- exactly the deceptive
comparison that pointwise regression tends to smooth over and that a ranking objective
should, in principle, get right.

We use SALI as a THIRD negative bucket for RankNet, alongside random and model-score
"hard" negatives (`ranking_chemprop.fit_pairwise`):

  * random      -- coverage / regularization
  * model-hard  -- inactives the model currently scores high (its false positives);
                   ADAPTIVE, re-mined every `mine_every` epochs
  * SALI-hard   -- inactives structurally near an active (intrinsic cliffs);
                   STATIC, precomputed once (this module)

For active->inactive RankNet pairs the activity gap is ~constant (active vs inactive),
so SALI collapses to "pick the structurally CLOSEST inactive to each active" -- which
also sidesteps the 1/(1 - sim) singularity at near-duplicates. We therefore rank
negatives by structural proximity to the actives (Morgan-Tanimoto), not by raw SALI.

Two granularities:
  * per-assay proxy (`precompute_sali_ranks`): each inactive scored by its MAX Tanimoto
    to ANY active in the assay -> one ranked list of inactives per assay. A drop-in
    analogue of the model-score `hard_rank[g]` list, so it plugs into `fit_pairwise`
    with the same top-of-pool sampling. O(n_act * n_neg) per assay.
  * per-active (future): each active's own nearest inactives -- truer to SALI but keyed
    by active index; heavier to thread through the batch loop. Not needed for the first
    cut (the per-assay proxy already isolates the cliff-prone inactives).

Pure CPU (RDKit fingerprints + BulkTanimotoSimilarity). No torch, no GPU.

Run the diagnostic on the E. coli WT subsample:
    micromamba run -n learn2rank python sali.py --group e_coli_wt --neg-pool 5000 --fig
"""
from __future__ import annotations

import numpy as np


def morgan_fps(smiles, radius: int = 2, n_bits: int = 2048):
    """Morgan bit-vector fingerprints for a list of SMILES, deduplicated by SMILES string
    (the same compound recurs across assays). Returns a list aligned to ``smiles``;
    unparseable SMILES map to ``None``.

    NB radius (Morgan/RDKit) = diameter/2 (ECFP), so the default ``radius=2`` is **ECFP4**
    -- the standard choice for Tanimoto similarity / activity-cliff work. ``radius=3`` is
    ECFP6."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    cache: dict = {}
    fps = []
    for s in smiles:
        if s not in cache:
            m = Chem.MolFromSmiles(s)
            cache[s] = gen.GetFingerprint(m) if m is not None else None
        fps.append(cache[s])
    return fps


def max_sim_to_actives(neg_fps, act_fps) -> np.ndarray:
    """For each negative fingerprint, the MAX Tanimoto to any active fingerprint.
    Returns an array aligned to ``neg_fps`` (0.0 where a fp is missing or no actives)."""
    from rdkit import DataStructs

    out = np.zeros(len(neg_fps), dtype=np.float32)
    if not act_fps:
        return out
    for i, fp in enumerate(neg_fps):
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, act_fps)
        if sims:
            out[i] = max(sims)
    return out


def nearest_active(neg_fps, act_fps):
    """Like ``max_sim_to_actives`` but also returns, per negative, the LOCAL index (into
    ``act_fps``) of the most-similar active. Used by the diagnostic to surface concrete
    (inactive, nearest-active) activity-cliff pairs. Returns (sims, argmax_local)."""
    from rdkit import DataStructs

    sims = np.zeros(len(neg_fps), dtype=np.float32)
    arg = np.full(len(neg_fps), -1, dtype=np.int64)
    if not act_fps:
        return sims, arg
    for i, fp in enumerate(neg_fps):
        if fp is None:
            continue
        s = DataStructs.BulkTanimotoSimilarity(fp, act_fps)
        if s:
            a = int(np.argmax(s))
            sims[i] = s[a]
            arg[i] = a
    return sims, arg


def precompute_sali_ranks(smiles, labels, groups, thr_map, sim_lo: float = 0.0,
                          sim_hi: float = 1.0, radius: int = 2, n_bits: int = 2048,
                          fps=None, verbose: bool = False) -> dict:
    """Per-assay "hit-like inactive" ranking for SALI-hard negative mining.

    Returns ``{group: neg_global_idx}`` where ``neg_global_idx`` is that assay's inactive
    row indices sorted by MAX Tanimoto to any active DESC (most cliff-prone first) -- the
    structural analogue of ``fit_pairwise``'s model-score ``hard_rank[g]``. Computed ONCE
    (structural landscape is static; unlike model-score mining it needn't be refreshed).

    ``[sim_lo, sim_hi]`` is a **semi-hard band**: only inactives whose max-Tanimoto to an
    active falls in the band are kept. ``sim_lo`` drops structurally-distant inactives that
    aren't cliffs (they'd just be random); ``sim_hi`` drops the sim->1 tail -- genuine but
    rare ECFP collisions with opposite labels that a fingerprint (and likely the D-MPNN)
    can't separate, so ranking them confidently injects near-noise gradient. Assays with no
    banded inactive are omitted (the trainer then falls back to random for that assay).

    Compute fingerprints over the SAME (already subsampled) training arrays that
    ``fit_pairwise`` will sample negatives from, so the ranking indexes the live pool.
    """
    labels = np.asarray(labels, dtype=float)
    groups = np.asarray(groups)
    if fps is None:
        fps = morgan_fps(smiles, radius, n_bits)
    ranks: dict = {}
    for g in np.unique(groups):
        thr = thr_map[g]
        gi = np.where(groups == g)[0]
        y = labels[gi]
        act = gi[y >= thr]
        neg = gi[y < thr]
        if len(act) == 0 or len(neg) == 0:
            continue
        act_fps = [fps[i] for i in act if fps[i] is not None]
        neg_fps = [fps[i] for i in neg]
        sims = max_sim_to_actives(neg_fps, act_fps)
        band = (sims >= sim_lo) & (sims <= sim_hi)
        band_neg, band_sims = neg[band], sims[band]
        if len(band_neg) == 0:
            continue
        ranks[g] = band_neg[np.argsort(-band_sims)]
        if verbose:
            print(f"[sali] {g}: {len(act)} act, {len(neg)} neg, "
                  f"{len(band_neg)} in band [{sim_lo:.2f},{sim_hi:.2f}] "
                  f"(max-sim p99={np.quantile(sims, 0.99):.3f})", flush=True)
    return ranks


# --------------------------------------------------------------------------- #
# CPU diagnostic: characterize the SALI landscape on one organism group before
# wiring `--sali-neg-frac` into training. Confirms (a) cliff-prone inactives exist,
# (b) the SALI bucket is a DIFFERENT population from random negatives.
# --------------------------------------------------------------------------- #
def _diagnostic(args):
    import json
    from pathlib import Path

    import pandas as pd

    from chemprop_ranker_lodo import build_group_long, subsample_train
    from organism_groups import load_organism_groups

    groups_map = load_organism_groups()
    if args.group not in groups_map:
        raise SystemExit(f"unknown group {args.group!r}; choices: {list(groups_map)}")

    print(f"=== SALI diagnostic: group={args.group} "
          f"(radius={args.radius}, n_bits={args.n_bits}, neg-pool={args.neg_pool}) ===",
          flush=True)
    wide = pd.read_csv(args.wide_csv)
    data = build_group_long(wide, groups_map[args.group])
    smi, y, grp, thr = data["smiles"], data["y"], data["task"], data["threshold"]

    # same training subsample the ranker sees: all actives + neg_pool inactives/assay
    keep = subsample_train(smi, y, grp, thr, args.neg_pool, args.seed)
    smi, y, grp, thr = smi[keep], y[keep], grp[keep], thr[keep]
    thr_map = {g: float(thr[grp == g][0]) for g in np.unique(grp)}
    print(f"subsample: {len(y):,} rows over {len(thr_map)} assays", flush=True)

    fps = morgan_fps(smi, args.radius, args.n_bits)
    rng = np.random.default_rng(args.seed)

    rows, hist = [], {}
    for g in np.unique(grp):
        gi = np.where(grp == g)[0]
        t = thr_map[g]
        act = gi[y[gi] >= t]
        neg = gi[y[gi] < t]
        if len(act) == 0 or len(neg) == 0:
            continue
        act_fps = [fps[i] for i in act if fps[i] is not None]
        neg_fps = [fps[i] for i in neg]
        sims, arg = nearest_active(neg_fps, act_fps)
        hist[g] = sims

        order = np.argsort(-sims)          # local into neg / sims / arg
        k = max(1, int(round(0.01 * len(neg))))   # top 1% "most hit-like" inactives
        rand_local = rng.choice(len(neg), size=min(k, len(neg)), replace=False)
        row = {
            "assay": g, "n_act": int(len(act)), "n_neg": int(len(neg)),
            "sim_median": float(np.median(sims)),
            "sim_p90": float(np.quantile(sims, 0.90)),
            "sim_p99": float(np.quantile(sims, 0.99)),
            "sim_max": float(sims.max()),
            "n_ge_0.5": int((sims >= 0.5).sum()),
            "n_ge_0.7": int((sims >= 0.7).sum()),
            "sali_topk_meansim": float(sims[order[:k]].mean()),
            "random_k_meansim": float(sims[rand_local].mean()),
        }
        rows.append(row)

        # concrete activity cliffs: the 3 most hit-like inactives + their nearest active
        cliffs = []
        for loc in order[:3]:
            a_loc = arg[loc]
            if a_loc < 0:
                continue
            cliffs.append({
                "inactive_smiles": smi[neg[loc]], "inactive_y": round(float(y[neg[loc]]), 1),
                "active_smiles": smi[act[a_loc]], "active_y": round(float(y[act[a_loc]]), 1),
                "tanimoto": round(float(sims[loc]), 3),
            })
        row["example_cliffs"] = cliffs

    # ---- report ----
    print(f"\n{'assay':30s} {'act':>4s} {'neg':>5s} {'med':>5s} {'p90':>5s} "
          f"{'p99':>5s} {'max':>5s} {'>=.5':>5s} {'>=.7':>5s} {'SALIk':>6s} {'randk':>6s}",
          flush=True)
    print("-" * 100, flush=True)
    for r in rows:
        print(f"{r['assay']:30s} {r['n_act']:4d} {r['n_neg']:5d} "
              f"{r['sim_median']:5.2f} {r['sim_p90']:5.2f} {r['sim_p99']:5.2f} "
              f"{r['sim_max']:5.2f} {r['n_ge_0.5']:5d} {r['n_ge_0.7']:5d} "
              f"{r['sali_topk_meansim']:6.2f} {r['random_k_meansim']:6.2f}", flush=True)

    print("\n-- example activity cliffs (most hit-like inactive per assay) --", flush=True)
    for r in rows:
        if r["example_cliffs"]:
            c = r["example_cliffs"][0]
            print(f"  [{r['assay']}] Tanimoto={c['tanimoto']}  "
                  f"active(y={c['active_y']})  vs  inactive(y={c['inactive_y']})", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.group}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump({"group": args.group, "radius": args.radius, "n_bits": args.n_bits,
                   "neg_pool": args.neg_pool, "rows": rows}, fh, indent=2)
    print(f"\nwrote {summary_path}", flush=True)

    if args.fig:
        _plot(hist, out_dir / f"{args.group}_simhist.png", args.group)


def _plot(hist, path, group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assays = list(hist)
    n = len(assays)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.4 * nrow), squeeze=False)
    bins = np.linspace(0, 1, 41)
    for ax, g in zip(axes.ravel(), assays, strict=False):
        ax.hist(hist[g], bins=bins, color="#1C7293", alpha=0.85)
        ax.axvline(0.5, color="#C0392B", ls="--", lw=1)
        ax.set_title(g, fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("max Tanimoto to any active", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"SALI proxy: inactive structural proximity to actives — {group}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}", flush=True)


def _main():
    import argparse

    p = argparse.ArgumentParser(description="SALI landscape diagnostic (CPU)")
    p.add_argument("--group", default="e_coli_wt")
    p.add_argument("--neg-pool", type=int, default=5000,
                   help="inactives kept per assay (mirror the training subsample)")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wide-csv", default="data/public_gram_negative_wide_df.csv.gz")
    p.add_argument("--out-dir", default="results/sali_prototype")
    p.add_argument("--fig", action="store_true", help="save a per-assay sim histogram")
    _diagnostic(p.parse_args())


if __name__ == "__main__":
    _main()
