#!/usr/bin/env python
"""Dataset-overview figure for the public gram-negative benchmark (issue #3).

Two stacked stories about the curated public data:

  A  chemical-space map -- a UMAP of a compound sample from the frozen CheMeleon
     embedding, as small multiples: one mini-panel per in-scope task, that task's
     tested compounds drawn over a grey backdrop of the whole sample, actives
     emphasised. Shows per-task coverage, overlap and diversity.
  B  activity distributions -- per-task KDE of the raw label. The %GI heads share one
     axis; the pMIC head is drawn separately because it is a different (log) scale --
     the scale-incomparability that motivates within-assay ranking.

Only the SAMPLED rows are read from the 4.6 GB embedding parquet (pyarrow predicate
push-down), so peak memory is ~1 GB, not the whole matrix.

Usage:
    micromamba run -n learn2rank python make_dataset_figure.py            # full, UMAP
    micromamba run -n learn2rank python make_dataset_figure.py --sample 2000   # smoke
    micromamba run -n learn2rank python make_dataset_figure.py --reducer pca
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curate_public_data as C

FIG_DIR = Path("results/figures")
EMB_PARQUET = "data/chemeleon_embeddings.parquet"
SMILES = C.SMILES_COL
PMIC = C.PMIC_TASK

# reuse the manuscript-table display names so the two deliverables agree
from make_benchmark_tables import TASK_LABEL  # noqa: E402


def sample_smiles(df: pd.DataFrame, tasks: list[str], n: int, seed: int) -> np.ndarray:
    """Union of every task's actives (rare, must not be sampled away) + a random fill
    to `n`, so the map shows both the bulk chemistry and where the actives sit."""
    rng = np.random.default_rng(seed)
    actives = set()
    for t in tasks:
        v = df[t]
        actives |= set(df.loc[v >= C.hit_threshold(t), SMILES])
    actives = np.array(sorted(actives))
    rest = df.loc[~df[SMILES].isin(actives), SMILES].to_numpy()
    fill = rng.choice(rest, size=max(0, n - len(actives)), replace=False)
    out = np.concatenate([actives, fill])
    return out if len(out) <= n else rng.choice(out, n, replace=False)


def load_embeddings(smiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read ONLY the requested rows from the embedding parquet (predicate push-down)."""
    import pyarrow.parquet as pq

    # pyarrow's "in" predicate wants a list, not a set (a set silently misbehaves).
    # Push-down only skips whole row groups, and a random SMILES sample hits every
    # group, so this still scans the file (~15 GB peak) regardless of sample size --
    # run it when the box is otherwise quiet.
    want = list(pd.unique(smiles))
    tbl = pq.read_table(EMB_PARQUET, filters=[(SMILES, "in", want)])
    edf = tbl.to_pandas()
    # the parquet stores canonicalized_smiles as the pandas index, so to_pandas restores
    # it there, not as a column -- only set_index when it is still a column
    if edf.index.name != SMILES:
        edf = edf.set_index(SMILES)
    edf = edf[~edf.index.duplicated()]
    keep = [s for s in want if s in edf.index]  # a few SMILES may lack an embedding
    X = edf.loc[keep].to_numpy(dtype=np.float32)
    return X, np.array(keep)


def reduce_2d(X: np.ndarray, reducer: str, seed: int) -> np.ndarray:
    if reducer == "umap":
        import umap

        return umap.UMAP(n_neighbors=30, min_dist=0.25, metric="euclidean",
                         random_state=seed).fit_transform(X)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(X)


def fig_dataset(sample: int = 40000, reducer: str = "umap", seed: int = 0,
                out: Path | None = None) -> None:
    out = out or FIG_DIR / f"dataset_overview_{reducer}.png"
    df = pd.read_csv(C.OUT_CSV, low_memory=False)
    pub = C.public_task_columns(df)
    in_scope = C.eligible_tasks(df, pub)[0]           # the 10 tasks that make a scope

    smi = sample_smiles(df, in_scope, sample, seed)
    X, smi = load_embeddings(smi)
    print(f"embedded {len(smi):,} / {len(smi):,} sampled compounds; running {reducer}…",
          flush=True)
    xy = reduce_2d(X, reducer, seed)
    pos = pd.DataFrame(xy, columns=["x", "y"], index=smi)
    # align each task's per-compound label onto the sampled/embedded rows
    act = df.set_index(SMILES).reindex(smi)

    gi_tasks = [t for t in in_scope if t != PMIC]
    ncol = 4
    nrow = int(np.ceil(len(in_scope) / ncol))
    fig = plt.figure(figsize=(4 * ncol, 3.4 * nrow + 4.2))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[*([1] * nrow), 1.5])

    # ----- Panel A: per-task chemical-space maps ----------------------------------- #
    for i, t in enumerate(in_scope):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        ax.scatter(pos.x, pos.y, s=2, c="#dddddd", linewidths=0, rasterized=True)
        v = act[t].to_numpy()
        tested = np.isfinite(v)
        thr = C.hit_threshold(t)
        hit = tested & (v >= thr)
        ax.scatter(pos.x[tested & ~hit], pos.y[tested & ~hit], s=3, c="#4C78A8",
                   linewidths=0, alpha=0.5, rasterized=True)
        ax.scatter(pos.x[hit], pos.y[hit], s=12, c="#E45756", edgecolor="white",
                   linewidths=0.3, zorder=3)
        # subtitle counts are the TRUE full-dataset totals (== Table 2), NOT the sample:
        # sample_smiles includes every active for visibility, so sample counts would
        # imply absurd hit rates (e.g. 96% for ΔtolC vs the real 0.43%).
        full = df[t]
        n_tested, n_act = int(full.notna().sum()), C.n_actives(df, t)
        ax.set_title(f"{TASK_LABEL.get(t, t)}\n{n_tested:,} tested · {n_act:,} active "
                     f"({100 * n_act / n_tested:.2f}%)", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    # legend for panel A
    fig.text(0.5, 0.955, f"A · Chemical space ({reducer.upper()} of CheMeleon embeddings; "
             f"{len(smi):,}-compound sample) — grey: all sampled · blue: tested inactive · "
             "red: active", ha="center", fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.938, "counts are true dataset totals; the sample includes every active "
             "for visibility, so red density over-represents true hit rate", ha="center",
             fontsize=8.5, style="italic", color="#555555")

    # ----- Panel B: activity distributions ----------------------------------------- #
    axb = fig.add_subplot(gs[nrow, : ncol - 1])
    for t in gi_tasks:
        v = df[t].dropna().to_numpy()
        axb.hist(v, bins=60, range=(-40, 130), histtype="step", linewidth=1.4,
                 density=True, label=TASK_LABEL.get(t, t))
    axb.axvline(C.GI_THRESHOLD, ls="--", color="#E45756", lw=1.2)
    axb.text(C.GI_THRESHOLD + 1, axb.get_ylim()[1] * 0.9, "hit ≥ 80% GI", color="#E45756",
             fontsize=8)
    axb.set_xlabel("% growth inhibition")
    axb.set_ylabel("density")
    axb.set_title("B · Activity distribution — %GI heads (shared scale)", fontsize=11,
                  fontweight="bold", loc="left")
    axb.legend(fontsize=6.5, ncol=2, frameon=False)

    axp = fig.add_subplot(gs[nrow, ncol - 1])
    vp = df[PMIC].dropna().to_numpy()
    axp.hist(vp, bins=50, color="#54A24B", density=True)
    axp.axvline(C.PMIC_THRESHOLD, ls="--", color="#E45756", lw=1.2)
    axp.set_xlabel("pMIC (log scale)")
    axp.set_title("pMIC head — separate axis", fontsize=9.5, fontweight="bold")

    fig.suptitle("Public gram-negative benchmark — dataset overview", fontsize=14,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=40000)
    p.add_argument("--reducer", choices=["umap", "pca"], default="umap")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    fig_dataset(a.sample, a.reducer, a.seed)
