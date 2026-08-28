#!/usr/bin/env python
"""Heatmap of BEDROC at an arbitrary alpha, recomputed from the persisted OOF vectors.

Reviewer question: are the pointwise-vs-ranking patterns an artifact of BEDROC's alpha?
alpha=20 concentrates the score on ~the top 8% of the ranked list; alpha=100 pushes the
selective pressure to ~the top 1-2%. BEDROC at any alpha is just a re-scoring of the
out-of-fold predictions (no retraining), so this reads results/benchmark/<run_id>_oof.npz
and recomputes each cell's within-assay BEDROC(alpha), unweighted-averaged over the scope's
endpoints -- exactly the aggregate the stored alpha=20 number uses (validated identical when
alpha=20). Cells whose OOF is not on disk are greyed (the early RDKit-CV runs predate
OOF-saving; regenerate with run_benchmark.py --features rdkit --split cv --force).

Mirrors make_benchmark_figures.fig_heatmap layout (CPU arms x representation, per
scope x split), so it sits beside the alpha=20 heatmap.

    micromamba run -n learn2rank python make_alpha_heatmap.py --alpha 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from bootstrap_ci import _load, _scored_assays
from make_benchmark_figures import (
    C_POINT,
    C_RANK,
    HEATMAP_COLS,
    HEATMAP_ROWS,
    INK,
    SCOPES,
    SPLITS,
    load,
)
from metrics import bedroc

FIG_DIR = Path("results/figures")
OOF_DIR = Path("results/benchmark")
# small shipped cache of per-config BEDROC(alpha) values, so the alpha heatmap regenerates
# without the large OOF archives (keyed by "<run_id>|<alpha>").
CACHE = OOF_DIR / "alpha_heatmap_cache.json"


def agg_bedroc(run_id: str, alpha: float, cache: dict | None = None) -> float:
    """Unweighted mean over scored assays of within-assay BEDROC(alpha).

    Recomputes from the config's OOF when it is on disk (updating ``cache``); otherwise
    falls back to the shipped cache. Returns NaN if neither is available.
    """
    if not run_id:
        return float("nan")
    key = f"{run_id}|{alpha:g}"
    p = OOF_DIR / f"{run_id}_oof.npz"
    if not p.exists():
        return float(cache[key]) if cache and key in cache else float("nan")
    oof, y, task, thr = _load(str(p))
    ybin = (y >= thr).astype(int)
    assays, idx = _scored_assays(ybin, task)
    vals = [bedroc(ybin[idx[a]], oof[idx[a]], alpha=alpha) for a in assays]
    val = float(np.mean(vals)) if vals else float("nan")
    if cache is not None and not np.isnan(val):
        cache[key] = val
    return val


def fig_alpha_heatmap(alpha: float, out: Path | None = None) -> None:
    data = load()
    out = out or FIG_DIR / f"benchmark_heatmap_bedroc_a{int(alpha)}.png"
    title_metric = f"BEDROC (α={int(alpha)})"

    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    # recompute every cell up front so the shared colour scale is anchored to alpha data
    M = {}
    for scope, _ in SCOPES:
        for split, _ in SPLITS:
            for i, (arch, loss, _) in enumerate(HEATMAP_ROWS):
                for j, (feat, _) in enumerate(HEATMAP_COLS):
                    v = data.get((scope, split, arch, feat, loss))
                    M[(scope, split, i, j)] = agg_bedroc(v["run_id"], alpha, cache) if v else np.nan
    CACHE.write_text(json.dumps(cache, indent=1))
    vals = [x for x in M.values() if not np.isnan(x)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)

    fig, axes = plt.subplots(len(SPLITS), len(SCOPES), figsize=(16, 9.5))
    im = None
    n_missing = 0
    for ri, (split, split_lbl) in enumerate(SPLITS):
        for ci, (scope, scope_lbl) in enumerate(SCOPES):
            ax = axes[ri, ci]
            A = np.full((len(HEATMAP_ROWS), len(HEATMAP_COLS)), np.nan)
            for i in range(len(HEATMAP_ROWS)):
                for j in range(len(HEATMAP_COLS)):
                    A[i, j] = M[(scope, split, i, j)]
            im = ax.imshow(A, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(HEATMAP_COLS)))
            ax.set_xticklabels([c[1] for c in HEATMAP_COLS], fontsize=8.5)
            if ci == 0:
                ax.set_yticks(range(len(HEATMAP_ROWS)))
                labels = ax.set_yticklabels([r[2] for r in HEATMAP_ROWS], fontsize=8)
                for lab, (_arch, loss, _lbl) in zip(labels, HEATMAP_ROWS, strict=True):
                    lab.set_color(C_POINT if loss in ("mse", "pointwise_norm") else C_RANK)
                    lab.set_fontweight("bold")
            else:
                ax.set_yticks([])
            if ri == 0:
                ax.set_title(scope_lbl, fontsize=10.5, fontweight="bold")
            for i in range(len(HEATMAP_ROWS)):
                for j in range(len(HEATMAP_COLS)):
                    if np.isnan(A[i, j]):
                        ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="#eeeeee", edgecolor="white"))
                        ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5,
                                color="#999999")
                        n_missing += 1
                        continue
                    frac = (A[i, j] - vmin) / (vmax - vmin + 1e-9)
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.8, fontweight="bold",
                            color="white" if frac > 0.6 else INK)
            if ci == 0:
                ax.text(-1.65, len(HEATMAP_ROWS) / 2 - 0.5, split_lbl, rotation=90,
                        va="center", ha="center", fontsize=9.5, fontweight="bold",
                        transform=ax.transData)
            for y in (1.5, 4.5, 6.5):
                ax.axhline(y, color="#bbbbbb", lw=0.6)

    fig.subplots_adjust(right=0.9)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(title_metric, fontsize=10, fontweight="bold")
    handles = [mpatches.Patch(color=C_POINT, label="Pointwise regression (MSE / z-MSE)"),
               mpatches.Patch(color=C_RANK,
                              label="Learn-to-rank (LambdaMART / RankNet / LambdaLoss)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.04),
               title="Row labels coloured by training objective")
    if n_missing:
        fig.text(0.5, 0.02, f"n/a = value not available ({n_missing} cells: neither an OOF "
                 "archive on disk nor a cached value). Regenerate the config with "
                 "run_benchmark.py, then re-run this script.",
                 ha="center", fontsize=8, style="italic", color="#666666")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (vmin={vmin:.3f} vmax={vmax:.3f}, {n_missing} n/a cells)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=100.0)
    a = ap.parse_args()
    fig_alpha_heatmap(a.alpha)


if __name__ == "__main__":
    main()
