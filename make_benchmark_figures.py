#!/usr/bin/env python
"""Figures for the public gram-negative benchmark (issue #3).

Reads the per-config result JSONs in results/benchmark/ (written by run_benchmark.py)
and renders the headline comparison. This is a SEPARATE pipeline from make_figures.py,
which draws the Phase-1 RankNet gate from results/ranknet_gate/.

  verdict     headline: best pointwise vs best ranking per (scope x split), BEDROC
  heatmap     method x representation (recovers the axis 'verdict' collapses)
  multimetric best-pointwise vs best-ranking across all metrics x conditions
  all         render every figure

Usage:
    micromamba run -n learn2rank python make_benchmark_figures.py verdict
    micromamba run -n learn2rank python make_benchmark_figures.py verdict --metric ef@0.01
    micromamba run -n learn2rank python make_benchmark_figures.py all
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from bootstrap_ci import bootstrap_aggregate_cached

FIG_DIR = Path("results/figures")
RESULTS = "results/benchmark/*.json"
BOOT_B = 1000   # bootstrap resamples for the CI whiskers
BOOT_CACHE = Path("results/benchmark/bootstrap_ci_cache.json")   # per-OOF CI cache

# palette shared with make_figures.py
C_POINT = "#F58518"   # pointwise family (warm)
C_RANK = "#7A4FA3"    # ranking family (cool)
INK = "#222222"

SCOPES = [("S1_wt_ecoli", "S1 · WT E. coli\n(single %GI scale)"),
          ("S2_ecoli_pmic", "S2 · E. coli + pMIC\n(cross-scale)"),
          ("S3_multispecies", "S3 · multi-species\n(cross-scale)")]
SPLITS = [("cv", "Chemistry holdout (5-fold CV)"),
          ("lodo", "Assay transfer (leave-one-source-out)")]

# The ablation losses belong in a separate figure, never the headline.
CORE_LOSSES = {"mse", "pointwise_norm", "ranknet", "lambdaloss", "lambdamart"}
FEAT_INITIAL = {"rdkit": "R", "morgan": "M", "chemeleon": "C", "learned": "D"}

# Each method is (arch, {loss family}); the pointwise cell takes the STRONGER of
# mse / pointwise_norm (pointwise_norm only exists on the cross-scale scopes), i.e.
# the steelman pointwise baseline. gnn_mt is the production-style multi-task D-MPNN;
# under LODO it is scored by sibling-head consensus (no head exists for a held-out
# assay), a DIFFERENT quantity from the other LODO bars -- flagged in the caption.
POINTWISE = [("rf", "RF"), ("xgb", "XGB"), ("mlp", "MLP"), ("gnn", "D-MPNN"),
             ("gnn_mt", "D-MPNN·MT")]
RANKING = [("mlp", "ranknet", "MLP·RankNet"), ("mlp", "lambdaloss", "MLP·LambdaLoss"),
           ("xgb", "lambdamart", "XGB·LambdaMART"), ("gnn", "ranknet", "D-MPNN·RankNet"),
           ("gnn", "lambdaloss", "D-MPNN·LambdaLoss")]


def load() -> dict:
    """(scope, split, arch, features, loss) -> {'bedroc':.., 'per_assay':[{metric:..}]}.

    LODO is reported on the COMPOUND-DISJOINT splits (reviewer F1: the held-out source's
    compounds are excluded from training). A config can have both an overlap and a disjoint
    LODO result on disk; when it does, the disjoint one wins. CV is per-compound holdout
    already, so `disjoint_lodo` never applies there.
    """
    out, is_dj = {}, {}
    for p in sorted(glob.glob(RESULTS)):
        r = json.load(open(p))
        if r.get("status") != "ok" or not r.get("aggregate"):
            continue
        c = r["config"]
        key = (c["scope"], c["split"], c["arch"], c["features"], c["loss"])
        dj = bool(c.get("disjoint_lodo"))
        # keep the first seen, unless a disjoint LODO result supersedes an overlap one
        if key in out and not (dj and not is_dj[key]):
            continue
        out[key] = {"agg": r["aggregate"], "per_assay": r["per_assay"],
                    "run_id": r.get("run_id")}
        is_dj[key] = dj
    return out


def _best(data: dict, scope: str, split: str, arch: str, losses: set,
          metric: str) -> tuple[float, str, str] | None:
    """Best (over representation & given losses) mean metric for one arch/scope/split.

    Returns (mean, run_id, feature_initial). run_id locates the winning config's OOF
    vector for the bootstrap CI.
    """
    best = None
    for (sc, sp, ar, feat, loss), v in data.items():
        if (sc, sp, ar) != (scope, split, arch) or loss not in losses:
            continue
        m = v["agg"].get(metric)
        if m is None:
            continue
        if best is None or m > best[0]:
            best = (m, v.get("run_id"), FEAT_INITIAL.get(feat, "?"))
    return best


def fig_verdict(metric: str = "bedroc", out: Path | None = None) -> None:
    data = load()
    out = out or FIG_DIR / f"benchmark_verdict_{metric.replace('@', '')}.png"
    title_metric = {"bedroc": "BEDROC (α=20)", "ef@0.01": "EF@1%",
                    "ef@0.05": "EF@5%", "ndcg": "NDCG",
                    "spearman": "Spearman ρ"}.get(metric, metric)

    # column order: each pointwise arch, then each ranking method
    bars = ([("point", ar, {"mse", "pointwise_norm"}, lbl) for ar, lbl in POINTWISE] +
            [("rank", ar, {ls}, lbl) for ar, ls, lbl in RANKING])

    fig, axes = plt.subplots(len(SPLITS), len(SCOPES), figsize=(15.5, 8.2),
                             sharey=True)
    boot_cache = json.load(open(BOOT_CACHE)) if BOOT_CACHE.exists() else {}
    no_ci_bars = False   # becomes True if any bar is drawn without a bootstrap CI
    for ri, (split, split_lbl) in enumerate(SPLITS):
        for ci, (scope, scope_lbl) in enumerate(SCOPES):
            ax = axes[ri, ci]
            x = np.arange(len(bars))
            xlabels = [lbl for _, _, _, lbl in bars]
            heights, initials = [], []
            for xi, (kind, arch, losses, _lbl) in enumerate(bars):
                b = _best(data, scope, split, arch, losses, metric)
                h = b[0] if b else 0.0
                heights.append(h)
                initials.append(b[2] if b else "")
                if h <= 0:
                    continue
                color = C_POINT if kind == "point" else C_RANK
                # bootstrap CI whisker: recompute from the winning config's OOF when it is
                # on disk, else fall back to the shipped cache (the OOF archives are large
                # and not distributed, so the cache is what makes the whiskers regenerable).
                # A bar with neither shows the estimate without a whisker (translucent hatch),
                # never a fabricated CI.
                ci_ = None
                if metric == "bedroc" and b and b[1]:
                    oof = Path("results/benchmark") / f"{b[1]}_oof.npz"
                    key = f"{b[1]}_oof.npz"
                    if oof.exists():
                        ci_ = bootstrap_aggregate_cached(str(oof), boot_cache, B=BOOT_B)
                    elif key in boot_cache:
                        ci_ = boot_cache[key]
                if ci_ is not None:
                    yerr = [[max(0, h - ci_["lo"])], [max(0, ci_["hi"] - h)]]
                    ax.bar(xi, h, color=color, edgecolor="white", linewidth=0.6, zorder=2)
                    ax.errorbar(xi, h, yerr=yerr, ecolor="#444", elinewidth=1.1,
                                capsize=3, fmt="none", zorder=3)
                else:
                    ax.bar(xi, h, color=color, alpha=0.45, edgecolor=color, linewidth=0.9,
                           hatch="///", zorder=2)
                    no_ci_bars = True
            # "line to beat": best pointwise in this panel
            best_pt = max(heights[:len(POINTWISE)]) if any(heights[:len(POINTWISE)]) else 0
            if best_pt > 0:
                ax.axhline(best_pt, ls="--", lw=1.1, color=C_POINT, alpha=0.7, zorder=1)
            for xi, (h, ini) in enumerate(zip(heights, initials, strict=True)):
                if h > 0:
                    ax.text(xi, h + 0.012, f"{h:.2f}", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold", color=INK)
                    ax.text(xi, 0.01, ini, ha="center", va="bottom", fontsize=7,
                            color="white", fontweight="bold", zorder=3)
            # divider between pointwise and ranking blocks
            ax.axvline(len(POINTWISE) - 0.5, color="#cccccc", lw=1.0, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=7.2, rotation=32, ha="right")
            ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(scope_lbl, fontsize=10.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split_lbl}\n\n{title_metric}", fontsize=9.5,
                              fontweight="bold")
    ax.set_ylim(0, max(0.85, 1.0))

    # a translucent-hatch legend entry is added only if some bar actually lacks a CI
    handles = [
        mpatches.Patch(color=C_POINT, label="Pointwise regression (MSE / per-assay z-MSE)"),
        mpatches.Patch(color=C_RANK, label="Learn-to-rank (RankNet / LambdaLoss / LambdaMART)"),
        plt.Line2D([], [], color="#444", lw=1.1, label="95% bootstrap CI"),
        plt.Line2D([], [], ls="--", color=C_POINT, label="best pointwise in panel")]
    if no_ci_bars:
        handles.insert(3, mpatches.Patch(facecolor="#999999", alpha=0.45, hatch="///",
                                         label="estimate — CI pending"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, 0.012, "Whiskers = 95% bootstrap CI (within-assay compound resampling, "
             f"{BOOT_B} draws, fixed endpoint set). LODO uses COMPOUND-DISJOINT splits (the "
             "held-out source's compounds are excluded from training). D-MPNN·MT LODO is "
             "scored by sibling-head consensus — a different quantity from the other LODO bars.",
             ha="center", fontsize=7.6, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    BOOT_CACHE.write_text(json.dumps(boot_cache, indent=1))
    print(f"wrote {out}")


# rows of the representation heatmap: only the CPU arms carry the three representations
# (gnn/gnn_mt are D-MPNN-learned, single-representation, so they live in `verdict`).
HEATMAP_ROWS = [("rf", "mse", "RF · MSE"), ("rf", "pointwise_norm", "RF · z-MSE"),
                ("xgb", "mse", "XGB · MSE"), ("xgb", "pointwise_norm", "XGB · z-MSE"),
                ("xgb", "lambdamart", "XGB · LambdaMART"),
                ("mlp", "mse", "MLP · MSE"), ("mlp", "pointwise_norm", "MLP · z-MSE"),
                ("mlp", "ranknet", "MLP · RankNet"),
                ("mlp", "lambdaloss", "MLP · LambdaLoss")]
HEATMAP_COLS = [("rdkit", "RDKit"), ("morgan", "Morgan"), ("chemeleon", "CheMeleon")]


def fig_heatmap(metric: str = "bedroc", out: Path | None = None) -> None:
    """method x representation, one heatmap per (scope x split). Shows whether a better
    representation rescues ranking (it does not) and how much it lifts pointwise."""
    data = load()
    out = out or FIG_DIR / f"benchmark_heatmap_{metric.replace('@', '')}.png"
    title_metric = {"bedroc": "BEDROC (α=20)", "ef@0.01": "EF@1%", "ef@0.05": "EF@5%",
                    "ndcg": "NDCG", "spearman": "Spearman ρ"}.get(metric, metric)

    # a shared colour scale across panels so cells are comparable; anchored to the data
    vals = [v["agg"][metric] for k, v in data.items()
            if k[2] in ("rf", "xgb", "mlp") and metric in v["agg"]]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)

    fig, axes = plt.subplots(len(SPLITS), len(SCOPES), figsize=(16, 9.5))
    im = None
    for ri, (split, split_lbl) in enumerate(SPLITS):
        for ci, (scope, scope_lbl) in enumerate(SCOPES):
            ax = axes[ri, ci]
            M = np.full((len(HEATMAP_ROWS), len(HEATMAP_COLS)), np.nan)
            for i, (arch, loss, _) in enumerate(HEATMAP_ROWS):
                for j, (feat, _) in enumerate(HEATMAP_COLS):
                    v = data.get((scope, split, arch, feat, loss))
                    if v and metric in v["agg"]:
                        M[i, j] = v["agg"][metric]
            im = ax.imshow(M, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(HEATMAP_COLS)))
            ax.set_xticklabels([c[1] for c in HEATMAP_COLS], fontsize=8.5)
            if ci == 0:
                ax.set_yticks(range(len(HEATMAP_ROWS)))
                labels = ax.set_yticklabels([r[2] for r in HEATMAP_ROWS], fontsize=8)
                # colour each row label by objective family (pointwise vs learn-to-rank)
                # so the two groups are distinguishable at a glance
                for lab, (_arch, loss, _lbl) in zip(labels, HEATMAP_ROWS, strict=True):
                    lab.set_color(C_POINT if loss in ("mse", "pointwise_norm") else C_RANK)
                    lab.set_fontweight("bold")
            else:
                ax.set_yticks([])
            if ri == 0:
                ax.set_title(scope_lbl, fontsize=10.5, fontweight="bold")
            # annotate every present cell; grey out the missing ones (z-MSE on S1)
            for i in range(len(HEATMAP_ROWS)):
                for j in range(len(HEATMAP_COLS)):
                    if np.isnan(M[i, j]):
                        ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="#eeeeee", edgecolor="white"))
                        continue
                    frac = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.8, fontweight="bold",
                            color="white" if frac > 0.6 else INK)
            if ci == 0:
                ax.text(-1.65, len(HEATMAP_ROWS) / 2 - 0.5, split_lbl, rotation=90,
                        va="center", ha="center", fontsize=9.5, fontweight="bold",
                        transform=ax.transData)
            # separate pointwise rows (first 2 of each arch block) visually from ranking
            for y in (1.5, 4.5, 6.5):  # rf|.., xgb pointwise|lambdamart, mlp pt|rank
                ax.axhline(y, color="#bbbbbb", lw=0.6)

    fig.subplots_adjust(right=0.9)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(title_metric, fontsize=10, fontweight="bold")
    # row labels are coloured by training objective; spell it out
    handles = [mpatches.Patch(color=C_POINT, label="Pointwise regression (MSE / z-MSE)"),
               mpatches.Patch(color=C_RANK,
                              label="Learn-to-rank (LambdaMART / RankNet / LambdaLoss)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.04),
               title="Row labels coloured by training objective")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _best_in_family(data: dict, scope: str, split: str, family: str,
                    metric: str) -> float | None:
    """Best mean `metric` over every config in a family (pointwise|ranking), any arch,
    any representation, both D-MPNN arms included."""
    point = {"mse", "pointwise_norm"}
    rank = {"ranknet", "lambdaloss", "lambdamart"}
    want = point if family == "point" else rank
    best = None
    for (sc, sp, _ar, _, loss), v in data.items():
        if (sc, sp) != (scope, split) or loss not in want:
            continue
        m = v["agg"].get(metric)
        if m is not None and (best is None or m > best):
            best = m
    return best


def fig_multimetric(out: Path | None = None) -> None:
    """One panel per metric; in each, best-pointwise vs best-ranking across all six
    (scope x split) conditions. Makes the point that the verdict is metric-invariant."""
    data = load()
    out = out or FIG_DIR / "benchmark_multimetric.png"
    metrics = [("bedroc", "BEDROC (α=20)"), ("ef@0.01", "EF@1%"),
               ("ndcg", "NDCG"), ("spearman", "Spearman ρ")]
    conds = [(sc, sp, f"{sc.split('_')[0]}·{sp}") for sp, _ in SPLITS for sc, _ in SCOPES]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.6))
    x = np.arange(len(conds))
    w = 0.38
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        pt = [_best_in_family(data, sc, sp, "point", metric) or 0 for sc, sp, _ in conds]
        rk = [_best_in_family(data, sc, sp, "rank", metric) or 0 for sc, sp, _ in conds]
        ax.bar(x - w / 2, pt, w, color=C_POINT, label="best pointwise",
               edgecolor="white", linewidth=0.5, zorder=2)
        ax.bar(x + w / 2, rk, w, color=C_RANK, label="best ranking",
               edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in conds], fontsize=7.2, rotation=40, ha="right")
        ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=C_POINT, label="best pointwise regression"),
               mpatches.Patch(color=C_RANK, label="best learn-to-rank")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.02, 1, 1.0])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["verdict", "heatmap", "multimetric", "all"])
    p.add_argument("--metric", default="bedroc",
                   choices=["bedroc", "ef@0.01", "ef@0.05", "ndcg", "spearman"])
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.which in ("verdict", "all"):
        fig_verdict(a.metric)
    if a.which in ("heatmap", "all"):
        fig_heatmap(a.metric)
    if a.which in ("multimetric", "all"):
        fig_multimetric()
