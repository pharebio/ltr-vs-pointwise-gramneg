#!/usr/bin/env python
"""
Figures for the Phase-1 RankNet go/no-go gate.

  fig1  architecture of the initial MLP scorer (shared by RankNet + pointwise MLP)
  fig2  RankNet vs pointwise-MLP vs Random Forest, mean+/-std across LODO folds

Usage:
    micromamba run -n learn2rank python make_figures.py fig1
    micromamba run -n learn2rank python make_figures.py fig2 --group e_coli_wt
    micromamba run -n learn2rank python make_figures.py all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path("results/figures")

# palette (color-blind friendly, works on light backgrounds)
C_INPUT = "#4C78A8"
C_HIDDEN = "#54A24B"
C_HEAD = "#E45756"
C_RANK = "#B279A2"
C_LAMBDA = "#5254A3"
C_POINT = "#F58518"
C_RF = "#72B7B2"
INK = "#222222"


# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, fc, tc="white", fs=10, ec="none", lw=0, alpha=1.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, zorder=3, linespacing=1.25)


def _arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, zorder=1, shrinkA=1, shrinkB=1))


def fig_architecture(out=FIG_DIR / "mlp_architecture.png"):
    """Schematic of MLPScorer + the three methods: RankNet & pointwise MLP are the
    SAME scorer f (shared weights, different losses); Random Forest branches from the
    input RDKit features (no MLP)."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # ---- central scorer pipeline f(x) (vertical, left-of-center) ----
    cx, w = 1.6, 3.1
    ys = [5.3, 4.35, 3.4, 2.45, 1.5, 0.55]
    h = 0.66
    _box(ax, (cx, ys[0]), w, h, "Molecule (SMILES)", "#6B7280", fs=10)
    _box(ax, (cx, ys[1]), w, h,
         "RDKit 2D normalized descriptors\n200-dim, CDF-normalized ∈ [0,1]", C_INPUT, fs=9.5)
    _box(ax, (cx, ys[2]), w, h, "Linear 200→256 · ReLU · Dropout 0.1", C_HIDDEN, fs=9.5)
    _box(ax, (cx, ys[3]), w, h, "Linear 256→256 · ReLU · Dropout 0.1", C_HIDDEN, fs=9.5)
    _box(ax, (cx, ys[4]), w, h, "embedding  z ∈ ℝ²⁵⁶", "#3B7A57", fs=10)
    _box(ax, (cx, ys[5]), w, h, "Linear 256→1   →  score  f(x) ∈ ℝ", C_HEAD, fs=10)
    for a, b in zip(ys[:-1], ys[1:], strict=True):
        _arrow(ax, (cx + w / 2, a), (cx + w / 2, b + h))

    ax.text(cx + w / 2, 6.02, "Shared scorer  f : molecule → ℝ",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.annotate("backbone\n(2 hidden layers)", xy=(cx, 3.73), xytext=(0.15, 3.73),
                ha="left", va="center", fontsize=8.5, color="#555555",
                rotation=90)

    # ---- right: the objectives. RF branches from the INPUT FEATURES (no MLP);
    #      RankNet + pointwise are the SAME scorer f trained with different losses ----
    rx, rw = 6.4, 4.2

    # Random Forest (top) -- consumes the RDKit features directly, NOT the scorer
    _box(ax, (rx, 4.32), rw, 0.95, "", "#E8F4F3", tc=INK, ec=C_RF, lw=1.4)
    ax.text(rx + rw / 2, 5.00, "Random Forest  —  classical reference",
            ha="center", fontsize=10, fontweight="bold", color="#2C7A74")
    ax.text(rx + rw / 2, 4.62,
            "same 200-dim RDKit features, separate (non-MLP) model",
            ha="center", fontsize=8.2, color="#444444")

    # RankNet (mid) -- the scorer f trained pairwise
    _box(ax, (rx, 2.05), rw, 2.0, "", "#F4ECF7", tc=INK, ec=C_RANK, lw=1.6)
    ax.text(rx + rw / 2, 3.83, "RankNet  —  pairwise (primary)",
            ha="center", fontsize=11, fontweight="bold", color=C_RANK)
    ax.text(rx + 0.25, 3.30,
            "two molecules from the SAME assay, i and j:", fontsize=8.8, color=INK, ha="left")
    _box(ax, (rx + 0.3, 2.55), 1.5, 0.5, "f(xᵢ)", C_HEAD, fs=10)
    _box(ax, (rx + 2.4, 2.55), 1.5, 0.5, "f(xⱼ)", C_HEAD, fs=10)
    ax.text(rx + rw / 2, 2.32,
            "loss = BCE( σ(f(xᵢ) − f(xⱼ)),  1[ yᵢ > yⱼ ] )",
            ha="center", fontsize=9.2, color=INK)
    ax.text(rx + rw / 2, 2.14,
            "active-anchored pairs · within-assay only · no cross-assay scale",
            ha="center", fontsize=7.8, color="#666666", style="italic")

    # Pointwise MLP (bottom) -- the SAME scorer f trained with MSE
    _box(ax, (rx, 0.45), rw, 1.35, "", "#FEF3E7", tc=INK, ec=C_POINT, lw=1.6)
    ax.text(rx + rw / 2, 1.55, "Pointwise MLP  —  regression baseline",
            ha="center", fontsize=11, fontweight="bold", color=C_POINT)
    ax.text(rx + rw / 2, 1.12, "SAME backbone + head, MSE loss:",
            ha="center", fontsize=8.8, color=INK)
    ax.text(rx + rw / 2, 0.78, "loss = ( f(x) − z-score(y) )²",
            ha="center", fontsize=9.2, color=INK)

    # RF arrow: branches from the INPUT FEATURES box (never touches the MLP)
    x_in = cx + w
    y_in = ys[1] + h / 2                       # RDKit-descriptors box, right edge
    _arrow(ax, (x_in, y_in), (rx, 4.80), color=C_RF, lw=1.3)
    ax.text((x_in + rx) / 2, 4.88, "from features", ha="center", va="bottom",
            fontsize=7.6, color="#2C7A74", style="italic")

    # RankNet + pointwise arrows: both leave the shared score head f(x)
    x_sc, y_sc = cx + w, ys[5] + h / 2         # score head, right edge
    _arrow(ax, (x_sc, y_sc + 0.12), (rx, 3.0), color=C_RANK, lw=1.3)
    _arrow(ax, (x_sc, y_sc - 0.12), (rx, 1.1), color=C_POINT, lw=1.3)
    ax.text(5.42, 1.66, "one scorer f,\nshared weights,\ntwo losses",
            ha="center", va="center", fontsize=7.4, color="#888888")

    fig.suptitle("Initial LTR scorer: MLP over RDKit descriptors",
                 fontsize=13.5, fontweight="bold", y=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
def fig_results(group="e_coli_wt", out=None, mode="lodo", features="rdkit"):
    """Grouped bar chart of mean+/-std across folds for each method.

    mode='lodo' reads {group}_per_fold.csv (one cell per held-out source);
    mode='cv'   reads {group}_cv_per_fold.csv (one cell per fold×assay).
    features='chemeleon' reads from the results/ranknet_gate/chemeleon/ subdir.
    """
    tag = "" if mode == "lodo" else "cv_"
    suffix = ("" if features == "rdkit" else f"_{features}") + ("" if mode == "lodo" else "_cv")
    results_dir = Path("results/ranknet_gate") / ("" if features == "rdkit" else features)
    out = out or FIG_DIR / f"ranknet_vs_baselines_{group}{suffix}.png"
    df = pd.read_csv(results_dir / f"{group}_{tag}per_fold.csv")

    all_methods = ["ranknet", "lambdaloss", "pointwise_mlp", "random_forest"]
    methods = [m for m in all_methods if (df["method"] == m).any()]
    short = {"ranknet": "RankNet", "lambdaloss": "LambdaLoss",
             "pointwise_mlp": "Pointwise MLP", "random_forest": "Random Forest"}
    legend_desc = {"ranknet": "RankNet — pairwise (ranking objective)",
                   "lambdaloss": "LambdaLoss-NDCG — listwise (ranking objective)",
                   "pointwise_mlp": "Pointwise MLP — MSE, same backbone",
                   "random_forest": "Random Forest — classical QSAR"}
    colors = {"ranknet": C_RANK, "lambdaloss": C_LAMBDA,
              "pointwise_mlp": C_POINT, "random_forest": C_RF}
    metrics = [("bedroc", "BEDROC (α=20)"), ("auroc", "AUROC"),
               ("ef@0.01", "EF@1%"), ("ndcg", "NDCG"), ("spearman", "Spearman ρ")]

    fold_col = "held_out_source" if mode == "lodo" else "fold"
    n_folds = df[fold_col].nunique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.9))
    x = np.arange(len(methods))
    for ax, (col, title) in zip(axes, metrics, strict=True):
        means = [df[df.method == m][col].mean() for m in methods]
        stds = [df[df.method == m][col].std() for m in methods]
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[colors[m] for m in methods],
                      edgecolor="white", linewidth=0.6, zorder=2,
                      error_kw=dict(ecolor="#555555", lw=1.2))
        # overlay per-fold points for honesty about spread
        for xi, m in enumerate(x):
            pts = df[df.method == methods[xi]][col].to_numpy()
            jitter = np.random.default_rng(xi).uniform(-0.1, 0.1, len(pts))
            ax.scatter(np.full_like(pts, m) + jitter, pts,
                       s=13, color="#333333", alpha=0.5, zorder=3)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([short[m] for m in methods], fontsize=8, rotation=18, ha="right")
        ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for b, mean in zip(bars, means, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{mean:.2f}" if col not in ("ef@0.01",) else f"{mean:.0f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        top = max(means) + max(stds) if any(stds) else max(means)
        ax.set_ylim(0, top * 1.22)

    handles = [matplotlib.patches.Patch(facecolor=colors[m], label=legend_desc[m])
               for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    design = (f"leave-one-source-out ({n_folds} folds)" if mode == "lodo"
              else f"chem-balanced {n_folds}-fold CV (production BalanceSplit), "
                   "per-assay then averaged")
    pts_label = "individual folds" if mode == "lodo" else "fold×assay cells"
    feat_label = "RDKit-descriptor" if features == "rdkit" else f"{features} (2048-dim) embedding"
    fig.suptitle(
        f"RankNet vs pointwise regression vs Random Forest  —  {group}, {design}\n"
        f"{feat_label} features; higher is better. Points = {pts_label}.",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_torchview(out=FIG_DIR / "mlp_torchview.png"):
    """Auto-traced computation graph of MLPScorer via torchview (needs graphviz)."""
    from torchview import draw_graph

    from model import MLPScorer

    g = draw_graph(
        MLPScorer(in_dim=200), input_size=(1, 200),
        graph_name="MLPScorer", graph_dir="TB", expand_nested=True,
        hide_inner_tensors=False, roll=False, save_graph=False,
    )
    g.visual_graph.format = "png"
    g.visual_graph.render(filename=out.stem, directory=str(out.parent), cleanup=True)
    print(f"wrote {out}")


def fig_architecture_combined(out=FIG_DIR / "mlp_architecture_combined.png"):
    """Composite: (a) annotated design + objectives, (b) torchview trace, side by side."""
    import matplotlib.image as mpimg

    a_path, b_path = FIG_DIR / "mlp_architecture.png", FIG_DIR / "mlp_torchview.png"
    fig_architecture()   # always regenerate so the composite can't embed a stale panel
    fig_torchview()
    imgA, imgB = mpimg.imread(a_path), mpimg.imread(b_path)
    (hA, wA), (hB, wB) = imgA.shape[:2], imgB.shape[:2]
    rA, rB = wA / hA, wB / hB  # aspect ratios at equal height

    fig = plt.figure(figsize=(0.9 * (rA + rB) * 6.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[rA, rB], wspace=0.03)
    panels = [
        (fig.add_subplot(gs[0]), imgA, "(a)  annotated design + training objectives"),
        (fig.add_subplot(gs[1]), imgB, "(b)  torchview computation-graph trace"),
    ]
    for ax, img, tag in panels:
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(tag, fontsize=11, fontweight="bold", loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_loss_curves(group="e_coli_wt", out=None, mode="lodo", features="rdkit"):
    """Per-epoch training loss for the MLP variants, one line per fold."""
    import json

    tag = "" if mode == "lodo" else "cv_"
    suffix = ("" if features == "rdkit" else f"_{features}") + ("" if mode == "lodo" else "_cv")
    results_dir = Path("results/ranknet_gate") / ("" if features == "rdkit" else features)
    out = out or FIG_DIR / f"loss_curves_{group}{suffix}.png"
    with open(results_dir / f"{group}_{tag}loss_curves.json") as fh:
        blob = json.load(fh)
    curves = blob["curves"]

    panel_defs = [("ranknet", "RankNet — pairwise BCE loss", C_RANK),
                  ("lambdaloss", "LambdaLoss-NDCG — listwise loss", C_LAMBDA),
                  ("pointwise_mlp", "Pointwise MLP — MSE loss", C_POINT)]
    panels = [p for p in panel_defs if curves.get(p[0])]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.8))
    if len(panels) == 1:
        axes = [axes]
    for ax, (method, title, base_c) in zip(axes, panels, strict=True):
        folds = curves.get(method, {})
        cmap = plt.get_cmap("viridis", max(len(folds), 1))
        for i, (held, hist) in enumerate(sorted(folds.items())):
            if not hist:
                continue
            ax.plot(np.arange(1, len(hist) + 1), hist, color=cmap(i),
                    lw=1.8, marker="o", ms=3, label=f"–{held}")
        ax.set_title(title, fontsize=12, fontweight="bold", color=base_c)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean training loss")
        ax.grid(ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        ax.legend(title="held-out source", fontsize=7.5, title_fontsize=8,
                  loc="upper right", ncol=1, framealpha=0.9)
    fig.suptitle(
        f"Training-loss curves for the MLP variants — {group}, "
        f"{'leave-one-source-out' if mode == 'lodo' else 'chem-balanced CV'}\n"
        "one curve per held-out fold; note the loss scales differ (BCE vs MSE).",
        fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["fig1", "fig1b", "combined", "fig2", "fig3", "all"])
    p.add_argument("--group", default="e_coli_wt")
    p.add_argument("--mode", choices=["lodo", "cv"], default="lodo",
                   help="which eval run to plot for fig2/fig3")
    p.add_argument("--features", choices=["rdkit", "chemeleon"], default="rdkit",
                   help="which feature-source results to plot for fig2/fig3")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.which in ("fig1", "all"):
        fig_architecture()
    if a.which in ("fig1b", "all"):
        fig_torchview()
    if a.which in ("fig1b", "combined", "all"):
        fig_architecture_combined()
    if a.which in ("fig2", "all"):
        fig_results(a.group, mode=a.mode, features=a.features)
    if a.which in ("fig3", "all"):
        fig_loss_curves(a.group, mode=a.mode, features=a.features)
