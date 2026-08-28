#!/usr/bin/env python
"""Figure + table for the R1 pMIC base-rate ablation (results/ablation/pmic_baserate.json).

Both are computed from the persisted ablation JSON -- no hand-typed numbers -- so the
caption's quantities and the drawn bars come from the same source.

The question: on the pMIC head RankNet is the best objective, but pMIC differs from the
%GI heads on scale AND base rate at once. Holding scale/granularity fixed and thinning
pMIC's actives to a sparse-head base rate (2,596 -> 108) removes the base-rate difference.
If RankNet's lead survives it is a scale effect; if it vanishes it is base rate.

    micromamba run -n learn2rank python make_ablation_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from make_benchmark_figures import C_POINT, C_RANK, INK

SRC = Path("results/ablation/pmic_baserate.json")
FIG = Path("results/figures/ablation_pmic_baserate.png")
TBL = Path("results/tables/ablation_pmic_baserate")

# global-z MSE is omitted from the figure: it is the straw-man pointwise baseline on a
# cross-scale scope (it drowns pMIC under %GI variance). z-MSE (per-assay) is the real
# pointwise comparator. global-z is kept in the TABLE for reference.
ORDER = ["pointwise_norm", "ranknet", "lambdaloss"]
LABEL = {"pointwise_norm": "z-MSE\n(per-assay)", "ranknet": "RankNet",
         "lambdaloss": "LambdaLoss"}
FAMILY = {"pointwise_norm": C_POINT, "ranknet": C_RANK, "lambdaloss": C_RANK}
PMIC = "gadiya_pmic_gram_negative__pMIC"
ABL = Path("results/ablation")
BOOT_B = 1000
# small shipped cache of the bootstrap CIs, so this figure regenerates without the large
# per-seed ablation OOF archives (keyed "full|<loss>" / "matched|<loss>").
CI_CACHE = ABL / "ablation_ci_cache.json"
_CACHE: dict = {}


def _margins(s: dict) -> tuple[float, float]:
    """RankNet minus the (steelman) per-assay z-MSE pointwise baseline, full & matched."""
    full = s["ranknet"]["full"] - s["pointwise_norm"]["full"]
    matched = s["ranknet"]["matched_mean"] - s["pointwise_norm"]["matched_mean"]
    return full, matched


def _ci_full(loss: str, point: float) -> tuple[float, float]:
    """95% bootstrap CI on the full-regime pMIC BEDROC (within-pMIC resampling).

    Computed from OOF when present (updating the cache); else served from the shipped
    cache; else (nan, nan) so the caller can skip the whisker."""
    key = f"full|{loss}"
    p = ABL / f"ablation-pmicFULL-{loss}_oof.npz"
    if not p.exists():
        c = _CACHE.get(key)
        return (c[0], c[1]) if c else (float("nan"), float("nan"))
    from bootstrap_ci import bootstrap_assay_draws
    dr = bootstrap_assay_draws(str(p), PMIC, BOOT_B)
    lo, hi = (float(v) for v in np.percentile(dr, [2.5, 97.5]))
    _CACHE[key] = [lo, hi]
    return lo, hi


def _ci_matched(loss: str, n_seeds: int) -> tuple[float, float]:
    """95% CI on the matched pMIC BEDROC, pooling bootstrap draws across the seeds --
    captures both the which-108-actives draw and within-pMIC compound resampling.
    Same OOF-then-cache fallback as _ci_full."""
    key = f"matched|{loss}"
    paths = [ABL / f"ablation-pmic108-{loss}-s{k}_oof.npz" for k in range(n_seeds)]
    if not all(p.exists() for p in paths):
        c = _CACHE.get(key)
        return (c[0], c[1]) if c else (float("nan"), float("nan"))
    from bootstrap_ci import bootstrap_assay_draws
    draws = np.concatenate([
        bootstrap_assay_draws(str(p), PMIC, BOOT_B, seed=k) for k, p in enumerate(paths)])
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    _CACHE[key] = [lo, hi]
    return lo, hi


def figure(d: dict) -> None:
    global _CACHE
    _CACHE = json.load(open(CI_CACHE)) if CI_CACHE.exists() else {}
    s = d["summary"]
    n_seeds = d["seeds"]
    x = np.arange(len(ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for i, loss in enumerate(ORDER):
        c = FAMILY[loss]
        pf = s[loss]["full"]
        lo_f, hi_f = _ci_full(loss, pf)
        ax.bar(i - w / 2, pf, w, color=c, edgecolor="white",
               label="_full" if i else "full base rate (2,596 actives)")
        if not np.isnan(lo_f):
            ax.errorbar(i - w / 2, pf, yerr=[[max(0, pf - lo_f)], [max(0, hi_f - pf)]],
                        ecolor="#444", elinewidth=1.1, capsize=3, fmt="none")
        pm = s[loss]["matched_mean"]
        lo_m, hi_m = _ci_matched(loss, n_seeds)
        ax.bar(i + w / 2, pm, w, color=c, alpha=0.45, hatch="///", edgecolor="white",
               label="_m" if i else f"base-rate-matched (108 actives, {n_seeds} seeds)")
        if not np.isnan(lo_m):
            ax.errorbar(i + w / 2, pm, yerr=[[max(0, pm - lo_m)], [max(0, hi_m - pm)]],
                        ecolor="#444", elinewidth=1.1, capsize=3, fmt="none")
        top_f = pf if np.isnan(hi_f) else hi_f
        top_m = pm if np.isnan(hi_m) else hi_m
        ax.text(i - w / 2, top_f + 0.008, f"{pf:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(i + w / 2, top_m + 0.008, f"{pm:.2f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[o] for o in ORDER], fontsize=9.5)
    ax.set_ylabel("pMIC per-head BEDROC (α=20)", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(s[o]["full"] for o in ORDER) * 1.25)

    mf, mm = _margins(s)
    ax.annotate(
        f"RankNet − z-MSE:\n  full  {mf:+.3f}\n  matched  {mm:+.3f}",
        xy=(0.985, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#cccccc"))
    handles = [mpatches.Patch(fc="#888", label="full base rate (2,596 actives)"),
               mpatches.Patch(fc="#888", alpha=0.45, hatch="///",
                              label=f"base-rate-matched (108 actives, {n_seeds} seeds)"),
               mpatches.Patch(fc=C_POINT, label="pointwise (z-MSE)"),
               mpatches.Patch(fc=C_RANK, label="learn-to-rank"),
               plt.Line2D([], [], color="#444", lw=1.1, label="95% bootstrap CI")]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    CI_CACHE.write_text(json.dumps(_CACHE, indent=1))
    print(f"wrote {FIG}")


def table(d: dict) -> None:
    s = d["summary"]
    nt, nf = d["n_target"], d["n_full_actives"]
    mf, mm = _margins(s)
    head = (f"**Table. pMIC base-rate ablation (S2 · CV · mlp/rdkit).** pMIC actives "
            f"subsampled {nf:,} → {nt} (matching coadd E. coli ATCC 25922), inactive pool "
            f"fixed, {d['seeds']} seeds. BEDROC α=20 on the pMIC head.")
    md = [head, "", "| Objective | Full base rate | Base-rate-matched (mean ± SD) | Δ |",
          "|---|--:|--:|--:|"]
    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Objective & Full base rate & Base-rate-matched & $\Delta$ \\", r"\midrule"]
    csv = ["objective,full,matched_mean,matched_sd,delta"]
    pretty = {"mse": "MSE (global-z)", "pointwise_norm": "z-MSE (per-assay)",
              "ranknet": "RankNet", "lambdaloss": "LambdaLoss"}
    # the table keeps global-z MSE for reference (it is dropped only from the figure)
    table_order = ["mse", "pointwise_norm", "ranknet", "lambdaloss"]
    for loss in table_order:
        v = s[loss]
        delta = v["matched_mean"] - v["full"]
        md.append(f"| {pretty[loss]} | {v['full']:.3f} | {v['matched_mean']:.3f} ± "
                  f"{v['matched_sd']:.3f} | {delta:+.3f} |")
        tex.append(f"{pretty[loss]} & {v['full']:.3f} & {v['matched_mean']:.3f} $\\pm$ "
                   f"{v['matched_sd']:.3f} & {delta:+.3f} \\\\")
        csv.append(f"{loss},{v['full']:.4f},{v['matched_mean']:.4f},{v['matched_sd']:.4f},"
                   f"{delta:.4f}")
    tex += [r"\midrule",
            f"RankNet $-$ z-MSE & {mf:+.3f} & {mm:+.3f} & \\\\",
            r"\bottomrule", r"\end{tabular}"]
    md += ["", f"RankNet − z-MSE (steelman pointwise): **{mf:+.3f}** (full) → "
           f"**{mm:+.3f}** (matched)."]
    TBL.parent.mkdir(parents=True, exist_ok=True)
    TBL.with_suffix(".md").write_text("\n".join(md) + "\n")
    TBL.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    TBL.with_suffix(".csv").write_text("\n".join(csv) + "\n")
    print("\n".join(md))
    print(f"\nwrote {TBL}.{{md,tex,csv}}")


def main() -> None:
    d = json.load(open(SRC))
    figure(d)
    table(d)


if __name__ == "__main__":
    main()
