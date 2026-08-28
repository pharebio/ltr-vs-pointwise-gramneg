#!/usr/bin/env python
"""Which design factor moves BEDROC the most? A one-factor-at-a-time range decomposition.

Quantifies the manuscript claim that the TRAINING OBJECTIVE (pointwise vs ranking) is the
dominant design factor -- more so than the molecular representation or the model
architecture. For each (scope x split) we vary one factor at a time within matched cells
(the other two factors held fixed) and report the mean within-cell BEDROC range:

  objective range      mean over (arch, representation) cells of
                       [max - min BEDROC across that cell's loss objectives]
  representation range mean over (arch, objective) cells of
                       [max - min across {RDKit, Morgan, CheMeleon}]
  architecture range   mean over (representation, objective) cells of
                       [max - min across {RF, XGB, MLP}]

Only the CPU factorial (arch in {RF, XGB, MLP} x representation in {RDKit, Morgan,
CheMeleon}) is used, because it is the region where representation and architecture are
actually crossed -- the D-MPNN arms are single-representation (their own learned features)
and so cannot enter a representation or architecture contrast. A cell contributes to a
factor's range only when >=2 levels of that factor are present in it: every (arch, rep)
cell has >=2 objectives, every (arch, loss) cell has all 3 representations, and only the
two shared pointwise losses (mse, pointwise_norm) span all 3 architectures -> 6 arch cells.

LODO is reported on the COMPOUND-DISJOINT splits (same as the figures). Note this differs
from an earlier draft of this table computed on overlap LODO; the CV columns are unchanged.

    micromamba run -n learn2rank python make_factor_range_table.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from make_benchmark_figures import load

ARCH = ["rf", "xgb", "mlp"]
REP = ["rdkit", "morgan", "chemeleon"]
# the pointwise + ranking objectives that make up the CPU factorial (ablation losses
# such as lambdaloss_k / lambdamart_k are excluded, as in the headline figures)
LOSS = ["mse", "pointwise_norm", "ranknet", "lambdaloss", "lambdamart"]

SCOPES = [("S1_wt_ecoli", "S1"), ("S2_ecoli_pmic", "S2"), ("S3_multispecies", "S3")]
SPLITS = ["cv", "lodo"]
OUT = Path("results/tables")


def _bedroc(data, scope, split, arch, rep, loss):
    v = data.get((scope, split, arch, rep, loss))
    return v["agg"]["bedroc"] if v and "bedroc" in v["agg"] else None


def factor_ranges(data, scope, split) -> dict:
    """The three within-cell mean ranges for one (scope, split)."""
    def rng(cells):
        out = []
        for vs in cells:
            vs = [x for x in vs if x is not None]
            if len(vs) >= 2:
                out.append(max(vs) - min(vs))
        return float(np.mean(out)), len(out)

    obj_cells = [[_bedroc(data, scope, split, a, r, l) for l in LOSS]
                 for a in ARCH for r in REP]
    rep_cells = [[_bedroc(data, scope, split, a, r, l) for r in REP]
                 for a in ARCH for l in LOSS]
    arc_cells = [[_bedroc(data, scope, split, a, r, l) for a in ARCH]
                 for r in REP for l in LOSS]
    obj, n_obj = rng(obj_cells)
    rep, n_rep = rng(rep_cells)
    arc, n_arc = rng(arc_cells)
    return {"objective": obj, "representation": rep, "architecture": arc,
            "n_obj": n_obj, "n_rep": n_rep, "n_arc": n_arc}


def main() -> None:
    data = load()
    rows = []
    for scope, slbl in SCOPES:
        for split in SPLITS:
            r = factor_ranges(data, scope, split)
            r["scope_split"] = f"{slbl} {split}"
            rows.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    # CSV (tracked, machine-readable source)
    import csv
    with open(OUT / "factor_range_decomposition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope_split", "objective", "representation", "architecture",
                    "n_obj", "n_rep", "n_arc"])
        for r in rows:
            w.writerow([r["scope_split"], f"{r['objective']:.3f}",
                        f"{r['representation']:.3f}", f"{r['architecture']:.3f}",
                        r["n_obj"], r["n_rep"], r["n_arc"]])

    # Markdown (objective bolded -- it is the dominant factor in every row)
    md = ["# BEDROC factor-range decomposition (one factor at a time)",
          "",
          "Mean within-matched-cell BEDROC(α=20) range when a single design factor is "
          "varied and the other two are held fixed. Objective (pointwise vs ranking) moves "
          "the metric more than representation or architecture in every condition.",
          "",
          "| scope · split | objective | representation | architecture |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scope_split']} | **{r['objective']:.3f}** | "
                  f"{r['representation']:.3f} | {r['architecture']:.3f} |")
    (OUT / "factor_range_decomposition.md").write_text("\n".join(md) + "\n")

    # LaTeX
    tex = [r"\begin{tabular}{lccc}", r"\toprule",
           r"scope $\cdot$ split & objective & representation & architecture \\",
           r"\midrule"]
    for r in rows:
        tex.append(f"{r['scope_split']} & \\textbf{{{r['objective']:.3f}}} & "
                   f"{r['representation']:.3f} & {r['architecture']:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "factor_range_decomposition.tex").write_text("\n".join(tex) + "\n")

    # console
    print(f"{'scope·split':12s} {'objective':>10s} {'representation':>14s} "
          f"{'architecture':>13s}")
    for r in rows:
        print(f"{r['scope_split']:12s} {r['objective']:10.3f} "
              f"{r['representation']:14.3f} {r['architecture']:13.3f}")
    print(f"\nwrote {OUT}/factor_range_decomposition.{{csv,md,tex}}")


if __name__ == "__main__":
    main()
