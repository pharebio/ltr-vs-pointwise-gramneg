#!/usr/bin/env python
"""Manuscript tables for the public gram-negative benchmark (issue #3).

Both tables are computed from the live data -- the experiment registry and the curated
public CSV -- so they cannot drift from what was actually run/curated. Emits GitHub
markdown (for review / the repo) and LaTeX booktabs (for the manuscript) side by side.

  design    Table 1: the config matrix, marginal counts per dimension
  tasks     Table 2: per-task n / actives / hit-rate / scope membership
  all       both

Usage:
    micromamba run -n learn2rank python make_benchmark_tables.py all
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import curate_public_data as C
import run_benchmark as RB

OUT_DIR = Path("results/tables")

# The five core objectives; everything else in the registry is a loss ablation reported
# separately, not part of the headline design.
CORE_LOSSES = {"mse", "pointwise_norm", "ranknet", "lambdaloss", "lambdamart"}

# Human-readable labels for the 14 curated public task columns (stable set).
TASK_LABEL = {
    "aid_0000710_e_coli_BW25113_100_uM": "E. coli BW25113 (100 µM)",
    "aid_1053175_e_coli__": "E. coli (AID 1053175)",
    "coadd_os_Acinetobacter baumannii_ATCC 19606_32_ug/mL": "A. baumannii ATCC 19606",
    "coadd_os_Escherichia coli_ATCC 25922_32_ug/mL": "E. coli ATCC 25922",
    "coadd_os_Escherichia coli_lpxC; MB4902_32_ug/mL": "E. coli lpxC (OM-defect)",
    "coadd_os_Escherichia coli_tolC; MB5747_32_ug/mL": "E. coli tolC (efflux-KO)",
    "coadd_os_Klebsiella pneumoniae_ATCC 700603; MDR_32_ug/mL": "K. pneumoniae MDR",
    "coadd_os_Pseudomonas aeruginosa_ATCC 27853_32_ug/mL": "P. aeruginosa ATCC 27853",
    "coadd_os_Pseudomonas aeruginosa_PAO1_32_ug/mL": "P. aeruginosa PAO1",
    "coadd_os_Pseudomonas aeruginosa_PAO397; PAO1 d(mexAB-oprM) d(mexCD-oprJ) "
    "d(mexEF-oprN) d(mexJKL) d(mexXY) d(opmH) d(pscC)_32_ug/mL":
        "P. aeruginosa PAO397 (efflux-KO)",
    "gadiya_pmic_gram_negative__pMIC": "gram-negative pMIC",
    "gadiya_validation_Escherichia_coli_50_uM": "E. coli (50 µM)",
    "gadiya_validation_Pseudomonas_aeruginosa_50_uM": "P. aeruginosa (50 µM)",
    "zahed_2021_e_coli_BW25113_deltaTolC_10_uM": "E. coli ΔtolC (10 µM)",
}
DIM_ORDER = {
    "scope": ["S1_wt_ecoli", "S2_ecoli_pmic", "S3_multispecies"],
    "split": ["cv", "lodo"],
    "arch": ["rf", "xgb", "mlp", "gnn", "gnn_mt"],
    "loss": ["mse", "pointwise_norm", "ranknet", "lambdaloss", "lambdamart"],
    "features": ["rdkit", "morgan", "chemeleon", "learned"],
}
DIM_PRETTY = {"scope": "Scope", "split": "Split", "arch": "Architecture",
              "loss": "Objective", "features": "Representation"}
VAL_PRETTY = {
    "S1_wt_ecoli": "S1 · WT E. coli", "S2_ecoli_pmic": "S2 · E. coli+pMIC",
    "S3_multispecies": "S3 · multi-species", "cv": "5-fold CV (chemistry holdout)",
    "lodo": "leave-one-source-out (assay transfer)", "rf": "Random Forest", "xgb": "XGBoost",
    "mlp": "MLP", "gnn": "D-MPNN (single-head)", "gnn_mt": "D-MPNN (multi-task)",
    "mse": "MSE", "pointwise_norm": "per-assay z-MSE", "ranknet": "RankNet",
    "lambdaloss": "LambdaLoss-NDCG", "lambdamart": "LambdaMART",
    "rdkit": "RDKit descriptors (200-d)", "morgan": "Morgan FP (2048-d)",
    "chemeleon": "CheMeleon embed (2048-d)", "learned": "learned D-MPNN",
}


def _write(stem: str, md: str, tex: str, csv: pd.DataFrame | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stem}.md").write_text(md)
    (OUT_DIR / f"{stem}.tex").write_text(tex)
    exts = "md, .tex"
    if csv is not None:
        # the machine-readable source of the table: raw column names, unrounded numbers,
        # boolean membership -- the .md/.tex are formatted views of exactly this
        csv.to_csv(OUT_DIR / f"{stem}.csv", index=False)
        exts += ", .csv"
    print(md)
    print(f"\nwrote {OUT_DIR / stem}.{{{exts}}}\n")


def table_design() -> None:
    exps = RB.enumerate_experiments()
    core = [e for e in exps if e["loss"] in CORE_LOSSES]
    abl = [e for e in exps if e["loss"] not in CORE_LOSSES]

    md = [f"**Table 1. The benchmark matrix — {len(exps)} configurations "
          f"({len(core)} core + {len(abl)} loss ablations).**", "",
          "| Dimension | Breakdown |", "|---|---|"]
    tex = [r"\begin{tabular}{ll}", r"\toprule",
           r"Dimension & Breakdown \\", r"\midrule"]
    csv_rows = []
    for dim, order in DIM_ORDER.items():
        c = Counter(e[dim] for e in exps)
        parts = [(v, c[v]) for v in order if v in c] + \
                [(v, n) for v, n in c.items() if v not in order]
        md_cell = " · ".join(f"{VAL_PRETTY.get(v, v)} {n}" for v, n in parts)
        md.append(f"| {DIM_PRETTY[dim]} | {md_cell} |")
        tex.append(f"{DIM_PRETTY[dim]} & {md_cell} \\\\")
        for v, n in parts:
            csv_rows.append({"dimension": dim, "value": v,
                             "label": VAL_PRETTY.get(v, v), "n_configs": n,
                             "is_core_loss": (dim != "loss") or (v in CORE_LOSSES)})
    tex += [r"\bottomrule", r"\end{tabular}"]
    footnote = (f"\nAblations ({len(abl)}): " +
                ", ".join(f"{k} {v}" for k, v in Counter(e["loss"] for e in abl).items()) +
                ". Asymmetries: per-assay z-MSE exists only on the cross-scale scopes "
                "(S2/S3); gnn/gnn\\_mt are D-MPNN, single-representation; gnn\\_mt runs MSE "
                "only. Counts are therefore marginal, not a full product.")
    _write("design_matrix", "\n".join(md) + "\n" + footnote,
           "\n".join(tex) + "\n% " + footnote.replace("\n", " "),
           csv=pd.DataFrame(csv_rows))


def table_tasks() -> None:
    from make_public_splits import source_of

    df = pd.read_csv(C.OUT_CSV, low_memory=False)
    pub = C.public_task_columns(df)
    s1, s2 = set(C.S1_WT_ECOLI), set(C.S2_ECOLI_PLUS_PMIC)
    s3 = set(C.eligible_tasks(df, pub)[0])

    missing = [t for t in pub if t not in TASK_LABEL]
    if missing:  # fail loud rather than silently print a 130-char raw column name
        raise KeyError(f"no display label for {len(missing)} task(s): {missing}")

    rows = []
    for t in pub:
        v = df[t].dropna()
        na = C.n_actives(df, t)
        rows.append({
            "task_column": t, "source": source_of(t), "task": TASK_LABEL[t],
            "n": len(v), "actives": na,
            "hit_%": 100 * na / len(v) if len(v) else 0.0,
            "S1": t in s1, "S2": t in s2, "S3": t in s3,
            "in_any_scope": t in (s1 | s2 | s3),
            "threshold": C.hit_threshold(t)})
    rows.sort(key=lambda r: (r["source"], r["task"]))

    md = [f"**Table 2. Curated public tasks ({len(pub)} total; "
          f"{sum(r['in_any_scope'] for r in rows)} pass the ≥{C.MIN_ACTIVES}-actives "
          "criterion).** "
          "Hit = %GI ≥ 80 (pMIC ≥ 7 for the log-scale head).", "",
          "| Source | Task | n | Actives | Hit % | S1 | S2 | S3 |",
          "|---|---|--:|--:|--:|:-:|:-:|:-:|"]
    tex = [r"\begin{tabular}{llrrrccc}", r"\toprule",
           r"Source & Task & $n$ & Actives & Hit \% & S1 & S2 & S3 \\", r"\midrule"]
    def tick(b: bool) -> str:
        return "✓" if b else ""
    for r in rows:
        note = "" if r["in_any_scope"] else " *(excl.)*"
        md.append(f"| {r['source']} | {r['task']}{note} | {r['n']:,} | {r['actives']:,} | "
                  f"{r['hit_%']:.3f} | {tick(r['S1'])} | {tick(r['S2'])} | {tick(r['S3'])} |")
        tex_task = r["task"].replace("µ", r"\textmu ") + (r" \textit{(excl.)}"
                                                          if not r["in_any_scope"] else "")
        tex.append(f"{r['source']} & {tex_task} & {r['n']:,} & {r['actives']:,} & "
                   f"{r['hit_%']:.3f} & {tick(r['S1'])} & {tick(r['S2'])} & "
                   f"{tick(r['S3'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    _write("task_overview", "\n".join(md), "\n".join(tex).replace("✓", r"\checkmark"),
           csv=pd.DataFrame(rows))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["design", "tasks", "all"])
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.which in ("design", "all"):
        table_design()
    if a.which in ("tasks", "all"):
        table_tasks()
