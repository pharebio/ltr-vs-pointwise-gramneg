"""Public gram-negative benchmark: task scopes, thresholds, and eligibility.

Defines the three evaluation scopes that span the homogeneous -> heterogeneous axis the
paper tests, plus the per-endpoint hit thresholds and the min-actives eligibility rule.
Operates on the shipped public dataframe (`data/public_gram_negative_wide_df.csv.gz`), whose
columns are the publicly available assay endpoints:

  PubChem AID 710 / AID 1053175 · CO-ADD (open-source subset) · Gadiya pMIC + validation ·
  Zahed 2021 tolC.

Usage:
    python curate_public_data.py            # profile the three scopes over the public CSV
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

OUT_CSV = "data/public_gram_negative_wide_df.csv.gz"
SMILES_COL = "canonicalized_smiles"

PMIC_TASK = "gadiya_pmic_gram_negative__pMIC"
GI_THRESHOLD = 80.0   # %GI hit
PMIC_THRESHOLD = 7.0  # pMIC hit (log scale -- never on the same axis as %GI)

# Evaluation inclusion criterion. Early-recognition metrics degenerate when an assay has
# almost no actives (BEDROC/EF are undefined at zero actives, and jump in huge quantized
# steps with a handful). Requiring >=25 actives keeps every reported number meaningful and
# is a single stateable criterion for the paper. Affects S3 only (S1/S2 tasks all have
# >=58 actives).
MIN_ACTIVES = 25

# --- task scopes: homogeneous -> heterogeneous ----------------------------------- #
# S1 homogeneous: WT E. coli %GI only.
S1_WT_ECOLI = [
    "aid_0000710_e_coli_BW25113_100_uM",
    "aid_1053175_e_coli__",
    "coadd_os_Escherichia coli_ATCC 25922_32_ug/mL",
    "gadiya_validation_Escherichia_coli_50_uM",
]
# S2 mixed E. coli: WT + efflux-KO (tolC) + OM-defect (lpxC) %GI, plus the pMIC log-scale head.
S2_ECOLI_PLUS_PMIC = S1_WT_ECOLI + [
    "coadd_os_Escherichia coli_tolC; MB5747_32_ug/mL",
    "zahed_2021_e_coli_BW25113_deltaTolC_10_uM",
    "coadd_os_Escherichia coli_lpxC; MB4902_32_ug/mL",
    PMIC_TASK,
]
# S3 multi-species: every public head passing MIN_ACTIVES (computed as "all public tasks").
# NB: name it honestly -- the >=25-actives criterion removes the ONLY K. pneumoniae head
# (15 actives) and BOTH wild-type P. aeruginosa heads (16 and 0 actives), so S3 is an
# E. coli-dominated multi-species subset (7 E. coli + 1 A. baumannii + 1 efflux-KO P.
# aeruginosa PAO397 + 1 pMIC), NOT "all gram-negative". State this in the paper.
S3_LABEL = "S3 multi-species (public, >=25 actives)"


def public_task_columns(df: pd.DataFrame) -> list[str]:
    """Every assay-endpoint column of the public dataframe (all columns but SMILES)."""
    return [c for c in df.columns if c != SMILES_COL]


def hit_threshold(task: str) -> float:
    """Per-endpoint hit threshold: 80% GI everywhere, pMIC 7 for the log-scale head."""
    return PMIC_THRESHOLD if task == PMIC_TASK else GI_THRESHOLD


def n_actives(df: pd.DataFrame, task: str) -> int:
    """Actives in a task under its own endpoint threshold."""
    v = df[task].dropna()
    return int((v >= hit_threshold(task)).sum())


def eligible_tasks(df: pd.DataFrame, tasks: list[str],
                   min_actives: int = MIN_ACTIVES) -> tuple[list[str], list[tuple[str, int]]]:
    """Split ``tasks`` into (eligible, [(excluded_task, n_actives), ...]) on MIN_ACTIVES."""
    keep, drop = [], []
    for t in tasks:
        na = n_actives(df, t)
        (keep if na >= min_actives else drop).append(t if na >= min_actives else (t, na))
    return keep, drop


def profile(df: pd.DataFrame, tasks: list[str], label: str) -> pd.DataFrame:
    """Per-task n / actives / hit-rate for one scope."""
    rows = []
    for t in tasks:
        v = df[t].dropna()
        thr = hit_threshold(t)
        n_act = int((v >= thr).sum())
        rows.append({"scope": label, "task": t, "n": len(v), "n_actives": n_act,
                     "hit_rate_%": round(100 * n_act / len(v), 3) if len(v) else np.nan,
                     "threshold": thr})
    out = pd.DataFrame(rows)
    cover = df[tasks].notna().any(axis=1).sum()
    print(f"\n=== {label} === {len(tasks)} tasks | {cover:,} compounds with >=1 measurement")
    print(out.to_string(index=False))
    print(f"  total measurements: {int(out['n'].sum()):,} | "
          f"total actives: {int(out['n_actives'].sum()):,}")
    return out


def main():
    argparse.ArgumentParser(description="Profile the public benchmark scopes.").parse_args()
    df = pd.read_csv(OUT_CSV)
    print(f"public df: {len(df):,} compounds x {len(public_task_columns(df))} tasks")
    s1, _ = eligible_tasks(df, S1_WT_ECOLI)
    s2, _ = eligible_tasks(df, S2_ECOLI_PLUS_PMIC)
    s3, _ = eligible_tasks(df, public_task_columns(df))
    profile(df, s1, "S1 homogeneous (WT E. coli %GI)")
    profile(df, s2, "S2 mixed (WT+KO E. coli %GI + pMIC)")
    profile(df, s3, S3_LABEL)


if __name__ == "__main__":
    main()
