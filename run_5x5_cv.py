#!/usr/bin/env python
"""5x5 repeated CV (Ash et al.) — best MSE/pointwise vs best learn-to-rank, per scope.

Three head-to-head comparisons (one per scope): the top pointwise arm vs the top ranking
arm, each selected from the 1x5 grid at its best representation, run under all 5
BalanceSplit repeats. Reports the across-repeat mean +/- SD for each arm and the PAIRED
per-repeat gap (both arms share each repeat's folds and representation -> row-aligned, so
the gap is a proper paired contrast). Hand the per-repeat values to the statistician agent
for the RM-ANOVA / paired test.

The three winners are all on chemeleon, so pointwise and ranking are row-aligned within a
scope. Selection is on the 1x5; the 5x5 repeats are independent BalanceSplits, so the
comparison is on held-out splits (mild selection optimism only, noted for the paper).

    micromamba run -n learn2rank python run_5x5_cv.py [--repeats 5] [--n-jobs 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_benchmark as RB

torch.set_num_threads(8)   # direct run_config caller: cap threads (avoid MLP deadlock)

FEAT = "chemeleon"
# (pointwise arm, ranking arm) per scope -- the 1x5 winners of each family, best rep.
COMPARISONS = {
    "S1_wt_ecoli":     (("mlp", "pointwise_norm"), ("mlp", "lambdaloss")),
    "S2_ecoli_pmic":   (("xgb", "pointwise_norm"), ("mlp", "ranknet")),
    "S3_multispecies": (("xgb", "pointwise_norm"), ("mlp", "ranknet")),
}
OUT_DIR = Path("results/five_by_five")


def _cfg(arch: str, scope: str, loss: str) -> dict:
    return next(x for x in RB.enumerate_experiments()
               if (x["arch"], x["features"], x["scope"], x["split"], x["loss"])
               == (arch, FEAT, scope, "cv", loss))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.n_jobs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(RB.resolve_data("wide"))
    rows = []
    for scope, ((pa, pl), (ra, rl)) in COMPARISONS.items():
        tasks = RB.scope_tasks(wide, scope)
        long = RB.build_long(wide, tasks)
        X, keep = RB.build_features(long["smiles"].to_numpy(), FEAT)
        X_f = X[keep]
        long_f = long.loc[keep].reset_index(drop=True)
        base = {"X": X_f, "smiles": long_f["smiles"].to_numpy(),
                "y": long_f["y"].to_numpy(dtype=float), "task": long_f["task"].to_numpy(),
                "threshold": long_f["threshold"].to_numpy(dtype=float)}
        pt_cfg, rk_cfg = _cfg(pa, scope, pl), _cfg(ra, scope, rl)
        pt_b, rk_b = [], []
        for r in range(a.repeats):
            folds = RB.fold_assignments(long_f, "cv", scope, repeat=r)
            data = {**base, "folds": folds}
            sc = scope.split("_")[0]
            pr = RB.run_config({**pt_cfg, "seed": r, "n_jobs": a.n_jobs,
                                "device_torch": "cpu",
                                "run_id": f"5x5-{pa}-{pl}-{sc}-r{r}"},
                               data, oof_path=OUT_DIR / f"5x5-{pa}-{pl}-{sc}-r{r}_oof.npz")
            rr = RB.run_config({**rk_cfg, "seed": r, "n_jobs": a.n_jobs,
                                "device_torch": "cpu",
                                "run_id": f"5x5-{ra}-{rl}-{sc}-r{r}"},
                               data, oof_path=OUT_DIR / f"5x5-{ra}-{rl}-{sc}-r{r}_oof.npz")
            pb, rb = pr["aggregate"]["bedroc"], rr["aggregate"]["bedroc"]
            pt_b.append(pb)
            rk_b.append(rb)
            print(f"  {sc} r{r}: pointwise {pa}/{pl} {pb:.3f} | rank {ra}/{rl} {rb:.3f} "
                  f"| gap {pb - rb:+.3f}", flush=True)
        pt_b, rk_b = np.array(pt_b), np.array(rk_b)
        gap = pt_b - rk_b                       # paired: same repeat/folds/representation
        rows.append({
            "scope": scope, "features": FEAT, "n_repeats": a.repeats,
            "pointwise": f"{pa}/{pl}", "ranking": f"{ra}/{rl}",
            "pointwise_mean": float(pt_b.mean()), "pointwise_sd": float(pt_b.std(ddof=1)),
            "ranking_mean": float(rk_b.mean()), "ranking_sd": float(rk_b.std(ddof=1)),
            "gap_mean": float(gap.mean()), "gap_sd": float(gap.std(ddof=1)),
            "pointwise_per_repeat": pt_b.tolist(), "ranking_per_repeat": rk_b.tolist()})
        print(f"* {scope}: pointwise {pt_b.mean():.3f}±{pt_b.std(ddof=1):.3f}  "
              f"rank {rk_b.mean():.3f}±{rk_b.std(ddof=1):.3f}  "
              f"paired gap {gap.mean():+.3f}±{gap.std(ddof=1):.3f}", flush=True)

    json.dump(rows, open(OUT_DIR / "five_by_five.json", "w"), indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("per_repeat")}
                  for r in rows]).to_csv(OUT_DIR / "five_by_five.csv", index=False)
    print(f"\nwrote {OUT_DIR}/five_by_five.{{json,csv}}  ({len(rows)} comparisons)")


if __name__ == "__main__":
    main()
