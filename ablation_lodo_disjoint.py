#!/usr/bin/env python
"""F1 follow-on: does the LODO pointwise-over-ranking gap shrink under compound-disjoint
splits?

The runner's LODO maps each row to its data source and trains on every other source, so a
compound measured in both the held-out and a retained source is seen in training before
being scored in test -- up to 96% of the test set on the worst fold (aid_0000710). The
compound-disjoint partition the split builder actually wrote is discarded (F1).

Overlap rewards memorising a compound->activity magnitude. Pointwise regression fits
magnitude directly; ranking losses see only within-assay order. So overlap may favour the
pointwise arms, and part of the LODO gap could be contamination rather than objective.
This runs each mlp/rdkit LODO config BOTH ways in one harness (so the contrast is free of
the ~0.03 threading noise a cross-run diff would carry) and reports the delta.

Pre-registered reading:
  * gap ~unchanged  -> overlap is objective-neutral; the LODO verdict stands as measured.
  * gap shrinks     -> contamination inflated the pointwise lead; a first-order finding,
                       and compound-disjoint LODO should become the reported design.

    micromamba run -n learn2rank python ablation_lodo_disjoint.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

import run_benchmark as RB

# run_benchmark.main() caps torch to --n-jobs threads; calling run_config directly (as
# this script does) leaves torch at its default of ALL cores (32 here), which
# intermittently deadlocks the tight per-fold MLP train loop -- it hung twice, each time
# a fold finishing then no progress. 8 is the setting the full benchmark ran on without a
# single deadlock across hundreds of MLP fits.
torch.set_num_threads(8)

SCOPES = ["S1_wt_ecoli", "S2_ecoli_pmic"]
ARCH, FEATS, SPLIT = "mlp", "rdkit", "lodo"
CORE = ("mse", "pointwise_norm", "ranknet", "lambdaloss")
POINT, RANK = {"mse", "pointwise_norm"}, {"ranknet", "lambdaloss"}
OUT_DIR = Path("results/ablation")


def _cfgs(scope: str) -> list[dict]:
    return [e for e in RB.enumerate_experiments()
            if (e["arch"], e["features"], e["scope"], e["split"]) ==
            (ARCH, FEATS, scope, SPLIT) and e["loss"] in CORE]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(RB.resolve_data("wide"))
    rows = []

    for scope in SCOPES:
        tasks = RB.scope_tasks(wide, scope)
        long = RB.build_long(wide, tasks)
        X, keep = RB.build_features(long["smiles"].to_numpy(), FEATS)
        X, long = X[keep], long.loc[keep].reset_index(drop=True)
        base = {"X": X, "smiles": long["smiles"].to_numpy(),
                "y": long["y"].to_numpy(dtype=float), "task": long["task"].to_numpy(),
                "threshold": long["threshold"].to_numpy(dtype=float),
                "folds": RB.fold_assignments(long, SPLIT, scope)}
        print(f"\n=== {scope} ===", flush=True)
        for e in _cfgs(scope):
            loss = e["loss"]
            for disjoint in (False, True):
                cfg = {**e, "device_torch": "cpu", "n_jobs": 8,
                       "disjoint_lodo": disjoint,
                       "run_id": f"ablation-lodo{'DISJ' if disjoint else 'OVLP'}-"
                                 f"{scope.split('_')[0]}-{loss}"}
                res = RB.run_config(cfg, base, oof_path=OUT_DIR / f"{cfg['run_id']}_oof.npz")
                agg = res["aggregate"]["bedroc"]
                per = {a["assay"]: a["bedroc"] for a in res["per_assay"]}
                rows.append({"scope": scope, "loss": loss,
                             "mode": "disjoint" if disjoint else "overlap",
                             "bedroc": agg, "per_assay": per})
                print(f"  {loss:15s} {'disjoint' if disjoint else 'overlap ':8s} "
                      f"BEDROC {agg:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_json(OUT_DIR / "lodo_disjoint.json", orient="records", indent=1)

    # gap = best pointwise - best ranking, per scope per mode
    print("\n=== pointwise-vs-ranking gap: overlap vs compound-disjoint ===")
    print(f"{'scope':16s} {'mode':10s} {'best_pt':>8s} {'best_rank':>10s} {'gap':>7s}")
    summary = {}
    for scope in SCOPES:
        for mode in ("overlap", "disjoint"):
            sel = df[(df.scope == scope) & (df["mode"] == mode)]
            pt = sel[sel.loss.isin(POINT)]["bedroc"].max()
            rk = sel[sel.loss.isin(RANK)]["bedroc"].max()
            summary[f"{scope}/{mode}"] = {"best_pt": pt, "best_rank": rk, "gap": pt - rk}
            print(f"{scope:16s} {mode:10s} {pt:8.3f} {rk:10.3f} {pt - rk:7.3f}")
        g_o = summary[f"{scope}/overlap"]["gap"]
        g_d = summary[f"{scope}/disjoint"]["gap"]
        print(f"  -> {scope}: gap {g_o:.3f} (overlap) -> {g_d:.3f} (disjoint)  "
              f"Δgap {g_d - g_o:+.3f}\n")

    json.dump({"rows": rows, "summary": summary},
              open(OUT_DIR / "lodo_disjoint_summary.json", "w"), indent=1, default=str)
    print(f"wrote {OUT_DIR}/lodo_disjoint*.json")


if __name__ == "__main__":
    main()
