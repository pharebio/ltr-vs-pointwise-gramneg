#!/usr/bin/env python
"""R1 follow-on: is RankNet's pMIC advantage scale, or base rate?

On the pMIC head alone (S2/cv) RankNet is the best objective in every cell -- but pMIC
differs from the %GI heads on THREE axes at once: scale (log vs linear), base rate
(6.53% actives vs 0.06-1.00%), and granularity (dilution-quantised). The ranking losses
are active-anchored and neg_pool-capped, so base rate directly sets how many training
anchors RankNet gets (~1:2 balance on pMIC vs ~1:86 on the sparsest %GI head). The
observed advantage is therefore equally consistent with "RankNet wins where actives are
plentiful" -- a weaker, differently-scoped claim.

This does NOT touch the main grid. The pMIC threshold (pMIC >= 7 == MIC <= 100 nM) is a
principled potency cutoff, and pMIC's high hit rate is a real property of a curated MIC
set, not an artefact. This is an IDENTIFICATION experiment: hold scale and granularity
fixed, remove the base-rate difference, and see whether the advantage survives.

Method: on S2/cv mlp/rdkit, globally subsample the pMIC head's ACTIVES from ~2,596 down
to N_TARGET (matching coadd_os E. coli ATCC 25922, 108 actives) by dropping the excess
pMIC-active ROWS -- which touches pMIC only, since a row is one (compound, assay) pair
and the same compounds remain under their other assays. The inactive pool and every
other head are untouched. Re-run {mse, pointwise_norm, ranknet, lambdaloss} over
several subsample seeds (BEDROC on ~108 actives is itself noisy) and report the pMIC
per-head BEDROC.

Pre-registered reading (stated before running, so this is a test not a search):
  * RankNet's margin over pointwise SURVIVES  -> the effect is SCALE. Central result.
  * RankNet's margin DISAPPEARS               -> the effect is BASE RATE. The cross-scale
                                                 framing must be retired, not softened.

    micromamba run -n learn2rank python ablation_pmic_baserate.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_benchmark as RB
from curate_public_data import PMIC_TASK, PMIC_THRESHOLD

# Cap torch threads: calling run_config directly leaves torch at all cores, which can
# deadlock the per-fold MLP loop (see ablation_lodo_disjoint.py). 8 = the benchmark's
# proven-stable setting.
torch.set_num_threads(8)

SCOPE, SPLIT, ARCH, FEATS = "S2_ecoli_pmic", "cv", "mlp", "rdkit"
LOSSES = ["mse", "pointwise_norm", "ranknet", "lambdaloss"]
N_TARGET = 108                       # coadd_os E. coli ATCC 25922 active count
OUT_DIR = Path("results/ablation")


def _cfg(loss: str, seed: int) -> dict:
    """The registry config for this cell, tagged with an ablation-specific run_id."""
    e = next(x for x in RB.enumerate_experiments()
             if (x["arch"], x["features"], x["scope"], x["split"], x["loss"])
             == (ARCH, FEATS, SCOPE, SPLIT, loss))
    cfg = {**e, "device_torch": "cpu", "n_jobs": 8, "seed": seed,
           "run_id": f"ablation-pmic{N_TARGET}-{loss}-s{seed}"}
    return cfg


def _drop_excess_pmic_actives(data: dict, keep_n: int, seed: int) -> dict:
    """Return a copy of `data` with all but `keep_n` pMIC-active rows removed.

    Only pMIC-active rows are dropped; inactives and every other assay are untouched.
    Folds are preserved on the surviving rows (global subsample, not per-fold)."""
    task, y = data["task"], data["y"]
    is_pmic_active = (task == PMIC_TASK) & (y >= PMIC_THRESHOLD)
    act_idx = np.flatnonzero(is_pmic_active)
    rng = np.random.default_rng(seed)
    drop = rng.choice(act_idx, size=len(act_idx) - keep_n, replace=False)
    mask = np.ones(len(y), dtype=bool)
    mask[drop] = False
    out = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(y) else v)
           for k, v in data.items()}
    return out


def _pmic_bedroc(res: dict) -> tuple[float, int]:
    for a in res.get("per_assay", []):
        if a["assay"] == PMIC_TASK:
            return a["bedroc"], a["n_actives"]
    return float("nan"), 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of subsample seeds")
    ap.add_argument("--n-target", type=int, default=N_TARGET)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # build the S2/cv mlp/rdkit data ONCE, exactly as run_benchmark.main does
    wide = pd.read_csv(RB.resolve_data("wide"))
    tasks = RB.scope_tasks(wide, SCOPE)
    long = RB.build_long(wide, tasks)
    X, keep = RB.build_features(long["smiles"].to_numpy(), FEATS)
    X, long = X[keep], long.loc[keep].reset_index(drop=True)
    base = {"X": X, "smiles": long["smiles"].to_numpy(),
            "y": long["y"].to_numpy(dtype=float), "task": long["task"].to_numpy(),
            "threshold": long["threshold"].to_numpy(dtype=float),
            "folds": RB.fold_assignments(long, SPLIT, SCOPE)}
    n_full_act = int(((base["task"] == PMIC_TASK) & (base["y"] >= PMIC_THRESHOLD)).sum())
    print(f"pMIC actives available: {n_full_act:,}; subsampling to {a.n_target} "
          f"over {a.seeds} seed(s)\n", flush=True)

    rows = []
    # FULL reference (no subsample): reruns the committed cell in this harness so the
    # comparison is same-code-path. Seed-independent, so run once per loss.
    for loss in LOSSES:
        cfg = {**_cfg(loss, 0), "run_id": f"ablation-pmicFULL-{loss}"}
        res = RB.run_config(cfg, base, oof_path=OUT_DIR / f"{cfg['run_id']}_oof.npz")
        b, na = _pmic_bedroc(res)
        rows.append({"loss": loss, "regime": "full", "seed": None,
                     "pmic_actives": na, "pmic_bedroc": b,
                     "aggregate_bedroc": res["aggregate"]["bedroc"]})
        print(f"[full ] {loss:15s} pMIC BEDROC {b:.3f}  (n_act={na:,})", flush=True)

    # base-rate-matched
    for seed in range(a.seeds):
        data = _drop_excess_pmic_actives(base, a.n_target, seed)
        for loss in LOSSES:
            cfg = _cfg(loss, seed)
            res = RB.run_config(cfg, data, oof_path=OUT_DIR / f"{cfg['run_id']}_oof.npz")
            b, na = _pmic_bedroc(res)
            rows.append({"loss": loss, "regime": "matched", "seed": seed,
                         "pmic_actives": na, "pmic_bedroc": b,
                         "aggregate_bedroc": res["aggregate"]["bedroc"]})
            print(f"[s{seed}   ] {loss:15s} pMIC BEDROC {b:.3f}  (n_act={na:,})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "pmic_baserate.csv", index=False)

    # summary: mean +/- sd across seeds, matched vs full, per loss
    print("\n=== pMIC per-head BEDROC: full vs base-rate-matched ===")
    print(f"{'loss':16s} {'full':>7s} {'matched(mean±sd)':>20s} {'Δ vs full':>10s}")
    summary = {}
    for loss in LOSSES:
        full = df[(df.loss == loss) & (df.regime == "full")]["pmic_bedroc"].iloc[0]
        m = df[(df.loss == loss) & (df.regime == "matched")]["pmic_bedroc"]
        summary[loss] = {"full": full, "matched_mean": float(m.mean()),
                         "matched_sd": float(m.std()), "n_seeds": int(len(m))}
        print(f"{loss:16s} {full:7.3f} {m.mean():11.3f} ± {m.std():.3f}   {m.mean()-full:+10.3f}")
    # the decisive contrast: RankNet vs the stronger pointwise baseline, matched regime
    rk = summary["ranknet"]["matched_mean"]
    pt = max(summary["mse"]["matched_mean"], summary["pointwise_norm"]["matched_mean"])
    print(f"\nMATCHED: RankNet {rk:.3f} vs best pointwise {pt:.3f}  ->  "
          f"margin {rk - pt:+.3f}")
    rk_f = summary["ranknet"]["full"]
    pt_f = max(summary["mse"]["full"], summary["pointwise_norm"]["full"])
    print(f"full-regime margin was RankNet {rk_f:.3f} vs best pointwise {pt_f:.3f} "
          f"= {rk_f - pt_f:+.3f}")

    json.dump({"n_target": a.n_target, "n_full_actives": n_full_act,
               "seeds": a.seeds, "summary": summary, "rows": rows},
              open(OUT_DIR / "pmic_baserate.json", "w"), indent=1, default=str)
    print(f"\nwrote {OUT_DIR}/pmic_baserate.{{json,csv}}")


if __name__ == "__main__":
    main()
