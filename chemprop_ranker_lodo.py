#!/usr/bin/env python
"""
Phase-2 end-to-end D-MPNN learn-to-rank driver: LODO + chemical-similarity CV.

The graph analogue of ``run_ranknet_gate.py``. Instead of an MLP over precomputed
features, the scorer is a ChemProp D-MPNN (``ranking_chemprop.RankingMPNN``,
CheMeleon-initialized, **not frozen**) so the ranking loss backpropagates through
message passing -- the one regime the Phase-1 frozen-representation gate never tested.

For one organism group it trains three objectives on the SAME architecture, data, and
optimizer -- differing only in loss -- so the comparison isolates the objective on the
D-MPNN encoder:
  * ranknet    : pairwise, within-assay active-anchored pairs (RankNet BCE)
  * lambdaloss : listwise, per-assay DeltaNDCG-weighted lists (reduction="sum")
  * pointwise  : MSE on z-scored %GI (the honest capacity/representation-matched baseline)

Evaluation mirrors the gate exactly (so numbers line up with the frozen runs and the
scorecard):
  * --eval-mode lodo : leave-one-DATA-SOURCE-out (assay transfer). One ranking_report
    over the held-out source's rows (its own hit threshold), averaged across folds.
  * --eval-mode cv   : reuse the production BalanceSplit 1x5 folds (per-COMPOUND
    chemistry holdout). One scorer per group; rank metrics computed WITHIN each
    held-out assay then averaged over assays and folds (never pooling across assays).

Training-set construction (per fold, shared by all three methods): keep ALL actives +
a capped random sample of ``--neg-pool`` inactives per assay. This bounds the graph
featurization/GPU memory while retaining plenty of negatives to sample pairs from, and
guarantees the objective comparison is data-matched.

Needs chemprop + CheMeleon + torch. SINGLE GPU ONLY -- CUDA_VISIBLE_DEVICES is pinned to
--gpu before any torch import (Lightning DDP hangs the D-MPNN). ChemProp Python API only
-- never the CLI. ClearML tracking is optional (--log_to_clearml) and off by default.

    # one small fold
    CUDA_VISIBLE_DEVICES=0 python chemprop_ranker_lodo.py \
        --group e_coli_wt --eval-mode lodo --epochs 5 --neg-pool 2000
    # full LODO run
    CUDA_VISIBLE_DEVICES=0 python chemprop_ranker_lodo.py \
        --group e_coli_wt --eval-mode lodo --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

SMILES_COL = "canonicalized_smiles"
GI_HIT_THRESHOLD = 80.0
PMIC_HIT_THRESHOLD = 7.0

# Public data-source prefixes (mirror run_ranknet_gate.SOURCES); source_of = longest match.
SOURCES = [
    "aid_0000710", "aid_1053175", "coadd_os",
    "gadiya_validation", "gadiya_pmic", "zahed_2021",
]

# Known objectives. `pointwise_norm` (per-assay-normalized MSE) is the steelman
# comparator for the cross-scale %GI+pMIC test; excluded from the default set (on a
# single-endpoint group it's redundant with `pointwise`).
METHODS = ["ranknet", "lambdaloss", "pointwise", "pointwise_norm"]
DEFAULT_METHODS = ["ranknet", "lambdaloss", "pointwise"]
METRIC_COLS = ["spearman", "ndcg", "bedroc", "auroc", "rie", "ef@0.01", "ef@0.05"]


def source_of(task: str) -> str:
    matches = [s for s in SOURCES if task.startswith(s)]
    if not matches:
        raise ValueError(f"No known source prefix for task {task!r}")
    return max(matches, key=len)


def hit_threshold(task: str) -> float:
    return PMIC_HIT_THRESHOLD if task.endswith("pMIC") else GI_HIT_THRESHOLD


def build_task_set(groups: dict, group: str, extra_groups=None) -> list:
    """Union the primary organism group's tasks with any ``extra_groups`` tasks, giving
    ONE scorer's assay set. For the cross-scale test, e.g. group=e_coli_wt (%GI) +
    extra_groups=[gram_neg_pmic] (pMIC) pools incomparable endpoints under a single
    latent score; pairs/lists still form within an assay (never %GI-vs-pMIC), and each
    assay keeps its own hit threshold (80 vs 7 via hit_threshold)."""
    task_set = list(groups[group])
    for eg in (extra_groups or []):
        if eg not in groups:
            raise SystemExit(f"unknown --extra-groups {eg!r}; choices: {list(groups)}")
        task_set += [t for t in groups[eg] if t not in task_set]
    return task_set


# --------------------------------------------------------------------------- #
def build_group_long(wide_df, group_tasks):
    """Melt the wide table to long ``(smiles, y, task, source, threshold)`` for one
    organism group -- one row per (compound, task) actually tested (non-NaN), keeping
    SMILES (no feature alignment; the D-MPNN featurizes graphs itself). Unparseable
    SMILES are dropped so downstream index arrays stay aligned."""
    import pandas as pd
    from rdkit import Chem

    tasks = [t for t in group_tasks if t in wide_df.columns]
    frames = []
    for t in tasks:
        sub = wide_df[[SMILES_COL, t]].dropna(subset=[t])
        frames.append(pd.DataFrame({
            "smiles": sub[SMILES_COL].to_numpy(),
            "y": sub[t].to_numpy(dtype=float),
            "task": t,
            "source": source_of(t),
            "threshold": hit_threshold(t),
        }))
    long = pd.concat(frames, ignore_index=True)
    parseable = long["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None).to_numpy()
    n_drop = int((~parseable).sum())
    if n_drop:
        print(f"[build] dropped {n_drop} unparseable SMILES rows", flush=True)
    long = long.loc[parseable].reset_index(drop=True)
    return {
        "smiles": long["smiles"].to_numpy(),
        "y": long["y"].to_numpy(dtype=float),
        "source": long["source"].to_numpy(),
        "task": long["task"].to_numpy(),
        "threshold": long["threshold"].to_numpy(dtype=float),
        "long": long,
    }


def subsample_train(smiles, y, grp, thr, neg_pool, seed):
    """Keep ALL actives + up to ``neg_pool`` random inactives per assay (task). Bounds
    graph featurization/GPU memory; identical set is handed to all three objectives."""
    import numpy as np

    if neg_pool is None or neg_pool <= 0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(grp):
        gi = np.where(grp == g)[0]
        t = float(thr[gi][0])
        act = gi[y[gi] >= t]
        neg = gi[y[gi] < t]
        if len(neg) > neg_pool:
            neg = rng.choice(neg, size=neg_pool, replace=False)
        keep.append(np.concatenate([act, neg]))
    idx = np.unique(np.concatenate(keep))
    return idx


# --------------------------------------------------------------------------- #
def fit_all_methods(tr, thr_map, methods, cfg, device, seed):
    """Fit the requested objectives on one training split. ``tr`` = dict with
    smiles/y/grp arrays. Returns ({method: trainer}, {method: loss_history})."""
    from ranking_chemprop import RankingChemPropTrainer, RankingMPNN

    trainers, hists = {}, {}
    for m in methods:
        t0 = time.time()
        model = RankingMPNN.from_chemeleon(head_hidden=cfg["head_hidden"])
        trainer = RankingChemPropTrainer(
            model, encoder_lr=cfg["encoder_lr"], head_lr=cfg["head_lr"],
            warmup_frac=cfg["warmup_frac"], device=device)
        if m == "ranknet":
            trainer.fit_pairwise(
                tr["smiles"], tr["y"], tr["grp"], thr_map, epochs=cfg["epochs"],
                n_act_per_batch=cfg["n_act_per_batch"], neg_ratio=cfg["neg_ratio"],
                n_neg_per_active=cfg["n_neg_per_active"], n_pos_per_active=cfg["n_pos_per_active"],
                pos_weight=cfg["pos_weight"], min_gap=cfg["min_gap"],
                hard_neg_frac=cfg["hard_neg_frac"], mine_every=cfg["mine_every"],
                sali_neg_frac=cfg["sali_neg_frac"], sali_sim_lo=cfg["sali_sim_lo"],
                sali_sim_hi=cfg["sali_sim_hi"],
                seed=seed, verbose=True)
        elif m == "lambdaloss":
            trainer.fit_listwise(
                tr["smiles"], tr["y"], tr["grp"], thr_map, epochs=cfg["epochs"],
                list_act=cfg["list_act"], list_neg=cfg["list_neg"],
                lists_per_group=cfg["lists_per_group"], relevance_mode=cfg["relevance_mode"],
                rel_levels=cfg["rel_levels"], rel_max=cfg["rel_max"], gain=cfg["lambda_gain"],
                hard_neg_frac=cfg["hard_neg_frac"], mine_every=cfg["mine_every"],
                sali_neg_frac=cfg["sali_neg_frac"], sali_sim_lo=cfg["sali_sim_lo"],
                sali_sim_hi=cfg["sali_sim_hi"],
                seed=seed, verbose=True)
        elif m == "pointwise":
            trainer.fit_pointwise(
                tr["smiles"], tr["y"], epochs=cfg["epochs"],
                batch_size=cfg["pointwise_batch"], seed=seed, verbose=True)
        elif m == "pointwise_norm":
            trainer.fit_pointwise_grouped(
                tr["smiles"], tr["y"], tr["grp"], epochs=cfg["epochs"],
                batch_size=cfg["pointwise_batch"], seed=seed, verbose=True)
        else:
            raise SystemExit(f"unknown method {m!r}")
        trainers[m] = trainer
        hists[m] = list(trainer.loss_history_)
        print(f"  [{m}] trained in {time.time()-t0:.1f}s", flush=True)
    return trainers, hists


# --------------------------------------------------------------------------- #
def run(group, cfg, methods, out_dir, gpu, neg_pool, seed, device,
        extra_groups=None, log_to_clearml=False):
    """Leave-one-data-source-out (assay transfer)."""
    import numpy as np
    import pandas as pd

    import data_access
    from metrics import ranking_report
    from organism_groups import load_organism_groups

    groups = load_organism_groups()
    if group not in groups:
        raise SystemExit(f"unknown group {group!r}; choices: {list(groups)}")
    print(f"=== D-MPNN ranker (LODO): group={group} ===", flush=True)

    task = _init_task(group, "lodo", cfg, methods, neg_pool, seed, gpu) if log_to_clearml else None

    wide = pd.read_csv(data_access.wide_csv())
    task_set = build_task_set(groups, group, extra_groups)
    data = build_group_long(wide, task_set)
    smi, y, src, grp, thr = (data["smiles"], data["y"], data["source"],
                             data["task"], data["threshold"])
    xg = f" (+extra_groups {extra_groups})" if extra_groups else ""
    print(f"long table: {len(y):,} (compound,task) rows over sources {sorted(set(src))} "
          f"| {len(set(grp))} assays{xg}", flush=True)

    held_sources = sorted(set(src))
    if cfg.get("held_out_source"):
        if cfg["held_out_source"] not in held_sources:
            raise SystemExit(f"--held-out-source {cfg['held_out_source']!r} not in {held_sources}")
        held_sources = [cfg["held_out_source"]]

    rows, loss_curves = [], {m: {} for m in methods}
    for held in held_sources:
        te = src == held
        tr_mask = ~te
        held_thr = float(thr[te][0])
        n_act_te = int((y[te] >= held_thr).sum())
        print(f"\n--- held-out source: {held}  (test n={int(te.sum()):,}, "
              f"actives={n_act_te}, train n={int(tr_mask.sum()):,}) ---", flush=True)
        if n_act_te == 0:
            print("  no actives in held-out source -> skipping fold", flush=True)
            continue

        keep = subsample_train(smi[tr_mask], y[tr_mask], grp[tr_mask], thr[tr_mask],
                               neg_pool, seed)
        tr = {"smiles": smi[tr_mask][keep], "y": y[tr_mask][keep], "grp": grp[tr_mask][keep]}
        thr_map = {g: float(thr[tr_mask][keep][grp[tr_mask][keep] == g][0])
                   for g in np.unique(tr["grp"])}
        print(f"  train set: {len(tr['y']):,} rows over {len(thr_map)} assays "
              f"(actives kept in full + up to {neg_pool}/assay negatives)", flush=True)

        trainers, hists = fit_all_methods(tr, thr_map, methods, cfg, device, seed)
        for m in methods:
            loss_curves[m][held] = hists[m]
            scores = trainers[m].score(list(smi[te]))
            rep = ranking_report(y[te], scores, active_threshold=held_thr)
            rep.update({"method": m, "held_out_source": held})
            rows.append(rep)
            print(f"    {m:12s} bedroc={rep['bedroc']:.3f} auroc={rep['auroc']:.3f} "
                  f"ef@1%={rep['ef@0.01']:.1f} ndcg={rep['ndcg']:.3f} "
                  f"spearman={rep['spearman']:.3f}", flush=True)

    _write_and_summarize(group, "lodo", rows, loss_curves, cfg, methods, neg_pool,
                         seed, out_dir, sorted(set(src.tolist())), task)


def run_cv(group, cfg, methods, out_dir, gpu, neg_pool, seed, device,
           extra_groups=None, cv_split_csv=None, log_to_clearml=False):
    """Chemical-similarity CV reusing the production BalanceSplit 1x5 folds."""
    import numpy as np
    import pandas as pd

    import data_access
    from metrics import ranking_report
    from organism_groups import load_organism_groups

    groups = load_organism_groups()
    if group not in groups:
        raise SystemExit(f"unknown group {group!r}; choices: {list(groups)}")
    print(f"=== D-MPNN ranker (CV): group={group} ===", flush=True)

    task = _init_task(group, "cv", cfg, methods, neg_pool, seed, gpu) if log_to_clearml else None

    wide = pd.read_csv(data_access.wide_csv())
    task_set = build_task_set(groups, group, extra_groups)
    data = build_group_long(wide, task_set)
    smi, y, src, grp, thr = (data["smiles"], data["y"], data["source"],
                             data["task"], data["threshold"])

    if cv_split_csv is None:
        cv_split_csv = data_access.split_1x5_csv()
    sp = pd.read_csv(cv_split_csv, usecols=["canonicalized_smiles", "Folds"])
    smi2fold = dict(zip(sp["canonicalized_smiles"], sp["Folds"], strict=True))
    fold_arr = data["long"]["smiles"].map(smi2fold).to_numpy(dtype=float)
    valid = ~np.isnan(fold_arr)
    folds = sorted({int(f) for f in fold_arr[valid]})
    print(f"long table: {len(y):,} rows | CV folds mapped {int(valid.sum()):,}/{len(fold_arr):,} "
          f"-> {len(folds)} folds", flush=True)

    rows, loss_curves = [], {m: {} for m in methods}
    for k in folds:
        te = (fold_arr == k) & valid
        tr_mask = (fold_arr != k) & valid
        val_assays = [a for a in sorted(set(grp[te].tolist()))
                      if int((y[te][grp[te] == a] >= float(thr[te][grp[te] == a][0])).sum()) > 0]
        print(f"\n--- fold {k}  (test n={int(te.sum()):,} over {len(val_assays)} assays, "
              f"train n={int(tr_mask.sum()):,}) ---", flush=True)
        if not val_assays:
            print("  no assay with actives -> skipping fold", flush=True)
            continue

        keep = subsample_train(smi[tr_mask], y[tr_mask], grp[tr_mask], thr[tr_mask],
                               neg_pool, seed)
        tr = {"smiles": smi[tr_mask][keep], "y": y[tr_mask][keep], "grp": grp[tr_mask][keep]}
        thr_map = {g: float(thr[tr_mask][keep][grp[tr_mask][keep] == g][0])
                   for g in np.unique(tr["grp"])}
        print(f"  train set: {len(tr['y']):,} rows over {len(thr_map)} assays", flush=True)

        trainers, hists = fit_all_methods(tr, thr_map, methods, cfg, device, seed)
        if cfg.get("save_models"):
            import torch
            mdir = out_dir / "models"
            mdir.mkdir(parents=True, exist_ok=True)
            for m in methods:
                mp = mdir / f"{group}_cv_fold{k}_{m}_seed{seed}.pt"
                torch.save(trainers[m].model.state_dict(), mp)
                if task is not None:
                    task.upload_artifact(f"model_{m}_fold{k}_seed{seed}", artifact_object=str(mp))
                print(f"  [save] {mp}", flush=True)
        te_smi = list(smi[te])
        for m in methods:
            loss_curves[m][f"fold{k}"] = hists[m]
            scores = trainers[m].score(te_smi)
            for a in val_assays:
                ma = grp[te] == a
                thr_a = float(thr[te][ma][0])
                rep = ranking_report(y[te][ma], scores[ma], active_threshold=thr_a)
                rep.update({"method": m, "fold": f"fold{k}", "assay": a})
                rows.append(rep)
        fold_rows = [r for r in rows if r.get("fold") == f"fold{k}"]
        for m in methods:
            sub = [r for r in fold_rows if r["method"] == m]
            if sub:
                print(f"    {m:12s} bedroc={np.mean([r['bedroc'] for r in sub]):.3f} "
                      f"auroc={np.mean([r['auroc'] for r in sub]):.3f} "
                      f"ef@1%={np.mean([r['ef@0.01'] for r in sub]):.1f} "
                      f"(mean over {len(sub)} assays)", flush=True)

    _write_and_summarize(group, "cv", rows, loss_curves, cfg, methods, neg_pool,
                         seed, out_dir, sorted(set(src.tolist())), task)


# --------------------------------------------------------------------------- #
def _write_and_summarize(group, mode, rows, loss_curves, cfg, methods, neg_pool,
                         seed, out_dir, sources, task):
    import pandas as pd

    tag = "" if mode == "lodo" else "_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    per_fold = out_dir / f"{group}{tag}_per_fold.csv"
    df.to_csv(per_fold, index=False)

    summary = {}
    print(f"\n=== SUMMARY ({mode}, mean across cells) ===", flush=True)
    for m in methods:
        sub = df[df["method"] == m] if not df.empty else df
        if df.empty or sub.empty:
            continue
        summary[m] = {"n_cells": int(len(sub)),
                      **{c: {"mean": float(sub[c].mean()), "std": float(sub[c].std())}
                         for c in METRIC_COLS}}
        print(f"{m:12s} " + "  ".join(f"{c}={sub[c].mean():.3f}"
              for c in ["bedroc", "auroc", "ef@0.01", "ndcg", "spearman"]), flush=True)

    meta = {"group": group, "eval_mode": mode, "encoder": "chemeleon_dmpnn_unfrozen",
            "neg_pool": neg_pool, "seed": seed, "sources": sources,
            "metric_cols": METRIC_COLS, "methods": methods, "config": cfg,
            "summary": summary}
    summary_json = out_dir / f"{group}{tag}_summary.json"
    with open(summary_json, "w") as fh:
        json.dump(meta, fh, indent=2)
    curves_path = out_dir / f"{group}{tag}_loss_curves.json"
    with open(curves_path, "w") as fh:
        json.dump({"group": group, "eval_mode": mode, "epochs": cfg["epochs"],
                   "curves": loss_curves}, fh, indent=2)
    print(f"\nwrote {per_fold}\nwrote {summary_json}\nwrote {curves_path}", flush=True)

    if task is not None:
        _finish_task(task, summary, {"per_fold": per_fold, "loss_curves": curves_path,
                                     "summary_json": summary_json})


# --------------------------------------------------------------------------- #
def _init_task(group, eval_mode, cfg, methods, neg_pool, seed, gpu):
    from clearml import Task

    label = cfg.get("run_label") or ""
    name = f"dmpnn_ranker_{eval_mode}_{group}" + (f"_{label}" if label else "")
    task = Task.init(
        project_name=os.environ.get("LEARN2RANK_CLEARML_PROJECT", "learn2rank"),
        task_name=name,
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={"pytorch": False},  # avoid ClearML PyTorch auto-binding
        reuse_last_task_id=False,                     # never reuse/overwrite a prior task
    )
    tags = ["dmpnn_ranker", "phase2", "end_to_end", eval_mode, group]
    task.add_tags(tags + ([label] if label else []))
    # Objective summary in the ClearML **description** field (set_comment after
    # Task.init) + mirrored into the connected config so it is also searchable.
    objective = (
        f"Phase-2 end-to-end D-MPNN learn-to-rank ({eval_mode.upper()}) on organism group "
        f"'{group}'. Trains [{', '.join(methods)}] on ONE CheMeleon-initialized (unfrozen) "
        f"D-MPNN encoder + score_head — only the loss differs — to test whether a ranking "
        f"objective beats pointwise regression on early-recognition metrics (BEDROC/EF@1%) "
        f"when the ranking loss backpropagates through message passing. "
        f"encoder_lr={cfg.get('encoder_lr')}, head_lr={cfg.get('head_lr')}, "
        f"warmup_frac={cfg.get('warmup_frac')}, epochs={cfg.get('epochs')}, neg_pool={neg_pool}, "
        f"n_act_per_batch={cfg.get('n_act_per_batch')}, hard_neg_frac={cfg.get('hard_neg_frac')}, "
        f"sali_neg_frac={cfg.get('sali_neg_frac')}"
        + (f", label={label}" if label else "") + "."
    )
    task.set_comment(objective)
    task.connect({"group": group, "eval_mode": eval_mode, "methods": ",".join(methods),
                  "encoder": "chemeleon_dmpnn_unfrozen", "neg_pool": neg_pool,
                  "seed": seed, "gpu": gpu, "objective": objective, **cfg})
    print(f"[clearml] task {task.id}", flush=True)
    return task


def _finish_task(task, summary, artifacts):
    logger = task.get_logger()
    for m, s in summary.items():
        for c in ("bedroc", "auroc", "ef@0.01", "ndcg", "spearman"):
            logger.report_single_value(f"{m}/{c}", round(float(s[c]["mean"]), 4))
    for name, path in artifacts.items():
        task.upload_artifact(name, artifact_object=str(path))
    task.close()
    print("[clearml] task closed", flush=True)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="e_coli_wt")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help="comma-separated subset of: " + ",".join(METHODS) +
                        " (pointwise_norm = per-assay-normalized MSE, for the cross-scale test)")
    p.add_argument("--extra-groups", default="",
                   help="comma-separated organism groups to UNION into the scorer's assay set "
                        "(cross-scale test, e.g. gram_neg_pmic adds the pMIC endpoint alongside a "
                        "%%GI group). Pairs still form within an assay; thresholds stay per-assay.")
    p.add_argument("--eval-mode", choices=["lodo", "cv"], default="lodo")
    p.add_argument("--held-out-source", default=None,
                   help="LODO only: restrict to a single held-out data source (e.g. coadd_os) "
                        "for a quick validation fold; default runs every source")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--neg-pool", type=int, default=5000,
                   help="max inactives kept per assay for training (0 = all)")
    p.add_argument("--gpu", type=int, default=1, help="physical GPU (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--encoder-lr", type=float, default=1e-5,
                   help="LR for the pretrained CheMeleon encoder (keep low: 1e-5..5e-5)")
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="LR for the fresh score_head (10-100x encoder LR: 1e-3..3e-3)")
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="linear warmup-then-decay over this fraction of steps (0 = constant LR; "
                        "encoder-lr==head-lr + warmup-frac 0 reproduces the flat-LR baseline)")
    p.add_argument("--head-hidden", type=int, default=256)
    # pairwise (RankNet)
    p.add_argument("--n-act-per-batch", type=int, default=64)
    p.add_argument("--neg-ratio", type=int, default=3)
    p.add_argument("--n-neg-per-active", type=int, default=20)
    p.add_argument("--n-pos-per-active", type=int, default=10)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--hard-neg-frac", type=float, default=0.0,
                   help="RankNet semi-hard mining: fraction of each batch's negatives drawn from "
                        "the model's top-scoring inactives (0 = pure random). Try 0.3-0.5.")
    p.add_argument("--mine-every", type=int, default=5,
                   help="re-score inactives for hard-negative mining every N epochs")
    p.add_argument("--sali-neg-frac", type=float, default=0.0,
                   help="RankNet SALI-hard mining: fraction of each batch's negatives drawn "
                        "from structurally near-active inactives (activity cliffs, static "
                        "ECFP4-Tanimoto ranking). Combine with --hard-neg-frac for a "
                        "random:model-hard:SALI split (e.g. both 0.33).")
    p.add_argument("--sali-sim-lo", type=float, default=0.3,
                   help="SALI band floor: min ECFP4-Tanimoto to an active (below = not a cliff)")
    p.add_argument("--sali-sim-hi", type=float, default=0.9,
                   help="SALI band ceiling: drop near-duplicate cliffs (sim->1, likely noise)")
    # listwise (LambdaLoss)
    p.add_argument("--list-act", type=int, default=128)
    p.add_argument("--list-neg", type=int, default=256,
                   help="inactives per listwise list; list_act+list_neg bounds GPU memory "
                        "(512-mol lists peaked ~40GB; 384 keeps ~19GB headroom)")
    p.add_argument("--lists-per-group", type=int, default=20)
    p.add_argument("--lambda-relevance",
                   choices=["bins", "continuous", "continuous_gi", "rank_gi", "continuous_local"],
                   default="bins",
                   help="LambdaLoss relevance: bins/continuous = BINARIZED (inactives=0, "
                        "actives graded by %%GI); continuous_gi/rank_gi = NON-binarized "
                        "(every molecule graded from raw %%GI, no cliff at the hit threshold)")
    p.add_argument("--lambda-gain", choices=["exp", "linear"], default="exp",
                   help="LambdaLoss NDCG gain: exp = 2^rel-1 (top-heavy), linear = rel")
    p.add_argument("--lambda-levels", type=int, default=4,
                   help="relevance ceiling L (bounds 2^rel gains); used by all relevance modes")
    p.add_argument("--lambda-rel-max", type=float, default=100.0,
                   help="continuous_gi scale: rel = L*clip(y,0,rel_max)/rel_max (100 for %%GI)")
    # pointwise
    p.add_argument("--pointwise-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-label", default="",
                   help="suffix for the ClearML task name + a tag (e.g. 'difflr') to "
                        "distinguish configs; also disambiguates runs sharing an --out-dir")
    p.add_argument("--out-dir", default="results/dmpnn_ranker")
    p.add_argument("--cv-split-csv", default=None)
    p.add_argument("--log_to_clearml", action="store_true", default=False)
    p.add_argument("--save-models", action="store_true", default=False,
                   help="save each fold's trained model state_dict to <out-dir>/models/ (and "
                        "upload as ClearML artifacts if --log_to_clearml), for reuse/inference "
                        "-- e.g. a 1x5 CV ensemble to predict on a new screen")
    return p.parse_args()


def main():
    a = parse_args()
    # Pin the GPU BEFORE importing torch/chemprop (via ranking_chemprop).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    os.chdir(_here)
    warnings.filterwarnings("ignore", message="Attribute '.*' is an instance of `nn.Module`")

    import torch
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown methods {bad}; choices: {METHODS}")

    extra_groups = [g.strip() for g in a.extra_groups.split(",") if g.strip()]

    cfg = {"epochs": a.epochs, "encoder_lr": a.encoder_lr, "head_lr": a.head_lr,
           "warmup_frac": a.warmup_frac, "head_hidden": a.head_hidden,
           "n_act_per_batch": a.n_act_per_batch, "neg_ratio": a.neg_ratio,
           "n_neg_per_active": a.n_neg_per_active, "n_pos_per_active": a.n_pos_per_active,
           "pos_weight": a.pos_weight, "min_gap": a.min_gap,
           "hard_neg_frac": a.hard_neg_frac, "mine_every": a.mine_every,
           "sali_neg_frac": a.sali_neg_frac, "sali_sim_lo": a.sali_sim_lo,
           "sali_sim_hi": a.sali_sim_hi,
           "list_act": a.list_act, "list_neg": a.list_neg,
           "lists_per_group": a.lists_per_group, "relevance_mode": a.lambda_relevance,
           "rel_levels": a.lambda_levels, "rel_max": a.lambda_rel_max, "lambda_gain": a.lambda_gain,
           "pointwise_batch": a.pointwise_batch, "held_out_source": a.held_out_source,
           "extra_groups": ",".join(extra_groups), "run_label": a.run_label,
           "save_models": a.save_models}

    common = dict(gpu=a.gpu, neg_pool=a.neg_pool, seed=a.seed, device=device,
                  extra_groups=extra_groups, log_to_clearml=a.log_to_clearml)
    if a.eval_mode == "cv":
        run_cv(a.group, cfg, methods, Path(a.out_dir), cv_split_csv=a.cv_split_csv, **common)
    else:
        run(a.group, cfg, methods, Path(a.out_dir), **common)


if __name__ == "__main__":
    main()
