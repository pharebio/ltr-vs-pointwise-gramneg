#!/usr/bin/env python
"""
RankNet go/no-go gate (Phase-1).

Question: does a *ranking* objective (RankNet, pairwise) beat a *pointwise*
objective on held-out gram-negative assays, holding model capacity fixed?

Design (kept honest so the comparison isolates the objective, not capacity):
  * All three methods consume the SAME 200-dim RDKit descriptors (features.py).
  * RankNet and the pointwise MLP share the SAME MLPScorer backbone
    (2x256 ReLU+dropout -> linear); only the loss differs (RankNet pairwise vs
    MSE). A Random Forest is added as the strong classical QSAR reference.
    A linear RidgeCV would confound
    objective with capacity, so it is deliberately NOT the comparator here.
  * Evaluation = leave-one-DATA-SOURCE-out within a single organism group
    (default e_coli_wt), scored with identical rank-aware metrics (metrics.py).

NB: this RDKit-feature gate is NOT directly comparable to the D-MPNN ChemProp
scorecard baseline (different encoder). It answers only "is the ranking objective
worth carrying onto the D-MPNN?" (Phase 2).

Run (CPU is fine -- the MLP is tiny; no GPU authorization needed):
    micromamba run -n learn2rank python run_ranknet_gate.py
    micromamba run -n learn2rank python run_ranknet_gate.py --group k_pneumoniae
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_access
from features import align_features, load_feature_table
from metrics import ranking_report
from organism_groups import load_organism_groups

SMILES_COL = "canonicalized_smiles"
GI_HIT_THRESHOLD = 80.0
PMIC_HIT_THRESHOLD = 7.0

# Public data-source prefixes; source_of = longest match.
SOURCES = [
    "aid_0000710", "aid_1053175", "coadd_os",
    "gadiya_validation", "gadiya_pmic", "zahed_2021",
]


def source_of(task: str) -> str:
    matches = [s for s in SOURCES if task.startswith(s)]
    if not matches:
        raise ValueError(f"No known source prefix for task {task!r}")
    return max(matches, key=len)


def hit_threshold(task: str) -> float:
    return PMIC_HIT_THRESHOLD if task.endswith("pMIC") else GI_HIT_THRESHOLD


# --------------------------------------------------------------------------- #
def build_group_table(wide_df: pd.DataFrame, group_tasks: list[str], feat_table):
    """Melt the wide table to long form for one organism group.

    One row per (compound, task) that was actually TESTED (non-NaN value), tagged
    with its data source. Features are aligned and rows without RDKit features are
    dropped. Returns a dict of row-aligned arrays.
    """
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

    X, found = align_features(long["smiles"].to_numpy(), table=feat_table)
    long = long.loc[found].reset_index(drop=True)
    X = X[found]
    return {
        "X": X,
        "y": long["y"].to_numpy(dtype=float),
        "source": long["source"].to_numpy(),
        "task": long["task"].to_numpy(),
        "threshold": long["threshold"].to_numpy(dtype=float),
        "long": long,
    }


# --------------------------------------------------------------------------- #
def fit_score_ranknet(Xtr, ytr, gtr, thr_tr, Xte, epochs, n_neg, n_pos, pos_weight,
                      min_gap, seed, device="cpu", hard_neg_rounds=0):
    """RankNet pairwise MLP: active-anchored within-source pairs -> RankNet loss.

    Pairs are active-over-inactive (n_neg per active) plus, if n_pos>0,
    active-over-active ranked by %GI (top-of-list ordering signal).

    hard_neg_rounds>0 enables hard-negative mining: after a random-negative warmup,
    re-score the training set and resample each active's negatives from the current
    top-scoring inactives (its false positives), then keep training. The epoch budget
    is split evenly across the warmup + rounds. Targets the easy-pair saturation that
    makes plain RankNet's loss collapse (Shen ACANet / Chen 2019).
    """
    import torch

    from model import MLPScorer, RankingTrainer
    from pairs import sample_pairs_imbalanced

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # per-source (=per-task here) hit threshold map for active-anchored sampling
    thr_map = {g: float(thr_tr[gtr == g][0]) for g in np.unique(gtr)}

    def _sample(hard_scores=None):
        return sample_pairs_imbalanced(
            ytr, gtr, active_threshold=thr_map, n_neg_per_active=n_neg,
            n_pos_per_active=n_pos, pos_weight=pos_weight, min_gap=min_gap,
            hard_negative_scores=hard_scores, rng=rng)

    pairs = _sample()
    if not pairs:
        return None, {"total": 0, "act_act": 0, "act_inact": 0}, []
    scorer = MLPScorer(in_dim=Xtr.shape[1])
    trainer = RankingTrainer(scorer, objective="pairwise", device=device)

    if hard_neg_rounds <= 0:
        trainer.fit_pairwise(Xtr, pairs, epochs=epochs, verbose=False)
        hist = list(trainer.loss_history_)
    else:
        per = max(1, epochs // (hard_neg_rounds + 1))    # warmup + N mining rounds
        trainer.fit_pairwise(Xtr, pairs, epochs=per, verbose=False)
        hist = list(trainer.loss_history_)
        for _ in range(hard_neg_rounds):
            pairs = _sample(hard_scores=trainer.score(Xtr))   # top-scoring inactives
            if not pairs:
                break
            trainer.fit_pairwise(Xtr, pairs, epochs=per, verbose=False)
            hist += list(trainer.loss_history_)

    n_aa = sum(1 for p in pairs if ytr[p.i] >= thr_map[p.group] and ytr[p.j] >= thr_map[p.group])
    counts = {"total": len(pairs), "act_act": n_aa, "act_inact": len(pairs) - n_aa}
    return trainer.score(Xte), counts, hist


def _graded_relevance(y_act: np.ndarray, levels: int = 4, mode: str = "bins") -> np.ndarray:
    """Map active %GI values to graded relevance in [1, levels]; inactives get 0
    (added by caller). lambdaloss_ndcg uses gains = 2**rel - 1, so relevance must be
    small bounded values (raw %GI would overflow 2**label).

    mode='bins'       : quartile rank bins (1..levels) -- coarse, discards magnitude.
    mode='continuous' : min-max scale the actives' %GI to [1, levels] -- keeps the
                        continuous potency signal without the exponential-gain blow-up
                        (rel stays <= levels), so strong hits are ordered above weak
                        ones smoothly.
    """
    n = len(y_act)
    if n == 1:
        return np.array([float(levels)])
    if mode == "continuous":
        lo, hi = float(y_act.min()), float(y_act.max())
        if hi - lo < 1e-9:
            return np.full(n, float(levels))
        return 1.0 + (levels - 1) * (y_act - lo) / (hi - lo)
    ranks = np.argsort(np.argsort(y_act))          # 0..n-1 ascending by %GI
    return 1.0 + np.minimum(ranks * levels // n, levels - 1)  # 1..levels


def fit_score_lambdaloss(Xtr, ytr, gtr, thr_tr, Xte, epochs, list_neg, seed,
                         lists_per_group=20, device="cpu", relevance_mode="bins",
                         list_act=None, levels=4, gain="exp", ndcg_k=None):
    """Listwise LambdaLoss-NDCG: per-assay lists, ΔNDCG-weighted pairs.

    Unlike RankNet BCE (every pair weighted equally -> easy pairs dominate and the
    loss collapses), LambdaLoss reweights each pair by the NDCG change a swap would
    cause, concentrating gradient on the top of the list (Wang et al. 2018). Each
    assay contributes a list = all its actives (graded 1..4 by %GI) + a fresh sample
    of ``list_neg`` inactives (relevance 0); this keeps the O(n^2) list matrix
    tractable on the huge inactive pool.

    ``lists_per_group`` resampled lists per assay per epoch keep the gradient-step
    count comparable to the pairwise/pointwise trainers (with only ~6 training assays,
    one list/assay would give ~6 steps/epoch and badly undertrain the listwise model).
    """
    import torch

    from losses import lambdaloss_ndcg
    from model import MLPScorer

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    thr_map = {g: float(thr_tr[gtr == g][0]) for g in np.unique(gtr)}
    X_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    scorer = MLPScorer(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-3, weight_decay=1e-5)

    # precompute per-group active/inactive indices + active relevance
    gdata = {}
    for g in np.unique(gtr):
        gi = np.where(gtr == g)[0]
        y = ytr[gi]
        act, neg = gi[y >= thr_map[g]], gi[y < thr_map[g]]
        if len(act) == 0 or len(neg) == 0:
            continue
        gdata[g] = (act, neg, _graded_relevance(ytr[act], levels=levels,
                                                mode=relevance_mode))
    if not gdata:
        return None, []

    loss_hist = []
    for _ in range(epochs):
        total, nlists = 0.0, 0
        for act, neg, rel_act in gdata.values():
            for _ in range(lists_per_group):
                negs = (rng.choice(neg, size=list_neg, replace=False)
                        if len(neg) > list_neg else neg)
                # Cap the actives per list too. Previously every active went into every
                # list, so an assay with 2,596 actives built a 2,596+list_neg list while a
                # sparse one built 58+list_neg -- the O(n^2) lambda matrix and the
                # actives:negatives ratio then varied ~45x across assays. list_act was
                # swept in its own right (task #23); the registry declared 128 and this
                # function silently ignored it.
                if list_act is not None and len(act) > list_act:
                    pick = rng.choice(len(act), size=list_act, replace=False)
                    acts, rels = act[pick], rel_act[pick]
                else:
                    acts, rels = act, rel_act
                idx = np.concatenate([acts, negs])
                rel = torch.tensor(np.concatenate([rels, np.zeros(len(negs))]),
                                   dtype=torch.float32, device=device)
                opt.zero_grad()
                s = scorer(X_t[torch.tensor(idx, device=device)])
                loss = lambdaloss_ndcg(s, rel, gain=gain, k=ndcg_k)
                loss.backward()
                opt.step()
                total += loss.item()
                nlists += 1
        loss_hist.append(total / max(nlists, 1))

    with torch.no_grad():
        scorer.eval()
        scores = scorer(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    return scores, loss_hist


def fit_score_pointwise(Xtr, ytr, Xte, epochs, seed, device="cpu"):
    """Capacity-matched pointwise MLP: SAME backbone, MSE loss."""
    import torch

    from model import MLPScorer, RankingTrainer

    torch.manual_seed(seed)
    scorer = MLPScorer(in_dim=Xtr.shape[1])
    trainer = RankingTrainer(scorer, objective="pairwise", device=device)
    trainer.fit_pointwise(Xtr, ytr, epochs=epochs, verbose=False)
    return trainer.score(Xte), trainer.loss_history_


def fit_score_rf(Xtr, ytr, Xte, seed, n_estimators=200, n_jobs=-1):
    """Random Forest regressor -- strong classical QSAR reference."""
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed,
        max_features="sqrt", min_samples_leaf=2,
    )
    rf.fit(Xtr, ytr)
    return rf.predict(Xte)


METHODS = ["ranknet", "lambdaloss", "pointwise_mlp", "random_forest"]
METRIC_COLS = ["spearman", "ndcg", "bedroc", "auroc", "rie", "ef@0.01", "ef@0.05"]


def _resolve_features(features: str, out_dir: Path):
    """Map a --features choice to (feature_parquet_or_None, out_dir).

    RDKit uses the default ClearML parquet (path None) and the top-level out_dir;
    other feature sets (e.g. CheMeleon embeddings) load from their own parquet and
    write under an out_dir subdir so results don't collide with the RDKit run.
    """
    if features == "rdkit":
        return None, out_dir
    if features == "chemeleon":
        return data_access.LOCAL_CHEMELEON_EMB, out_dir / "chemeleon"
    raise SystemExit(f"unknown --features {features!r} (choices: rdkit, chemeleon)")


def _init_task(group, eval_mode, params):
    """Bootstrap a ClearML task for a tracked gate run (optional; --log_to_clearml).
    Project is generic and configurable via LEARN2RANK_CLEARML_PROJECT."""
    from clearml import Task

    feat = params.get("features", "rdkit")
    task = Task.init(
        project_name=os.environ.get("LEARN2RANK_CLEARML_PROJECT", "learn2rank"),
        task_name=f"ranknet_gate_{eval_mode}_{feat}_{group}",
        auto_connect_frameworks={"pytorch": False},
        # always mint a fresh task id -- never reuse/overwrite a prior task's
        # artifacts (never reuse/overwrite a prior task's artifacts).
        reuse_last_task_id=False,
    )
    task.add_tags(["ranknet_gate", eval_mode, feat, group])
    task.connect(params)
    return task


def _finish_task(task, summary, artifacts):
    """Log per-method mean metrics as single values + upload output files, then close."""
    logger = task.get_logger()
    for m, s in summary.items():
        for c in ("bedroc", "auroc", "ef@0.01", "ndcg", "spearman"):
            logger.report_single_value(f"{m}/{c}", round(float(s[c]["mean"]), 4))
    for name, path in artifacts.items():
        task.upload_artifact(name, artifact_object=str(path))
    task.close()


def _fit_all_methods(Xtr, ytr, gtr, thr_tr, Xte, epochs, n_neg, n_pos, pos_weight,
                     min_gap, list_neg, rf_trees, seed, device="cpu", skip_rf=False,
                     hard_neg_rounds=0, relevance_mode="bins"):
    """Fit the methods on one train split; return {method: test_scores} and
    {method: loss_history} (RankNet/LambdaLoss/pointwise only). Shared by run (LODO)
    and run_cv so the two eval modes fit identically. MLPs use `device`; RF is CPU/
    sklearn and is skipped when `skip_rf` (for fast GPU tuning sweeps)."""
    scored, hists = {}, {}
    t0 = time.time()
    s_rank, pc, hists["ranknet"] = fit_score_ranknet(
        Xtr, ytr, gtr, thr_tr, Xte, epochs, n_neg, n_pos, pos_weight, min_gap, seed,
        device=device, hard_neg_rounds=hard_neg_rounds)
    scored["ranknet"] = s_rank
    hn = f", hard-neg×{hard_neg_rounds}" if hard_neg_rounds else ""
    print(f"  ranknet        ({pc['total']:,} pairs = {pc['act_inact']:,} act-inact "
          f"+ {pc['act_act']:,} act-act{hn})   {time.time()-t0:5.1f}s", flush=True)
    t0 = time.time()
    scored["lambdaloss"], hists["lambdaloss"] = fit_score_lambdaloss(
        Xtr, ytr, gtr, thr_tr, Xte, epochs, list_neg, seed, device=device,
        relevance_mode=relevance_mode)
    print(f"  lambdaloss     (list_neg={list_neg}, rel={relevance_mode})   "
          f"{time.time()-t0:5.1f}s", flush=True)
    t0 = time.time()
    scored["pointwise_mlp"], hists["pointwise_mlp"] = fit_score_pointwise(
        Xtr, ytr, Xte, epochs, seed, device=device)
    print(f"  pointwise_mlp                    {time.time()-t0:5.1f}s", flush=True)
    if not skip_rf:
        t0 = time.time()
        scored["random_forest"] = fit_score_rf(Xtr, ytr, Xte, seed, n_estimators=rf_trees)
        print(f"  random_forest  ({rf_trees} trees)    {time.time()-t0:5.1f}s", flush=True)
    return scored, hists


# --------------------------------------------------------------------------- #
def run(group: str, epochs: int, n_neg: int, n_pos: int, pos_weight: float,
        min_gap: float, list_neg: int, seed: int, rf_trees: int, out_dir: Path,
        log_to_clearml: bool = False, features: str = "rdkit", device: str = "cpu",
        skip_rf: bool = False, hard_neg_rounds: int = 0, relevance_mode: str = "bins"):
    groups = load_organism_groups()
    if group not in groups:
        raise SystemExit(f"unknown group {group!r}; choices: {list(groups)}")
    print(f"=== RankNet gate: group={group} | features={features} ===", flush=True)

    feat_path, out_dir = _resolve_features(features, out_dir)
    task = None
    if log_to_clearml:
        task = _init_task(group, "lodo", {
            "group": group, "eval_mode": "lodo", "features": features, "epochs": epochs,
            "n_neg_per_active": n_neg, "n_pos_per_active": n_pos,
            "pos_weight": pos_weight, "min_gap": min_gap, "list_neg": list_neg,
            "rf_trees": rf_trees, "seed": seed, "skip_rf": skip_rf,
            "hard_neg_rounds": hard_neg_rounds, "relevance_mode": relevance_mode})

    feat_table = load_feature_table(feat_path)
    wide = pd.read_csv(data_access.wide_csv())
    data = build_group_table(wide, groups[group], feat_table)
    X, y, src, grp, thr = (data["X"], data["y"], data["source"],
                           data["task"], data["threshold"])
    print(f"long table: {len(y):,} (compound,task) rows over sources "
          f"{sorted(set(src))}", flush=True)

    methods = [m for m in METHODS if not (skip_rf and m == "random_forest")]
    rows = []
    loss_curves = {"ranknet": {}, "lambdaloss": {}, "pointwise_mlp": {}}
    for held in sorted(set(src)):
        te = src == held
        tr = ~te
        Xtr, ytr, gtr, thr_tr = X[tr], y[tr], grp[tr], thr[tr]
        Xte, yte = X[te], y[te]
        held_thr = float(thr[te][0])
        n_act_te = int((yte >= held_thr).sum())
        print(f"\n--- held-out source: {held}  (test n={te.sum():,}, "
              f"actives={n_act_te}, train n={tr.sum():,}) ---", flush=True)
        if n_act_te == 0:
            print("  no actives in held-out source -> skipping fold", flush=True)
            continue

        scored, hists = _fit_all_methods(Xtr, ytr, gtr, thr_tr, Xte, epochs, n_neg,
                                         n_pos, pos_weight, min_gap, list_neg, rf_trees, seed,
                                         device=device, skip_rf=skip_rf,
                                         hard_neg_rounds=hard_neg_rounds,
                                         relevance_mode=relevance_mode)
        for name in loss_curves:
            loss_curves[name][held] = hists[name]

        for m in methods:
            if scored.get(m) is None:
                continue
            rep = ranking_report(yte, scored[m], active_threshold=held_thr)
            rep.update({"method": m, "held_out_source": held})
            rows.append(rep)
            print(f"    {m:15s} bedroc={rep['bedroc']:.3f} auroc={rep['auroc']:.3f} "
                  f"ef@1%={rep['ef@0.01']:.1f} ndcg={rep['ndcg']:.3f} "
                  f"spearman={rep['spearman']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = out_dir / f"{group}_per_fold.csv"
    df.to_csv(per_fold, index=False)

    metric_cols = ["spearman", "ndcg", "bedroc", "auroc", "rie", "ef@0.01", "ef@0.05"]
    summary = {}
    print("\n=== SUMMARY (mean across folds) ===", flush=True)
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        summary[m] = {
            "n_folds": int(len(sub)),
            **{c: {"mean": float(sub[c].mean()), "std": float(sub[c].std())}
               for c in metric_cols},
        }
        print(f"{m:15s} " + "  ".join(
            f"{c}={sub[c].mean():.3f}" for c in ["bedroc", "auroc", "ef@0.01", "ndcg", "spearman"]
        ), flush=True)

    meta = {
        "group": group, "features": features, "epochs": epochs, "n_neg_per_active": n_neg,
        "n_pos_per_active": n_pos, "pos_weight": pos_weight, "min_gap": min_gap,
        "list_neg": list_neg, "rf_trees": rf_trees, "seed": seed,
        "hard_neg_rounds": hard_neg_rounds, "relevance_mode": relevance_mode,
        "n_rows": int(len(y)), "sources": sorted(set(src.tolist())),
        "metric_cols": metric_cols, "methods": methods, "summary": summary,
    }
    with open(out_dir / f"{group}_summary.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    curves_path = out_dir / f"{group}_loss_curves.json"
    with open(curves_path, "w") as fh:
        json.dump({"group": group, "epochs": epochs, "curves": loss_curves}, fh, indent=2)
    print(f"\nwrote {per_fold}\nwrote {out_dir / f'{group}_summary.json'}\n"
          f"wrote {curves_path}", flush=True)
    if task is not None:
        _finish_task(task, summary, {"per_fold": per_fold, "loss_curves": curves_path,
                                     "summary_json": out_dir / f"{group}_summary.json"})
    return df, meta


# --------------------------------------------------------------------------- #
def run_cv(group: str, epochs: int, n_neg: int, n_pos: int, pos_weight: float,
           min_gap: float, list_neg: int, seed: int, rf_trees: int, out_dir: Path,
           cv_split_csv: str | None = None, log_to_clearml: bool = False,
           features: str = "rdkit", device: str = "cpu", skip_rf: bool = False,
           hard_neg_rounds: int = 0, relevance_mode: str = "bins"):
    """Chemical-similarity CV variant of the gate.

    Reuses the production BalanceSplit fold assignment (per-COMPOUND, chemically
    balanced across tasks) instead of leave-one-source-out: every assay appears in
    every training fold, and the held-out fold is unseen *chemistry*. One scorer per
    organism group; each held-out compound gets ONE score; rank metrics are computed
    WITHIN each assay of the held-out fold (its own threshold) and then averaged over
    assays and folds (never pooling scores across assays).
    """
    groups = load_organism_groups()
    if group not in groups:
        raise SystemExit(f"unknown group {group!r}; choices: {list(groups)}")
    print(f"=== RankNet gate (CV): group={group} | features={features} ===", flush=True)

    feat_path, out_dir = _resolve_features(features, out_dir)
    task = None
    if log_to_clearml:
        task = _init_task(group, "cv", {
            "group": group, "eval_mode": "cv", "features": features, "epochs": epochs,
            "n_neg_per_active": n_neg, "n_pos_per_active": n_pos,
            "pos_weight": pos_weight, "min_gap": min_gap, "list_neg": list_neg,
            "rf_trees": rf_trees, "seed": seed, "skip_rf": skip_rf,
            "hard_neg_rounds": hard_neg_rounds, "relevance_mode": relevance_mode})

    feat_table = load_feature_table(feat_path)
    wide = pd.read_csv(data_access.wide_csv())
    data = build_group_table(wide, groups[group], feat_table)
    X, y, src, grp, thr = (data["X"], data["y"], data["source"],
                           data["task"], data["threshold"])

    if cv_split_csv is None:
        cv_split_csv = data_access.split_1x5_csv()
    sp = pd.read_csv(cv_split_csv, usecols=["canonicalized_smiles", "Folds"])
    smi2fold = dict(zip(sp["canonicalized_smiles"], sp["Folds"], strict=True))
    fold_arr = data["long"]["smiles"].map(smi2fold).to_numpy(dtype=float)
    valid = ~np.isnan(fold_arr)
    folds = sorted({int(f) for f in fold_arr[valid]})
    print(f"long table: {len(y):,} rows | CV folds mapped {int(valid.sum()):,}/{len(fold_arr):,} "
          f"-> {len(folds)} folds from {os.path.basename(cv_split_csv)}", flush=True)

    rows = []
    loss_curves = {"ranknet": {}, "lambdaloss": {}, "pointwise_mlp": {}}
    for k in folds:
        te = (fold_arr == k) & valid
        tr = (fold_arr != k) & valid
        Xtr, ytr, gtr, thr_tr = X[tr], y[tr], grp[tr], thr[tr]
        Xte, yte, gte, thr_te = X[te], y[te], grp[te], thr[te]
        val_assays = [a for a in sorted(set(gte.tolist()))
                      if int((yte[gte == a] >= float(thr_te[gte == a][0])).sum()) > 0]
        print(f"\n--- fold {k}  (test n={int(te.sum()):,} over {len(val_assays)} assays, "
              f"train n={int(tr.sum()):,}) ---", flush=True)
        if not val_assays:
            print("  no assay with actives -> skipping fold", flush=True)
            continue

        scored, hists = _fit_all_methods(Xtr, ytr, gtr, thr_tr, Xte, epochs, n_neg,
                                         n_pos, pos_weight, min_gap, list_neg, rf_trees, seed,
                                         device=device, skip_rf=skip_rf,
                                         hard_neg_rounds=hard_neg_rounds,
                                         relevance_mode=relevance_mode)
        for name in loss_curves:
            loss_curves[name][f"fold{k}"] = hists[name]

        for a in val_assays:
            ma = gte == a
            thr_a = float(thr_te[ma][0])
            for m in METHODS:
                if scored.get(m) is None:
                    continue
                rep = ranking_report(yte[ma], scored[m][ma], active_threshold=thr_a)
                rep.update({"method": m, "fold": f"fold{k}", "assay": a})
                rows.append(rep)
        fold_df = pd.DataFrame([r for r in rows if r["fold"] == f"fold{k}"])
        for m in METHODS:
            sub = fold_df[fold_df["method"] == m]
            if not sub.empty:
                print(f"    {m:15s} bedroc={sub['bedroc'].mean():.3f} "
                      f"auroc={sub['auroc'].mean():.3f} ef@1%={sub['ef@0.01'].mean():.1f} "
                      f"ndcg={sub['ndcg'].mean():.3f}  (mean over {len(sub)} assays)", flush=True)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = out_dir / f"{group}_cv_per_fold.csv"
    df.to_csv(per_fold, index=False)

    summary = {}
    print("\n=== SUMMARY (CV, mean across fold×assay cells) ===", flush=True)
    for m in METHODS:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        summary[m] = {"n_cells": int(len(sub)),
                      **{c: {"mean": float(sub[c].mean()), "std": float(sub[c].std())}
                         for c in METRIC_COLS}}
        print(f"{m:15s} " + "  ".join(f"{c}={sub[c].mean():.3f}"
              for c in ["bedroc", "auroc", "ef@0.01", "ndcg", "spearman"]), flush=True)

    meta = {"group": group, "eval_mode": "cv", "features": features, "epochs": epochs,
            "n_neg_per_active": n_neg, "n_pos_per_active": n_pos, "pos_weight": pos_weight,
            "min_gap": min_gap, "list_neg": list_neg, "rf_trees": rf_trees, "seed": seed,
            "hard_neg_rounds": hard_neg_rounds, "relevance_mode": relevance_mode,
            "cv_split_csv": cv_split_csv, "n_rows": int(len(y)),
            "sources": sorted(set(src.tolist())), "metric_cols": METRIC_COLS,
            "methods": [m for m in METHODS if m in set(df["method"])], "summary": summary}
    with open(out_dir / f"{group}_cv_summary.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    curves_path = out_dir / f"{group}_cv_loss_curves.json"
    with open(curves_path, "w") as fh:
        json.dump({"group": group, "eval_mode": "cv", "epochs": epochs,
                   "curves": loss_curves}, fh, indent=2)
    print(f"\nwrote {per_fold}\nwrote {out_dir / f'{group}_cv_summary.json'}\n"
          f"wrote {curves_path}", flush=True)
    if task is not None:
        _finish_task(task, summary, {"per_fold": per_fold, "loss_curves": curves_path,
                                     "summary_json": out_dir / f"{group}_cv_summary.json"})
    return df, meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="e_coli_wt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-neg-per-active", type=int, default=100,
                   help="active-over-inactive pairs per active (training volume)")
    p.add_argument("--n-pos-per-active", type=int, default=25,
                   help="active-over-active pairs per active, ranked by %%GI (0 disables)")
    p.add_argument("--pos-weight", type=float, default=1.0,
                   help="RankNet loss weight on active-active pairs")
    p.add_argument("--min-gap", type=float, default=0.0,
                   help="skip pairs whose label gap is <= this (assay-noise DMZ)")
    p.add_argument("--list-neg", type=int, default=1000,
                   help="inactives sampled per assay-list for the LambdaLoss method")
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/ranknet_gate")
    p.add_argument("--eval-mode", choices=["lodo", "cv"], default="lodo",
                   help="lodo = leave-one-data-source-out (assay transfer); "
                        "cv = chemical-similarity CV reusing the production BalanceSplit folds")
    p.add_argument("--cv-split-csv", default=None,
                   help="wide CSV with a 'Folds' column; default is the shipped "
                        "public 1x5 split under data/")
    p.add_argument("--log_to_clearml", action="store_true", default=False,
                   help="optionally track the run in ClearML (off by default)")
    p.add_argument("--features", choices=["rdkit", "chemeleon"], default="rdkit",
                   help="feature source: 200-dim RDKit descriptors (default) or 2048-dim "
                        "frozen CheMeleon embeddings (run extract_embeddings.py first)")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="device for the MLP methods (RF is always CPU). RDKit is CPU-fine; "
                        "use cuda for the 2048-dim CheMeleon run. Pin CUDA_VISIBLE_DEVICES.")
    p.add_argument("--skip-rf", action="store_true", default=False,
                   help="skip the Random Forest baseline (fast GPU tuning sweeps)")
    p.add_argument("--hard-neg-rounds", type=int, default=0,
                   help="RankNet hard-negative-mining rounds after a random-neg warmup "
                        "(0 = off); resamples negatives from the model's top-scoring inactives")
    p.add_argument("--lambda-relevance", choices=["bins", "continuous"], default="bins",
                   help="LambdaLoss graded relevance: quartile bins or continuous %%GI min-max")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = a.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    common = dict(log_to_clearml=a.log_to_clearml, features=a.features, device=device,
                  skip_rf=a.skip_rf, hard_neg_rounds=a.hard_neg_rounds,
                  relevance_mode=a.lambda_relevance)
    if a.eval_mode == "cv":
        run_cv(a.group, a.epochs, a.n_neg_per_active, a.n_pos_per_active, a.pos_weight,
               a.min_gap, a.list_neg, a.seed, a.rf_trees, Path(a.out_dir),
               cv_split_csv=a.cv_split_csv, **common)
    else:
        run(a.group, a.epochs, a.n_neg_per_active, a.n_pos_per_active, a.pos_weight,
            a.min_gap, a.list_neg, a.seed, a.rf_trees, Path(a.out_dir), **common)
