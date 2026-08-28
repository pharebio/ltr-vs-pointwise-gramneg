"""Local batch runner for the public benchmark registry (issue #3).

Executes the configs enumerated by ``experiment_registry`` and scores them with OOF
pooling. Designed for an unattended multi-hour batch:

  resumable         -- each config writes results/benchmark/<run_id>.json; existing files
                       are skipped, so an interrupted batch resumes where it stopped.
  failure-isolated  -- one config raising records a failure row and the batch continues;
                       a single bad cell must not cost the other 125.
  cache-friendly    -- configs are grouped by (scope, features) so the long table and
                       feature matrix are built ONCE per group rather than per config
                       (126 configs share only 12 such combinations; the Morgan matrix
                       alone takes ~1 min and 1.7 GB to build).

OOF POOLING (the scoring decision): every fold contributes predictions for its held-out
rows into one out-of-fold vector, and each assay is scored ONCE on that full vector.
Per-fold scoring would bottom out at 8 actives for the sparse assays; pooled, every assay
has its full 58-2,596. Metrics are always computed WITHIN an assay, then averaged across
the assays of a scope -- %GI and pMIC are never pooled numerically.

    micromamba run -n learn2rank python run_benchmark.py --dry-run
    micromamba run -n learn2rank python run_benchmark.py --arch xgb --scope S1_wt_ecoli
    micromamba run -n gram_negative python run_benchmark.py --arch gnn --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from chemprop_ranker_lodo import subsample_train  # stdlib-only at import time
from curate_public_data import (
    OUT_CSV,
    S1_WT_ECOLI,
    S2_ECOLI_PLUS_PMIC,
    SMILES_COL,
    eligible_tasks,
    hit_threshold,
)
from experiment_registry import (
    SCORING,
    base_loss,
    enumerate_experiments,
    run_id,
    uses_global_z,
)
from make_public_splits import CV_CSV, LODO_META, source_of
from metrics import ranking_report

OUT_DIR = Path("results/benchmark")
METRICS = SCORING["metrics"]
# Validation split for the GNN arms (Ian, 2026-08-09): gives a val curve and enables early
# stopping. Carved from each fold's TRAIN compounds -- never the test fold, which is what
# OOF scoring reports. Applied to ALL GNN losses so the within-architecture mse-vs-ranking
# comparison stays clean; the fixed-feature arms (rf/mlp/xgb) keep a flat 30 epochs, a
# protocol difference to state in the paper.
VAL_FRAC = 0.10
EARLY_STOP_PATIENCE = 5          # 30 epochs / patience 5

# Benchmark inputs, all under data/. The combined public dataframe and the fold
# assignments ship with the repo; the feature tables are regenerated locally (see
# compute_rdkit_features.py / extract_embeddings.py).
CV5X5_CSV = "data/public_5x5_split.csv.gz"
_LOCAL = {"wide": OUT_CSV, "cv": CV_CSV, "lodo": LODO_META, "cv5x5": CV5X5_CSV}


def resolve_data(kind: str) -> str:
    """Local path to a benchmark input under data/."""
    import os

    local = _LOCAL[kind]
    if not os.path.exists(local):
        raise FileNotFoundError(
            f"benchmark input {kind!r} not found at {local}. The public dataframe and "
            f"split files ship under data/; regenerate splits with make_public_splits.py "
            f"if needed.")
    return local


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def scope_tasks(wide: pd.DataFrame, scope: str) -> list[str]:
    pub = [c for c in wide.columns if c != SMILES_COL]
    spec = {"S1_wt_ecoli": S1_WT_ECOLI, "S2_ecoli_pmic": S2_ECOLI_PLUS_PMIC,
            "S3_multispecies": pub}[scope]
    keep, _ = eligible_tasks(wide, spec)
    return keep


def build_long(wide: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """One row per (compound, task) actually TESTED, tagged with source + threshold."""
    frames = []
    for t in tasks:
        sub = wide[[SMILES_COL, t]].dropna(subset=[t])
        frames.append(pd.DataFrame({
            "smiles": sub[SMILES_COL].to_numpy(),
            "row": sub.index.to_numpy(),
            "y": sub[t].to_numpy(dtype=float),
            "task": t,
            "source": source_of(t),
            "threshold": hit_threshold(t),
        }))
    return pd.concat(frames, ignore_index=True)


def build_features(smiles: np.ndarray, kind: str):
    """Feature matrix for the fixed-representation arms. Returns (X, found).

    rdkit and chemeleon are both SMILES-keyed parquets, so they share
    ``features.load_feature_table`` / ``align_features`` (the same path the gate uses);
    morgan is computed on the fly.
    """
    if kind == "morgan":
        from features import morgan_features
        return morgan_features(smiles)
    if kind in ("rdkit", "chemeleon"):
        import data_access
        from features import align_features, load_feature_table
        path = (data_access.features_parquet() if kind == "rdkit"
                else data_access.chemeleon_parquet())
        return align_features(smiles, table=load_feature_table(path))
    raise ValueError(f"unknown feature kind {kind!r}")


def fold_assignments(long: pd.DataFrame, split: str, scope: str,
                     repeat: int | None = None) -> np.ndarray:
    """Per-row fold id. CV: the compound's BalanceSplit fold. LODO: the index of the
    fold whose held-out SOURCE produced that row (so 'held out' == that source's rows).

    repeat: for 5x5 repeated CV, use the `repeat`-th BalanceSplit (column Folds_<repeat>
    of the 5x5 file) instead of the single 1x5 split. Ignored for LODO."""
    if split == "cv":
        if repeat is not None:
            sp = pd.read_csv(resolve_data("cv5x5"))
            col = f"Folds_{repeat}"
            if col not in sp.columns:
                raise ValueError(f"5x5 split has no column {col!r}")
        else:
            sp = pd.read_csv(resolve_data("cv"))
            col = "Folds" if "Folds" in sp.columns else sp.columns[-1]
        m = dict(zip(sp[SMILES_COL], sp[col], strict=True))
        return long["smiles"].map(m).to_numpy(dtype=float)
    with open(resolve_data("lodo")) as fh:
        lodo = json.load(fh)
    order = [f["held_out_source"] for f in lodo["scopes"][scope]["folds"]]
    idx = {s: k for k, s in enumerate(order)}
    return long["source"].map(idx).to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def report_point(title: str, series: str, value: float, iteration: int) -> None:
    """One ClearML scalar point. No-op when untracked; never raises.

    Loss histories are otherwise only written to JSON and plotted afterwards by
    make_figures.py -- this makes the curve visible WHILE a job runs, which matters most
    for the multi-hour GNN arms. Reported explicitly rather than through ClearML's
    Lightning/TB auto-binding, to stay clear of the auto-binding footguns.
    """
    from clearml import Task

    task = Task.current_task()
    if task is None:
        return
    try:
        task.get_logger().report_scalar(title, series, value=value, iteration=iteration)
    except Exception:  # noqa: BLE001 -- telemetry must never kill a run
        pass


def report_curve(title: str, series: str, history) -> None:
    """Push a whole loss history at once (for arms that only return it after fitting)."""
    for i, v in enumerate(history or []):
        report_point(title, series, float(v), i)


def sibling_percentile_score(P: np.ndarray, te_inv: np.ndarray,
                             cols: np.ndarray) -> np.ndarray:
    """Score each test row from a multi-task head matrix ``P`` [n_unique, n_tasks].

    ``cols[i]`` is the column of the head that owns row i's assay, or -1 when the model
    has no head for it. ``te_inv[i]`` maps row i to its compound's row in P.

    A row WITH a head is scored by that head directly -- how production ranks
    within-assay. A row WITHOUT one falls back to the mean PERCENTILE RANK across all
    heads, the ``within_group_sibling`` convention for held-out assays.
    Percentile rather than raw score because the heads are separately unscaled and their
    magnitudes are not comparable; only the ordering each sibling induces is meaningful.

    The fallback is not an edge case under LODO but the norm: every task belongs to
    exactly one source, so holding out a source deletes the head for precisely the assay
    being tested. Without it every assay is NaN and the config dies in OOF pooling.
    """
    from scipy.stats import rankdata

    out = np.full(len(cols), np.nan, dtype=float)
    ok = cols >= 0
    out[ok] = P[te_inv[ok], cols[ok]]
    if not ok.all():
        pct = rankdata(P, axis=0) / len(P)              # [n_unique, n_tasks] per-head pct
        out[~ok] = pct.mean(axis=1)[te_inv[~ok]]
    return out


def torch_cat_predict(trainer, model, dset):
    """Stock chemprop/Lightning predict -> a single [n, n_tasks] array."""
    import torch
    from chemprop.data import build_dataloader

    batches = trainer.predict(model, build_dataloader(dset, batch_size=256, shuffle=False))
    return torch.cat([b for b in batches]).cpu().numpy().reshape(len(dset), -1)


def _provenance() -> dict:
    """ClearML task id, git commit and UTC timestamp for the run producing this result."""
    import datetime
    import subprocess

    out = {"finished_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    try:
        from clearml import Task
        t = Task.current_task()
        out["clearml_task_id"] = t.id if t is not None else None
    except Exception:  # noqa: BLE001 -- provenance must never kill a run
        out["clearml_task_id"] = None
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            check=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:  # noqa: BLE001
        out["git_commit"] = None
    return out


def fit_predict(cfg: dict, Xtr, ytr, gtr, thr_tr, Xte, smi_tr=None, smi_te=None,
                g_te=None):
    """Train one config on a fold and return held-out scores (higher = better ranked).

    ``g_te`` (per-test-row assay) is used only by the multi-task arm, which reads each
    row's score from that row's OWN head.
    """
    arch, seed = cfg["arch"], cfg["seed"]
    # "lambdamart_full" trains exactly like "lambdamart"; only its neg_pool/n_neg differ,
    # and the distinct name exists to keep run_id unique (see registry.base_loss).
    loss = base_loss(cfg["loss"])

    # -1 = every core, correct when this process owns the box. Running several configs
    # concurrently makes that wrong: each would claim all 32 cores and thrash. Note
    # OMP_NUM_THREADS does NOT cap sklearn's RF -- its parallelism is joblib threads over
    # trees, not BLAS -- so the limit has to be passed explicitly.
    n_jobs = cfg.get("n_jobs", -1)

    if loss == "pointwise_norm" and arch in ("rf", "mlp", "xgb"):
        # Per-assay z-scoring of the TARGET, the fixed-feature counterpart of the GNN's
        # fit_pointwise_grouped. These arms pool every assay into one regression target
        # with no task indicator, so on S2/S3 the pMIC head (~4-10) shares an axis with
        # %GI (0-100) and is drowned by its variance. Standardising within assay puts
        # every endpoint on a comparable scale before the single model sees it.
        # Train statistics only -- there is no test-side transform to leak, since the
        # scores are used for WITHIN-assay ranking and rank metrics are order-only.
        ytr = np.asarray(ytr, dtype=float).copy()
        for a in np.unique(gtr):
            m = gtr == a
            mu, sd = ytr[m].mean(), ytr[m].std()
            ytr[m] = (ytr[m] - mu) / (sd if sd > 0 else 1.0)
        loss = "mse"                     # identical objective from here on

    if arch == "rf":
        from run_ranknet_gate import fit_score_rf
        return fit_score_rf(Xtr, ytr, Xte, seed, n_jobs=n_jobs)

    if arch == "xgb":
        if loss == "mse":
            from xgb_scorer import fit_score_xgb_pointwise
            return fit_score_xgb_pointwise(Xtr, ytr, Xte, seed=seed, n_jobs=n_jobs)
        from xgb_scorer import fit_score_xgb_lambdamart
        # The pair budget must scale with the pool, or uncapping neg_pool just spreads a
        # fixed number of sampled pairs over ~100x more documents (see LOSS_HPARAMS).
        extra = ({"lambdarank_num_pair_per_sample": cfg["num_pair_per_sample"]}
                 if cfg.get("num_pair_per_sample") else {})
        if cfg.get("ndcg_k"):
            # XGBoost expresses NDCG@k as topk pair construction truncated at k, so the
            # same knob that budgets pairs also sets the cutoff -- unlike our torch loss,
            # where the two are independent.
            extra = {"lambdarank_pair_method": "topk",
                     "lambdarank_num_pair_per_sample": cfg["ndcg_k"],
                     "eval_metric": f"ndcg@{cfg['ndcg_k']}"}
        return fit_score_xgb_lambdamart(Xtr, ytr, gtr, Xte, seed=seed, n_jobs=n_jobs,
                                        relevance_mode=cfg["relevance"],
                                        levels=cfg.get("levels", 4), **extra)

    if arch == "mlp":
        from run_ranknet_gate import (
            fit_score_lambdaloss,
            fit_score_pointwise,
            fit_score_ranknet,
        )
        # NB these gate helpers return TUPLES, not bare score arrays -- pointwise and
        # lambdaloss give (scores, loss_hist), ranknet gives (scores, counts, hist).
        # Returning them raw produced
        #   "setting an array element with a sequence ... detected shape was (2,)/(3,)"
        # when assigned into the OOF vector. fit_score_rf is the odd one out (bare array),
        # which is why only the MLP arms failed.
        dev = cfg.get("device_torch", "cpu")
        if loss == "mse":
            scores, hist = fit_score_pointwise(Xtr, ytr, Xte, epochs=30, seed=seed,
                                               device=dev)
        elif loss == "ranknet":
            scores, _, hist = fit_score_ranknet(Xtr, ytr, gtr, thr_tr, Xte, epochs=30,
                                                n_neg=cfg.get("n_neg", 8), n_pos=1,
                                                pos_weight=1.0, min_gap=0.0, seed=seed,
                                                device=dev)
        else:
            scores, hist = fit_score_lambdaloss(Xtr, ytr, gtr, thr_tr, Xte, epochs=30,
                                                list_neg=cfg.get("list_neg", 256),
                                                list_act=cfg.get("list_act"),
                                                levels=cfg.get("levels", 4),
                                                gain=cfg.get("gain", "exp"),
                                                ndcg_k=cfg.get("ndcg_k"),
                                                seed=seed, device=dev,
                                                relevance_mode=cfg["relevance"])
        # these arms only expose the history after fitting, so the curve lands per fold
        report_curve("train_loss", f"fold{cfg.get('_fold', 0)}", hist)
        if scores is None:
            # the helpers return scores=None when no pairs/lists could be sampled for the
            # fold (e.g. an assay with too few actives) -- surface it, do not store None
            raise RuntimeError(f"{loss}: no pairs/lists could be sampled for this fold")
        return scores

    if arch == "gnn_mt":
        # Production-style multi-task D-MPNN, assembled from STOCK chemprop parts -- no
        # custom head, loss, or training loop. RegressionFFN(n_tasks) is the multi-task
        # head and MPNN.training_step already does the NaN-masked multi-task MSE
        # (mask = targets.isfinite(); criterion(preds, targets, mask, ...)), so Lightning
        # drives it directly. The only bespoke piece is loading the CheMeleon weights.
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.models import MPNN
        from chemprop.nn import MeanAggregation, RegressionFFN
        from chemprop.nn.transforms import UnscaleTransform
        from lightning import pytorch as pl
        from rdkit import Chem

        from ranking_chemprop import MIN_DELTA_REL, build_chemeleon_mp

        # long rows -> wide [unique compound x assay], NaN where untested: exactly the
        # layout chemprop's mask expects
        assays = sorted(set(gtr.tolist()))
        col = {a: i for i, a in enumerate(assays)}
        uniq, inv = np.unique(np.asarray(smi_tr), return_inverse=True)
        Y = np.full((len(uniq), len(assays)), np.nan, dtype=float)
        Y[inv, [col[a] for a in gtr]] = ytr

        # Ada has Tensor Cores; Lightning warns that the fp32 default leaves them idle.
        # "high" uses TF32 for matmuls -- a large speedup at negligible precision cost for
        # this regression, and the GNN arms dominate the GPU budget.
        import torch as _torch
        _torch.set_float32_matmul_precision("high")

        # Validation split for val loss + early stopping. Carved from the TRAIN compounds
        # only -- the test fold is what OOF scoring reports, so using it here would leak
        # into the paper's own metric. Split by compound (uniq), preserving the
        # chemistry-holdout property of the fold itself.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(uniq))
        n_val = max(1, int(round(VAL_FRAC * len(uniq))))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        def _dset(idx):
            return MoleculeDataset([MoleculeDatapoint(mol=Chem.MolFromSmiles(uniq[i]),
                                                      y=Y[i]) for i in idx])

        tr_dset, val_dset = _dset(tr_idx), _dset(val_idx)
        # Per-task target scaling, as chemprop's own CLI does (normalize_targets +
        # UnscaleTransform on the predictor). Without it the heads train on RAW values, so
        # a task's influence on the shared trunk scales with its variance: %GI (0-100,
        # variance in the hundreds) would swamp pMIC (~4-10, variance ~1) and starve the
        # pMIC head -- the very cross-scale failure this benchmark measures. Harmless on
        # S1 (one scale) but decisive on S2/S3. Rank metrics are order-only, and the
        # UnscaleTransform restores the original units anyway.
        scaler = tr_dset.normalize_targets()
        # The val set MUST be scaled with the same scaler, as the CLI does
        # (cli/train.py:987-989). chemprop keeps output_transform in TRAIN mode during
        # validation (MPNN.on_validation_model_eval calls output_transform.train(), and
        # UnscaleTransform.forward is the identity while training), so validation
        # predictions come out z-scored. Leaving val targets raw compares z-scores to
        # %GI, making val_loss ~= mean(y)^2 + var(y) -- a near-constant offset (~140 on
        # S3 against a train_loss of ~0.8) that barely tracks model quality, and that is
        # dominated by whichever task has the largest raw scale. Since val_loss is what
        # _ValSelect uses for best-epoch selection and early stopping, that reintroduces
        # the very %GI-swamps-pMIC imbalance the train-side scaling above prevents.
        val_dset.normalize_targets(scaler)

        mp, _ = build_chemeleon_mp()
        model = MPNN(mp, MeanAggregation(),
                     RegressionFFN(n_tasks=len(assays), input_dim=mp.output_dim,
                                   output_transform=UnscaleTransform.from_standard_scaler(
                                       scaler)),
                     batch_norm=False)
        # devices=1 always: --devices auto makes Lightning attempt NCCL/DDP, which hangs
        # ChemProp model init with no error output.
        # logger=False + enable_progress_bar=False means MPNN.training_step's
        # self.log("train_loss") has NO sink -- a multi-hour fit would run completely
        # silent, with no way to see progress or spot a diverging loss. tqdm is not the
        # answer in a non-TTY worker log (thousands of lines), so emit one line every few
        # epochs, matching the cadence the ranking arms already print.
        fold_series = f"fold{cfg.get('_fold', 0)}"

        class _ValSelect(pl.Callback):
            """Keep the BEST val_loss weights; log train and val curves.

            Selection criterion is val MSE, NOT val NDCG: this is the pointwise arm, and
            picking its epoch by a ranking metric would build ranking-awareness into the
            baseline's own training procedure -- exactly the bias the benchmark is meant to
            measure, unlike a setup that selects on val_loss. The ranking
            GNN arms select on val NDCG instead, so each objective is selected by its own
            natural criterion. Same principle as per-loss neg_pool.

            MPNN.validation_step already logs val_loss (the masked multi-task MSE), so no
            extra scoring pass is needed.

            Best-EPOCH selection is the point here; early stopping is a bonus that may
            never fire -- the loss was still falling steeply at epoch 10.
            """

            def __init__(self):
                self.best = np.inf          # val_loss: lower is better
                self.best_state = None
                self.best_epoch = None
                self.bad = 0
                self.stopped_at = None

            def on_validation_epoch_end(self, trainer_, module_):
                if trainer_.sanity_checking:
                    return
                ep = trainer_.current_epoch
                tl = trainer_.callback_metrics.get("train_loss")
                vl = trainer_.callback_metrics.get("val_loss")
                if tl is not None:
                    report_point("loss", f"train_{fold_series}", float(tl), ep)
                if vl is None:
                    return
                vl = float(vl)
                report_point("loss", f"val_{fold_series}", vl, ep)

                # Same split as RankingChemPropTrainer._val_check: any gain keeps the
                # weights, only a gain above the RELATIVE floor resets patience. val_loss
                # here is a masked multi-task MSE, a different scale from the ranking
                # arms' val NDCG, which is exactly why the floor is relative.
                # both tests use the PREVIOUS best, so decide before mutating it
                first = not np.isfinite(self.best)
                gain = np.inf if first else (self.best - vl)
                meaningful = first or gain > MIN_DELTA_REL * abs(self.best)
                if gain > 0:
                    self.best, self.best_epoch = vl, ep
                    self.best_state = {k: v.detach().cpu().clone()
                                       for k, v in module_.state_dict().items()}
                if meaningful:
                    self.bad = 0
                else:
                    self.bad += 1
                    if self.bad >= EARLY_STOP_PATIENCE:
                        self.stopped_at = ep
                        trainer_.should_stop = True
                if ep % 5 == 0 or self.stopped_at is not None:
                    t_s = f"{float(tl):.4f}" if tl is not None else "n/a"
                    print(f"    [gnn_mt] epoch {ep:3d}  train {t_s}  val_mse {vl:.4f}"
                          f"{'  <- best' if self.bad == 0 else ''}", flush=True)

        sel = _ValSelect()
        trainer = pl.Trainer(max_epochs=30, accelerator="gpu", devices=1, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False,
                             callbacks=[sel])
        trainer.fit(model,
                    build_dataloader(tr_dset, batch_size=64, shuffle=True),
                    build_dataloader(val_dset, batch_size=256, shuffle=False))
        # restore the BEST epoch's weights before scoring the held-out fold
        if sel.best_state is not None:
            model.load_state_dict(sel.best_state)
        print(f"    [gnn_mt] best val_mse {sel.best:.4f} @ epoch {sel.best_epoch}"
              + (f" (early stop @ {sel.stopped_at})" if sel.stopped_at is not None
                 else " (ran all epochs)"), flush=True)

        te_uniq, te_inv = np.unique(np.asarray(smi_te), return_inverse=True)
        te_dset = MoleculeDataset([MoleculeDatapoint(mol=Chem.MolFromSmiles(s),
                                                     y=np.full(len(assays), np.nan))
                                   for s in te_uniq])
        P = torch_cat_predict(trainer, model, te_dset)      # [n_uniq, n_tasks]
        if g_te is None:
            raise ValueError("gnn_mt needs g_te to read each row's own assay head")
        # each test ROW is scored by ITS assay's head -- how production ranks within-assay
        return sibling_percentile_score(P, te_inv,
                                        np.array([col.get(a, -1) for a in g_te]))

    if arch == "gnn":
        from ranking_chemprop import RankingChemPropTrainer, RankingMPNN
        model = RankingMPNN.from_chemeleon(head_hidden=256)
        tr = RankingChemPropTrainer(model, device=cfg.get("device_torch", "cuda"))

        # Stream the curves WHILE the fold runs. report_curve below only fires once the
        # fit returns, which on the pointwise arms is ~85 min of silence per fold.
        fold_series = f"fold{cfg.get('_fold', 0)}"

        def _on_epoch(tag, ep, train_loss, val_metric):
            report_point("loss", f"train_{fold_series}", train_loss, ep)
            if val_metric is not None:
                crit = "mse" if loss in ("mse", "pointwise_norm") else "ndcg"
                report_point("val", f"{crit}_{fold_series}", val_metric, ep)

        tr.on_epoch = _on_epoch

        # Same val protocol as gnn_mt: hold out VAL_FRAC of the fold's TRAIN compounds
        # (by compound, never the test fold) for best-epoch selection. Criterion is per
        # objective -- fit_pointwise selects on val MSE, the ranking fits on val NDCG.
        rng = np.random.default_rng(seed)
        u = np.unique(np.asarray(smi_tr))
        v_smi = set(u[rng.permutation(len(u))[:max(1, int(round(VAL_FRAC * len(u))))]])
        is_val = np.array([s in v_smi for s in smi_tr])
        if is_val.all() or not is_val.any():        # degenerate; train on everything
            is_val = np.zeros(len(smi_tr), dtype=bool)
        fit_smi = [s for s, m in zip(smi_tr, is_val, strict=True) if not m]
        val = ([s for s, m in zip(smi_tr, is_val, strict=True) if m],
               ytr[is_val], gtr[is_val]) if is_val.any() else None
        f_y, f_g, f_thr = ytr[~is_val], gtr[~is_val], thr_tr[~is_val]
        thr_map = {g: float(f_thr[f_g == g][0]) for g in np.unique(f_g)}

        if loss == "pointwise_norm":
            # per-assay z-scored MSE: the STEELMAN pointwise baseline on cross-scale
            # scopes. Plain "mse" z-scores globally, which E13 measured as scale-broken on
            # pMIC (BEDROC 0.098 vs 0.291 here) -- without this arm the cross-scale
            # ranking result would be measured against a baseline we already know is
            # crippled. Cross-scale scopes only; identical to mse on S1.
            tr.fit_pointwise_grouped(fit_smi, f_y, f_g, epochs=30, verbose=True,
                                     val=val, patience=EARLY_STOP_PATIENCE)
        elif loss == "mse":
            tr.fit_pointwise(fit_smi, f_y, epochs=30, verbose=True, val=val,
                             patience=EARLY_STOP_PATIENCE)
        elif loss == "ranknet":
            # n_neg_per_active defaulted to 20 against the registry's declared 8, so this
            # arm was training on 2.5x the pairs the manifest claimed
            tr.fit_pairwise(fit_smi, f_y, f_g, thr_map, epochs=30, verbose=True,
                            hard_neg_frac=cfg.get("hard_neg_frac", 0.33),
                            n_neg_per_active=cfg.get("n_neg", 8),
                            n_act_per_batch=cfg.get("n_act_per_batch", 64),
                            val=val, patience=EARLY_STOP_PATIENCE)
        else:
            # list geometry from the registry, not the function defaults: list_neg
            # defaulted to 384 here against the declared 256, so the GNN and MLP
            # LambdaLoss arms were training on different list shapes
            tr.fit_listwise(fit_smi, f_y, f_g, thr_map, epochs=30, verbose=True,
                            relevance_mode=cfg["relevance"],
                            list_act=cfg.get("list_act", 128),
                            list_neg=cfg.get("list_neg", 384),
                            rel_levels=cfg.get("levels", 4),
                            gain=cfg.get("gain", "exp"),
                            val=val, patience=EARLY_STOP_PATIENCE)
        # No report_curve here: the on_epoch hook above already streamed both curves.
        # Replaying them at fold end duplicated every point -- visibly on "val" (a twin
        # ndcg_foldK / foldK series pair) and silently on "loss", where both paths used
        # the same series name and just rewrote each iteration.
        return tr.score(list(smi_te))

    raise ValueError(f"unknown arch {cfg['arch']!r}")


# --------------------------------------------------------------------------- #
# one config, OOF-pooled
# --------------------------------------------------------------------------- #
def run_config(cfg: dict, data: dict, oof_path: str | Path | None = None) -> dict:
    """Train across all folds, pool OOF predictions, score each assay once.

    ``oof_path`` persists the raw out-of-fold vector. Worth the ~5-15 MB: from it any
    metric can be recomputed without retraining -- a different BEDROC alpha, EF fraction,
    NDCG@k, an ensemble, a scatter plot -- and the per-fold values a repeated-measures
    comparison (Ash et al.) consumes are derivable too. The alternative is a re-run, which
    for a single GNN config is ~5 h and for the queued GPU set is over a week. Model
    weights are deliberately NOT saved (RF alone would be ~1 TB across the matrix, and
    nobody reloads them); the predictions are what carry the reusable information.
    """
    y, grp, thr, folds = data["y"], data["task"], data["threshold"], data["folds"]
    X, smi = data.get("X"), data["smiles"]

    oof = np.full(len(y), np.nan)
    per_fold = []
    for k in sorted({int(f) for f in folds[~np.isnan(folds)]}):
        te, tr = (folds == k), (folds != k) & ~np.isnan(folds)
        if cfg.get("disjoint_lodo"):
            # F1 fix (off by default): under LODO the fold array maps each row to its
            # source, so a compound measured in BOTH the held-out and a retained source
            # lands in test AND train -- the model sees the molecule (correlated
            # cross-strain labels) before scoring it, up to 96% of the test set on the
            # worst fold. Enforce the compound-disjoint partition the split builder
            # actually wrote: drop from train every row whose compound is in this fold's
            # test set. A no-op under CV (each compound already lives in one fold).
            # Hash-based membership, NOT np.isin: np.isin on object/string arrays is
            # O(n*m) and effectively hangs on the large held-out sources (~327k SMILES).
            test_cmpds = set(smi[te].tolist())
            tr &= np.fromiter((s not in test_cmpds for s in smi),
                              dtype=bool, count=len(smi))
        if not te.any() or not tr.any():
            continue
        t0 = time.time()

        # Cap the TRAINING inactive pool (all actives + <=neg_pool inactives per assay),
        # exactly as the D-MPNN driver does. This is a fairness requirement, not just a
        # speed one: the GPU arm has always subsampled, so leaving the CPU arms on the
        # full ~480k rows would train the architectures on different data and confound
        # the cross-architecture comparison the paper makes. Evaluation is untouched --
        # every held-out row is still scored.
        ti = np.flatnonzero(tr)
        keep = subsample_train(smi[tr], y[tr], grp[tr], thr[tr],
                               cfg.get("neg_pool"), cfg["seed"])
        ti = ti[keep]

        scores = fit_predict(
            {**cfg, "_fold": k},          # names this fold's loss-curve series in ClearML
            None if X is None else X[ti], y[ti], grp[ti], thr[ti],
            None if X is None else X[te],
            smi_tr=smi[ti], smi_te=smi[te], g_te=grp[te],
        )
        oof[te] = scores
        # per-fold numbers are kept ONLY for variance reporting (see SCORING)
        for a in sorted(set(grp[te])):
            m = te & (grp == a)
            if (y[m] >= thr[m][0]).sum() > 0:
                r = ranking_report(y[m], oof[m], active_threshold=float(thr[m][0]))
                per_fold.append({"fold": k, "assay": a, **{c: r[c] for c in METRICS}})
        # n_tr is the POST-neg_pool count -- what the model actually saw. Printing the
        # pre-subsample figure overstated a ranking fold's training set by ~24x
        # (482,874 vs ~20k), and this is the number that goes in the methods section.
        print(f"    fold {k}: n_tr={len(ti):,} (of {int(tr.sum()):,} avail) "
              f"n_te={int(te.sum()):,} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- OOF pooling: score each assay ONCE on its full out-of-fold vector -------- #
    rows = []
    for a in sorted(set(grp)):
        m = (grp == a) & ~np.isnan(oof)
        if not m.any():
            # An assay no fold ever scored. Skip it loudly rather than indexing thr[m][0]
            # on an empty selection -- that IndexError killed gnn_mt/lodo after 3.4 h of
            # GPU time, reporting a config-level failure instead of "assay X unscored".
            print(f"    [warn] assay {a}: no scored rows in the OOF vector, skipped",
                  flush=True)
            continue
        n_act = int((y[m] >= thr[m][0]).sum())
        if n_act == 0:
            continue
        r = ranking_report(y[m], oof[m], active_threshold=float(thr[m][0]))
        rows.append({"assay": a, "n": int(m.sum()), "n_actives": n_act,
                     **{c: r[c] for c in METRICS}})
    if not rows:
        raise RuntimeError("no assay had actives in the OOF vector")

    if oof_path is not None:
        # float32 for the scores (rank order is all we ever use them for); labels and
        # thresholds kept exact. `fold` is float because unassigned rows are NaN.
        Path(oof_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            oof_path,
            oof=oof.astype(np.float32), y=np.asarray(y, dtype=np.float32),
            task=np.asarray(grp).astype("U"), threshold=np.asarray(thr, dtype=np.float32),
            fold=np.asarray(folds, dtype=np.float32),
            run_id=np.array(cfg["run_id"]))

    per_assay = pd.DataFrame(rows)
    agg = {c: float(per_assay[c].mean()) for c in METRICS}   # average ACROSS assays
    return {"run_id": cfg["run_id"], "config": cfg, "status": "ok",
            "aggregate": agg, "per_assay": rows, "per_fold": per_fold,
            "n_rows_scored": int((~np.isnan(oof)).sum()),
            # provenance: which ClearML task and which commit produced THIS number.
            # A run_id is stable across re-runs (it hashes only the config), so after a
            # bug fix the same id can have several tasks behind it -- without these two
            # fields there is no way to tell which task the result on disk came from.
            **_provenance()}


# --------------------------------------------------------------------------- #
# Optional ClearML experiment tracking. All identifiers are generic / configurable;
# set LEARN2RANK_CLEARML_PROJECT to change the project. Tracking is off unless
# --log_to_clearml is passed, and the benchmark runs fully locally without it.
CLEARML_PROJECT = os.environ.get("LEARN2RANK_CLEARML_PROJECT", "learn2rank")
DOCKER_IMAGE = "pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime"
# CheMeleon foundation weights (34 MB). ranking_chemprop resolves them from ~/.chemprop,
# which inside a container is /root/.chemprop -- empty unless mounted, so mount the host
# copy read-only. The mount is an OPTIMISATION, not a requirement: chemprop can download
# the file itself (zenodo.org/records/15460715/files/chemeleon_mp.pt) if absent.
CHEMELEON_DIR = os.path.expanduser("~/.chemprop")
# --no-deps stops pip replacing the image's CUDA-built torch with a PyPI wheel; the
# transitives chemprop imports eagerly are then installed explicitly.
DOCKER_SETUP = (
    "set -ex && "
    "pip install 'google-cloud-storage>=1.13.2' pyarrow && "
    "pip install --force-reinstall --no-deps 'chemprop==2.2.4' && "
    # NB astartes[MOLECULES], not plain astartes -- that is what chemprop 2.2.4 declares,
    # and the extra pulls aimsim_core -> multiprocess. Installing plain astartes (as for
    # chemprop 2.2.1) fails at `import chemprop` with
    # "ModuleNotFoundError: No module named 'multiprocess'".
    "pip install 'astartes[molecules]' ConfigArgParse descriptastorus lightning rich "
    "rdkit xgboost scikit-learn pandas scipy pydantic && "
    'python -c "import torch; assert torch.cuda.is_available(), \'no CUDA\'" && '
    # Verify what the runner imports, so a missing dep fails HERE (seconds into setup)
    # rather than after the container is built and a config has started.
    # NB do NOT check clearml here: this script runs BEFORE the agent builds the task venv
    # and installs the task's own requirements, so clearml legitimately does not exist yet
    # (asserting it failed a run). Only packages this script itself installed are fair game.
    'python -c "import chemprop, rdkit, xgboost, sklearn, pandas, scipy, pyarrow; '
    'from chemprop.models import MPNN; from chemprop.data import MoleculeDataset; '
    'print(\'deps ok\')"'
)


def _docker_args(a) -> str:
    """docker run flags every benchmark task needs (GPU pin + CheMeleon weights mount)."""
    return (f"--gpus '\"device={1 if a.gpu is None else a.gpu}\"' --shm-size=8g "
            f"-v {CHEMELEON_DIR}:/root/.chemprop:ro")


def _init_task_early():
    """Create the ClearML task BEFORE argparse runs.

    Order matters twice over:
      * execute_remotely() aborts the local process and re-enqueues, so anything above it
        would run twice; and
      * ClearML patches argparse at Task.init, so a task created FIRST has the worker's
        stored arg values injected into the later parse_args() automatically. Creating the
        task after parsing (and relying on task.connect(vars(args)) to write back) does
        NOT round-trip: the values were stored correctly but never re-applied, so a
        `--limit 1` job ran the full matrix on the worker.

    Named generically here; the real name/tags/comment are set once the args are known.
    """
    from clearml import Task

    Task.ignore_requirements("balancesplit")
    Task.force_requirements_env_freeze(force=False)
    return Task.init(
        project_name=CLEARML_PROJECT,
        task_name="public_benchmark",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False},
    )


def _apply_disjoint_lodo(exps: list[dict], enabled: bool) -> list[dict]:
    """Stamp disjoint_lodo on the LODO configs and recompute their run_id, so the F1
    compound-disjoint runs are distinct from (do not overwrite) the overlap results. CV
    configs pass through untouched -- disjointness is a no-op there."""
    if not enabled:
        return exps
    for e in exps:
        if e["split"] == "lodo":
            e["disjoint_lodo"] = True
            e["run_id"] = run_id(e)
    return exps


def _configure_task(task, a):
    """Name/tag/describe the task and, if asked, ship it to a queue (args now known)."""
    from clearml import Task

    # One task per variant: name it for the config it runs. Task.init needs a placeholder
    # because the args are not known yet; the real name is set after parse_args.
    label = a.run_id or "-".join(filter(None, [
        "+".join(a.arch or ["all"]), "+".join(a.scope or ["all"]),
        "+".join(a.split or ["all"])]))
    if Task.running_locally():
        task.set_name(f"bench_{label}")
    tags = ["public_benchmark", "neurips", *(a.arch or []), *(a.scope or [])]
    cfg = None
    if a.run_id:
        # tag by the variant's dimensions so the UI can filter/compare across tasks
        cfg = next((e for e in _apply_disjoint_lodo(enumerate_experiments(),
                                                     a.disjoint_lodo)
                    if e["run_id"] == a.run_id), None)
        if cfg:
            tags += [cfg["arch"], cfg["features"], cfg["loss"], cfg["split"], cfg["scope"]]
    task.add_tags(sorted(set(tags)))
    if a.execute_remotely:
        # Only clearml: everything else is installed globally by DOCKER_SETUP and, in
        # docker mode, the agent's venv inherits system site-packages. Letting ClearML
        # auto-pin this machine's env instead produced numpy==2.5.1 (py3.12 local) against
        # a py3.11 container -> "Could not find a version that satisfies the requirement".
        task.set_packages(["clearml"])
    # The description must name what THIS task tests. With one
    # task per variant, a batch-level "arch=all scope=all" summary is not just vague, it is
    # false -- and it would be identical across all 136 tasks, so the field would carry no
    # information at all in the UI or in search.
    scoring = (
        "Scored by OOF pooling: held-out predictions are concatenated across folds and each "
        "assay is scored ONCE on its full out-of-fold vector, metrics computed within-assay "
        f"then averaged (BEDROC alpha={SCORING['bedroc_alpha']}, EF@1%/5%, NDCG, Spearman). "
        "%GI and pMIC are never pooled numerically."
    )
    if cfg:
        pool = cfg["neg_pool"]
        crit = "val MSE" if cfg["loss"] in ("mse", "pointwise_norm") else "val NDCG"
        objective = (
            f"Public gram-negative benchmark (issue #3), variant {cfg['run_id']}: "
            f"arch={cfg['arch']} loss={cfg['loss']} features={cfg['features']} "
            f"scope={cfg['scope']} split={cfg['split']} seed={cfg['seed']}"
            + (f" relevance={cfg['relevance']}" if cfg["relevance"] else "")
            + ". neg_pool="
            + ("None (trains on EVERY row -- MSE needs no binary cutoff, so imposing one "
               "would discard the full-magnitude, all-data advantage pointwise regression "
               "has here)" if pool is None else
               f"{pool:,} inactives/assay (negative sampling is intrinsic to pair/list "
               f"construction). Evaluation is never subsampled")
            + f". {scoring}"
            + (f" Best epoch selected on {crit} from a {VAL_FRAC:.0%} compound holdout of the "
               f"fold's training set, so each objective is selected by its own natural "
               f"criterion." if cfg["arch"] in ("gnn", "gnn_mt") else "")
        )
    else:
        objective = (
            f"Public gram-negative benchmark batch (issue #3): arch={a.arch or 'all'} "
            f"scope={a.scope or 'all'} split={a.split or 'all'}. Trains every selected "
            f"registry config in one process. {scoring} neg_pool is per-loss: mse uses ALL "
            f"rows, ranking losses use 5,000 inactives/assay."
        )
    task.set_comment(objective)
    task.connect({"objective": objective})

    if a.execute_remotely:
        task.set_base_docker(
            docker_image=a.docker_image,
            docker_setup_bash_script=None if a.no_docker_setup else DOCKER_SETUP,
            docker_arguments=_docker_args(a),
        )
        if Task.running_locally():
            # execute_remotely() is a no-op on the worker, so printing this there would
            # claim a submission that is not happening -- confusing in a worker log
            print(f"[clearml] submitting task {task.id} to queue {a.queue!r} "
                  f"(image {a.docker_image})", flush=True)
        task.execute_remotely(queue_name=a.queue)  # process exits here; agent takes over
    return task


def main():
    import sys

    # Task.init MUST precede parse_args (see _init_task_early): ClearML patches argparse at
    # init, which is how the worker's stored --arch/--scope/--limit get injected.
    #
    # Deciding whether to init needs care, because the flags are not parsed yet AND the
    # agent does NOT replay them on the command line -- it passes them via the task's Args
    # section. So an argv peek alone is False on the worker, Task.init never runs, argparse
    # is never patched, and every filter silently falls back to its default: a `--limit 1`
    # job became the full 132-config matrix, three runs in a row. running_remotely() reads
    # the agent's env vars and works BEFORE Task.init, so it is the reliable signal.
    # ONE TASK PER MODEL VARIANT: each invocation handles a single config (--run-id) and
    # becomes its own ClearML task, so variants are independently queueable/retryable and
    # comparable in the UI. run_public_benchmark.sh loops the registry and calls this once
    # per config -- a shell loop over per-shard submissions, rather than a Python submitter
    # using Task.create.
    from clearml.config import running_remotely
    want_task = running_remotely() or any(
        f in sys.argv for f in ("--log_to_clearml", "--execute_remotely"))
    task = _init_task_early() if want_task else None

    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", action="append")
    ap.add_argument("--split", action="append")
    ap.add_argument("--arch", action="append")
    ap.add_argument("--features", action="append",
                    help="restrict to one or more feature sets (rdkit/morgan/chemeleon/"
                         "learned). Keeps a single feature matrix resident -- see main()")
    ap.add_argument("--gpu", type=int, help="CUDA device for the gnn arm")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="cores per CPU fit (rf/xgb/torch). -1 = all. Set this when "
                         "running several configs concurrently, or each claims every core")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, then exit")
    ap.add_argument("--force", action="store_true", help="re-run configs already on disk")
    ap.add_argument("--limit", type=int,
                    help="run at most N configs (cheapest first) -- use for a small test "
                         "job that validates the container/data/GPU path before a batch")
    ap.add_argument("--run-id",
                    help="run exactly this one config (what each submitted task executes)")
    ap.add_argument("--disjoint_lodo", action="store_true",
                    help="F1 fix: compound-disjoint LODO -- drop from training every "
                         "compound in the held-out fold's test set (no leakage across "
                         "sources). Distinct run_id (-disjoint), so it does not overwrite "
                         "the overlap LODO results. No-op for CV.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--log_to_clearml", action="store_true", help="track this batch")
    ap.add_argument("--execute_remotely", action="store_true",
                    help="submit to a ClearML queue instead of running here (implies "
                         "--log_to_clearml); survives SSH drops")
    ap.add_argument("--queue", default="default",
                    help="ClearML queue to submit to (default: 'default')")
    ap.add_argument("--docker_image", default=DOCKER_IMAGE)
    ap.add_argument("--no_docker_setup", action="store_true",
                    help="skip the pip setup script -- use with a prebuilt image from "
                         "`clearml-agent build`")
    a = ap.parse_args()

    # Task was created before parse_args (so ClearML could inject the worker's stored
    # args); now that they are known, name/tag it and -- if asked -- enqueue. Still BEFORE
    # any heavy work: execute_remotely() aborts and re-enqueues this process.
    if task is not None:
        _configure_task(task, a)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exps = _apply_disjoint_lodo(enumerate_experiments(a.scope, a.split, a.arch),
                                a.disjoint_lodo)
    if a.features:
        # Feature filter matters for MEMORY, not just selection: the runner builds one
        # matrix per (scope, features) group and reuses it across that group's configs.
        # Restricting to one feature set keeps a single matrix resident -- measured 20.7 GB
        # peak for chemeleon S1, and the 2048-dim matrices scale with row count, so mixing
        # feature sets in one process is what pushes past a RAM budget.
        exps = [e for e in exps if e["features"] in a.features]

    if a.run_id:                      # single-variant mode: what each submitted task runs
        exps = [e for e in exps if e["run_id"] == a.run_id]
        if not exps:
            raise SystemExit(f"unknown --run-id {a.run_id!r}")
    todo = [e for e in exps if a.force or not (out_dir / f"{e['run_id']}.json").exists()]
    n_done = len(exps) - len(todo)          # count BEFORE --limit, else skipped reads as done
    # Sort into EXECUTION order up front so --limit selects the configs that will actually
    # run first. Cost is driven by training-set size, which now depends on the loss
    # (mse trains on every row; ranking losses on a ~20k pooled subset), so order by that
    # rather than the unreliable est_min.
    todo.sort(key=lambda e: (e["neg_pool"] is None, e["scope"], e["features"],
                             e["split"], e["arch"], e["loss"]))
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(exps)} configs selected | {n_done} already done | "
          f"{len(todo)} to run | est {sum(e['est_min'] for e in todo)/60:.1f} h")
    if a.dry_run:
        for e in todo:
            print(f"  {e['run_id']:<44} {e['arch']:<4} {e['features']:<10} "
                  f"{e['loss']:<11} {e['split']:<5} {e['scope']}")
        return

    if a.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)  # pin BEFORE torch is imported

    if a.n_jobs and a.n_jobs > 0:
        # torch ignores n_jobs, so cap it separately; set the BLAS vars before torch and
        # numpy read them at import. rf/xgb get n_jobs passed directly (see fit_predict).
        import os
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, str(a.n_jobs))
        import torch
        torch.set_num_threads(a.n_jobs)

    wide = pd.read_csv(resolve_data("wide"))
    # group by (scope, features) so the long table + feature matrix are built once
    # (already sorted into execution order above, before --limit was applied)
    n_ok = n_fail = 0

    for (scope, feats), group in groupby(todo, key=lambda e: (e["scope"], e["features"])):
        group = list(group)
        print(f"\n=== {scope} | features={feats} | {len(group)} configs ===", flush=True)
        tasks = scope_tasks(wide, scope)
        long = build_long(wide, tasks)

        if feats == "learned":                      # gnn: SMILES in, no feature matrix
            X, keep = None, np.ones(len(long), dtype=bool)
        else:
            X, keep = build_features(long["smiles"].to_numpy(), feats)
            X = X[keep]
        long = long.loc[keep].reset_index(drop=True)
        print(f"  long table: {len(long):,} rows over {len(tasks)} assays "
              f"({int((~keep).sum()):,} dropped for missing features)", flush=True)

        base = {"X": X, "smiles": long["smiles"].to_numpy(),
                "y": long["y"].to_numpy(dtype=float), "task": long["task"].to_numpy(),
                "threshold": long["threshold"].to_numpy(dtype=float)}

        for split, cfgs in groupby(group, key=lambda e: e["split"]):
            data = {**base, "folds": fold_assignments(long, split, scope)}
            for cfg in cfgs:
                # Device comes from the REGISTRY (cfg["device"]), not from --gpu. Keying it
                # off the CLI flag meant a gnn config submitted without --gpu trained on
                # CPU: no error, just orders of magnitude slower, indistinguishable from a
                # hang. --gpu selects WHICH physical GPU the container gets (see
                # _docker_args); inside that container the pinned device is simply cuda:0.
                cfg = {**cfg, "device_torch": "cuda" if cfg["device"] == "cuda" else "cpu",
                       "n_jobs": a.n_jobs}
                print(f"  -> {cfg['run_id']}", flush=True)
                t0 = time.time()
                try:
                    oof_p = out_dir / f"{cfg['run_id']}_oof.npz"
                    res = run_config(cfg, data, oof_path=oof_p)
                    res["runtime_min"] = (time.time() - t0) / 60
                    n_ok += 1
                    print("     ok  " + "  ".join(
                        f"{c}={res['aggregate'][c]:.3f}" for c in METRICS)
                        + ("   [global-z on a cross-scale scope: ablation, NOT the "
                           "pointwise baseline -- use pointwise_norm]"
                           if uses_global_z(cfg) else ""), flush=True)
                except Exception as e:  # noqa: BLE001 -- isolate one bad cell
                    res = {"run_id": cfg["run_id"], "config": cfg, "status": "failed",
                           "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc(),
                           "runtime_min": (time.time() - t0) / 60}
                    n_fail += 1
                    print(f"     FAILED {type(e).__name__}: {e}", flush=True)
                with open(out_dir / f"{cfg['run_id']}.json", "w") as fh:
                    json.dump(res, fh, indent=1, default=str)
                if task is not None:
                    # per-config artifact + scalars, so a long batch is inspectable in the
                    # UI while it runs rather than only at the end
                    task.upload_artifact(cfg["run_id"],
                                         artifact_object=str(out_dir / f"{cfg['run_id']}.json"))
                    # the OOF vector, so any future metric is a rescore rather than a
                    # retrain -- the whole reason for keeping it
                    oof_f = out_dir / f"{cfg['run_id']}_oof.npz"
                    if oof_f.exists():
                        task.upload_artifact(f"{cfg['run_id']}_oof",
                                             artifact_object=str(oof_f))
                    if res["status"] == "ok":
                        # single VALUES, not scalars: with one task per variant these are
                        # one number each, and report_scalar renders them as a lone dot on
                        # an empty line chart. As single values they land in the Summary
                        # panel, which is also what ClearML's compare view tables across
                        # tasks -- the whole point of splitting the matrix into 136 of them.
                        log = task.get_logger()
                        for c, v in res["aggregate"].items():
                            log.report_single_value(c, float(v))
                        log.report_single_value("runtime_min",
                                                round(res.get("runtime_min", 0), 2))
                        # per_fold is one record per (fold, assay), so count DISTINCT folds
                        log.report_single_value(
                            "n_folds", len({r["fold"] for r in res.get("per_fold", [])}))
                        log.report_single_value("n_assays", len(res.get("per_assay", [])))
                        log.report_single_value("n_rows_scored",
                                                int(res.get("n_rows_scored", 0)))

    print(f"\nbatch complete: {n_ok} ok, {n_fail} failed -> {out_dir}")

    # Keep the run_id -> task index current. A run_id is stable across re-runs, so after a
    # fix the same id points at a NEW task; regenerating here means the index cannot go
    # stale by anyone forgetting to run it. Derived from the result JSONs, so it is cheap
    # and idempotent.
    try:
        import sys as _sys

        from make_task_index import main as _index
        _argv, _sys.argv = _sys.argv, ["make_task_index"]
        try:
            _index()
        finally:
            _sys.argv = _argv
    except Exception as e:  # noqa: BLE001 -- bookkeeping must not fail a finished batch
        print(f"  [index] could not regenerate task_index.md: {type(e).__name__}: {e}")

    if task is not None:
        rows = []
        for p in sorted(out_dir.glob("*.json")):
            with open(p) as fh:
                r = json.load(fh)
            rows.append({"run_id": r["run_id"], "status": r["status"],
                         "runtime_min": round(r.get("runtime_min", 0), 1),
                         # travels with the numbers so no downstream table can quote a
                         # scale-broken pointwise run as the cross-scale baseline
                         "global_z_caveat": uses_global_z(r["config"]),
                         "clearml_task_id": r.get("clearml_task_id"),
                         "git_commit": r.get("git_commit"),
                         "finished_utc": r.get("finished_utc"),
                         **{k: r["config"].get(k) for k in
                            ("arch", "features", "loss", "split", "scope")},
                         **{c: round(r.get("aggregate", {}).get(c, float("nan")), 4)
                            for c in METRICS}})
        summary = pd.DataFrame(rows)
        csv = out_dir / "benchmark_summary.csv"
        summary.to_csv(csv, index=False)
        task.upload_artifact("benchmark_summary", artifact_object=str(csv))
        task.get_logger().report_table("summary", "configs", iteration=0, table_plot=summary)
        task.get_logger().report_single_value("n_ok", n_ok)
        task.get_logger().report_single_value("n_failed", n_fail)
        print(f"wrote {csv} and uploaded it to task {task.id}")


if __name__ == "__main__":
    main()
