"""Build the public-benchmark splits: 1x5 chemically-balanced CV + per-scope LODO.

Regenerated FROM THE PUBLIC SUBSET ONLY -- computed directly on the public dataframe so
a reader can reproduce them, rather than reusing an internal split computed on a larger,
non-public set.

  1x5 CV  -- BalanceSplit (chemically balanced clustering, stratified on the task
             columns), run on the 828k public compounds. Per-compound fold id,
             scope-independent. SLOW (hours).
  LODO    -- leave-one-data-source-out, per scope. Deterministic and fast.
             test  = compounds with >=1 measurement in the held-out source's IN-SCOPE tasks
             val   = random slice of the remaining compounds (early stopping)
             train = the rest
             Folds are scope-dependent because each scope includes a different task subset.

Requires the `gram_negative` env (balancesplit lives there, not in learn2rank):
    micromamba run -n gram_negative python make_public_splits.py --lodo          # fast
    micromamba run -n gram_negative python make_public_splits.py --cv --downsample 5000
    micromamba run -n gram_negative python make_public_splits.py --cv            # hours
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from curate_public_data import (
    MIN_ACTIVES,
    OUT_CSV,
    S1_WT_ECOLI,
    S2_ECOLI_PLUS_PMIC,
    SMILES_COL,
    eligible_tasks,
    hit_threshold,
)

# Optional ClearML parent-dataset id for provenance linkage; unset by default (the split
# generation is fully local and needs no ClearML).
PUBLIC_DATASET_ID = os.environ.get("LEARN2RANK_PARENT_DATASET_ID", "")
# pandas infers gzip from the .csv.gz extension on both read and write
CV_CSV = "data/public_1x5_split.csv.gz"
LODO_JSON = "data/public_lodo_splits.json"        # full per-compound folds (internal; not shipped)
LODO_META = "data/public_lodo_splits_meta.json"   # fold order per scope (shipped; all we need)
N_SPLITS = 5
N_REPEATS = 1
N_JOBS = 20
_CV_REPEATS = N_REPEATS   # set from --n-repeats in main()
_CV_OUT = CV_CSV          # set from --out-cv in main()
VAL_FRACTION = 0.05
SEED = 0

PUBLIC_SOURCES = ["aid_0000710", "aid_1053175", "coadd_os",
                  "gadiya_pmic", "gadiya_validation", "zahed_2021"]


def source_of(task: str) -> str:
    for s in PUBLIC_SOURCES:
        if task.startswith(s):
            return s
    raise ValueError(f"unknown source for task {task!r}")


def scopes(df: pd.DataFrame) -> dict[str, list[str]]:
    """The three evaluation scopes, after the >=MIN_ACTIVES criterion."""
    pub = [c for c in df.columns if c != SMILES_COL]
    s1, _ = eligible_tasks(df, S1_WT_ECOLI)
    s2, _ = eligible_tasks(df, S2_ECOLI_PLUS_PMIC)
    s3, _ = eligible_tasks(df, pub)
    return {"S1_wt_ecoli": s1, "S2_ecoli_pmic": s2, "S3_multispecies": s3}


# --------------------------------------------------------------------------- #
# 1x5 chemically balanced CV
# --------------------------------------------------------------------------- #
def build_cv(df: pd.DataFrame, n_jobs: int = N_JOBS,
             n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """BalanceSplit n_repeats x N_SPLITS over the public compounds.

    n_repeats=1 gives the single `Folds` column the benchmark uses today. n_repeats>1
    gives one fold assignment per repeat, which is what a repeated-measures comparison
    (Ash et al. 2025) needs: repeats are the replicate unit, so a method's variance can be
    separated from fold-to-fold variance. Cost scales roughly linearly -- the 1x5 build
    took 14.85 h on 828k compounds.
    """
    from balancesplit.data import split_dataset
    from balancesplit.splitter import BalanceSplit

    t0 = time.time()
    print(f"BalanceSplit {n_repeats}x{N_SPLITS} on {len(df):,} compounds "
          f"x {len(df.columns) - 1} tasks (n_jobs={n_jobs}) ...", flush=True)
    out = split_dataset(
        data=df,
        smiles_col=SMILES_COL,
        splitter=BalanceSplit(n_splits=N_SPLITS, n_repeats=n_repeats, n_jobs=n_jobs),
        compute_stats=False,
    )
    print(f"  done in {(time.time() - t0)/60:.1f} min", flush=True)
    return out


# --------------------------------------------------------------------------- #
# per-scope LODO
# --------------------------------------------------------------------------- #
def build_lodo(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Leave-one-source-out folds for each scope (row positions into the public CSV)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    out = {"n_compounds": n, "val_fraction": VAL_FRACTION, "seed": seed,
           "min_actives": MIN_ACTIVES, "scopes": {}}

    for scope, tasks in scopes(df).items():
        by_src: dict[str, list[str]] = {}
        for t in tasks:
            by_src.setdefault(source_of(t), []).append(t)

        folds = []
        for k, (src, src_tasks) in enumerate(sorted(by_src.items())):
            test_mask = df[src_tasks].notna().any(axis=1).to_numpy()
            # only compounds measured in some IN-SCOPE task are usable at all
            in_scope = df[tasks].notna().any(axis=1).to_numpy()
            pool = np.flatnonzero(in_scope & ~test_mask)
            test = np.flatnonzero(in_scope & test_mask)
            perm = rng.permutation(pool)
            n_val = int(round(VAL_FRACTION * len(pool)))
            val, train = perm[:n_val], perm[n_val:]
            n_act = sum(int((df[t].iloc[test] >= hit_threshold(t)).sum()) for t in src_tasks)
            folds.append({"fold": k, "held_out_source": src, "tasks": src_tasks,
                          "n_train": len(train), "n_val": len(val), "n_test": len(test),
                          "n_test_actives": n_act,
                          "train": train.tolist(), "val": val.tolist(), "test": test.tolist()})
            print(f"  {scope:18s} fold{k} hold-out {src:18s} "
                  f"test={len(test):>7,} ({n_act:>5,} actives)  train={len(train):>7,}")
        out["scopes"][scope] = {"tasks": tasks, "n_folds": len(folds), "folds": folds}
    return out


def _init_task(a):
    """ClearML task for the split build. MUST be called before any heavy work so the
    task's wall-clock reflects the true BalanceSplit runtime (init must happen before any
    heavy work)."""
    from clearml import Task

    kinds = [k for k, on in (("cv", a.cv), ("lodo", a.lodo)) if on]
    # only the CV kind is NxK -- do not tack "1x5" onto a LODO-only run, whose fold count
    # is set by the number of held-out sources, not by n_splits
    suffix = f"_{a.n_repeats}x{N_SPLITS}" if a.cv else ""
    name = f"public_gram_negative_{'+'.join(kinds)}{suffix}_split"
    task = Task.init(
        project_name=os.environ.get("LEARN2RANK_CLEARML_PROJECT", "learn2rank"),
        task_name=name,
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,  # never reuse/overwrite a prior task
    )
    task.add_tags(["public_benchmark", "splits", "neurips", *kinds])
    objective = (
        f"Build the PUBLIC gram-negative benchmark splits ({'+'.join(kinds)}) for the NeurIPS "
        f"workshop paper (learn2rank issue #3). Regenerates BalanceSplit "
        f"{a.n_repeats}x{N_SPLITS} chemically-balanced CV (n_jobs={a.n_jobs}, stratified on all "
        f"14 public tasks) and/or per-scope LODO folds ON THE PUBLIC SUBSET ONLY. "
        f"The splits are computed directly on the public dataframe so a reader can "
        f"reproduce them; ~828k compounds."
    )
    task.set_comment(objective)
    # optionally record the INPUT dataset under Configuration -> Datasets
    if PUBLIC_DATASET_ID:
        link_dataset(PUBLIC_DATASET_ID, "public_gram_negative_wide_df")
    task.connect({"objective": objective, "n_splits": N_SPLITS, "n_repeats": a.n_repeats,
                  "n_jobs": a.n_jobs, "val_fraction": VAL_FRACTION, "seed": SEED,
                  "min_actives": MIN_ACTIVES,
                  "build_cv": a.cv, "build_lodo": a.lodo,
                  "downsample": a.downsample or "none"})
    return task


def _start_heartbeat(t0, period: int = 1800):
    """Console-only liveness + a main-thread stack sample every `period` seconds.

    Deliberately reports NO ClearML scalars: Task.init's auto_resource_monitoring already
    logs CPU/memory machine metrics every ~30s, at finer granularity than we would, so
    cpu/rss/elapsed series here would just be redundant. What ClearML cannot show is WHERE
    the code is -- so we sample the main thread's innermost frames. Combined with ClearML's
    CPU trace that is enough to tell the cases apart: CPU pegged = working; CPU ~0 with the
    same stack for hours = genuinely stuck; no output at all = the process died.

    Printed to stdout, which ClearML captures into the task console log. Daemon thread ->
    never blocks process exit.
    """
    import threading
    import traceback

    main_thread = threading.main_thread()

    def beat():
        while True:
            time.sleep(period)
            mins = (time.time() - t0) / 60
            frame = sys._current_frames().get(main_thread.ident)
            where = ("".join(traceback.format_stack(frame)[-2:]).rstrip()
                     if frame is not None else "  <no frame>")
            print(f"  [heartbeat] {mins:.1f} min elapsed; main thread at:\n{where}",
                  flush=True)

    threading.Thread(target=beat, daemon=True).start()


# CV and LODO are registered as SEPARATE ClearML datasets. They are independent split
# methodologies with different lifecycles -- LODO is deterministic and takes seconds, the
# 1x5 BalanceSplit takes hours to days -- and a given run consumes one or the other. Kept
# apart they version independently, and regenerating the CV split does not force a 90MB
# LODO re-upload (nor risk bundling a LODO file built by a different invocation).
_PROVENANCE = (
    "REGENERATED FROM PUBLIC DATA ONLY: computed directly on the public dataframe so it "
    "is reproducible (not an internal split computed on a larger, non-public set)."
)

DATASETS = {
    "cv": {
        "name": "public_gram_negative_1x5_split",
        "files": [CV_CSV],
        "desc": (
            f"1x{N_SPLITS} chemically-balanced CV split for the PUBLIC gram-negative "
            f"benchmark (learn2rank issue #3). BalanceSplit (n_repeats={N_REPEATS}, "
            f"n_splits={N_SPLITS}, n_jobs={N_JOBS}) stratified over all 14 public tasks; "
            f"per-compound fold id, scope-independent. {_PROVENANCE}"
        ),
    },
    "lodo": {
        "name": "public_gram_negative_lodo_splits",
        "files": [LODO_JSON, LODO_JSON.replace(".json", "_meta.json")],
        "desc": (
            f"Per-scope leave-one-data-source-out (LODO) folds for the PUBLIC "
            f"gram-negative benchmark (learn2rank issue #3). val_fraction="
            f"{VAL_FRACTION}, seed={SEED}, min_actives={MIN_ACTIVES}. Fold count is "
            f"scope-dependent (S1_wt_ecoli 4, S2_ecoli_pmic 6, S3_multispecies 6) "
            f"because each scope includes a different task subset. {_PROVENANCE}"
        ),
    },
}


def link_dataset(dataset_id: str, alias: str) -> None:
    """Force a dataset to show up under Configuration -> Datasets on the current task.

    ``Dataset.get(alias=...)`` writes ``Datasets/<alias> = <dataset_id>`` into the CURRENT
    task's hyperparameters (clearml/datasets/dataset.py: ``__hyperparams_section='Datasets'``
    -> ``current_task._set_parameters(...)``). That is the documented way to record an INPUT
    dataset, but it works just as well on a dataset we just produced -- so calling it after
    ``Dataset.create()``/``finalize()`` links our OUTPUT datasets in the UI too.

    This works because ``Dataset.create(use_current_task=False)`` builds its internal task
    with ``Task.create()``, which (unlike ``Task.init()``) does NOT become the current task
    -- so ``Task.current_task()`` is still ours when we call this. Using it avoids
    ``use_current_task=True``, which would force the dataset to inherit the TASK's name.
    """
    from clearml import Dataset

    Dataset.get(dataset_id=dataset_id, alias=alias)
    print(f"  [clearml] linked Datasets/{alias} = {dataset_id}", flush=True)


def register_clearml(kind: str, n_compounds: int) -> str:
    """Register one split kind ('cv' or 'lodo') as its own ClearML dataset.

    Child of the curated public subset. Safe to call after the fact: the split files are
    deterministic artifacts on disk, so registration never requires recomputation.
    """
    import os

    from clearml import Dataset, Task

    spec = dict(DATASETS[kind])
    # A repeated build is a DIFFERENT artifact with a different fold structure; registering
    # it under the 1x5 name would silently shadow the split the benchmark currently uses.
    if kind == "cv" and _CV_REPEATS != 1:
        spec["name"] = f"public_gram_negative_{_CV_REPEATS}x{N_SPLITS}_split"
        spec["files"] = [_CV_OUT]
        spec["desc"] = spec["desc"].replace(
            f"1x{N_SPLITS} chemically-balanced",
            f"{_CV_REPEATS}x{N_SPLITS} chemically-balanced").replace(
            f"n_repeats={N_REPEATS}", f"n_repeats={_CV_REPEATS}")
    paths = [p for p in spec["files"] if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"nothing to register for {kind!r} -- build it first")

    ds = Dataset.create(
        dataset_project=os.environ.get("LEARN2RANK_CLEARML_PROJECT", "learn2rank"),
        dataset_name=spec["name"],
        description=(f"{spec['desc']} Source data: {n_compounds:,} compounds from the "
                     f"public gram-negative dataframe. Built by make_public_splits.py."),
        parent_datasets=[PUBLIC_DATASET_ID] if PUBLIC_DATASET_ID else None,
        dataset_version=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    for p in paths:
        ds.add_files(p)
    ds.upload()
    ds.finalize()
    print(f"ClearML dataset '{spec['name']}' registered: {ds.id}  ({paths})", flush=True)
    # Surface the dataset we just produced under Configuration -> Datasets on our task.
    # Never fatal: this is a UI convenience, and the dataset is already finalized above --
    # letting it raise here would lose the registration of a multi-day CV run.
    if Task.current_task() is not None:
        try:
            link_dataset(ds.id, spec["name"])
        except Exception as e:  # noqa: BLE001
            print(f"  [clearml] WARNING: could not link {spec['name']}: "
                  f"{type(e).__name__}: {e}", flush=True)
    return ds.id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="build the 1x5 BalanceSplit (slow)")
    ap.add_argument("--lodo", action="store_true", help="build the per-scope LODO splits (fast)")
    ap.add_argument("--downsample", type=int, default=None,
                    help="use only N compounds (runtime smoke test for --cv)")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS,
                    help="BalanceSplit repeats. 1 = the split the benchmark uses; 5 gives "
                         "the replicate structure a repeated-measures comparison needs "
                         "(cost scales ~linearly; the 1x5 build took 14.85 h)")
    ap.add_argument("--out-cv", default=CV_CSV)
    ap.add_argument("--register-clearml", action="store_true",
                    help="register the split files already on disk as a ClearML dataset "
                         "(child of the curated public subset); does NOT rebuild them")
    ap.add_argument("--log_to_clearml", action="store_true",
                    help="track this build as a ClearML task (init before any heavy work, "
                         "so the task wall-clock is the true runtime)")
    a = ap.parse_args()
    global _CV_REPEATS, _CV_OUT
    _CV_REPEATS, _CV_OUT = a.n_repeats, a.out_cv
    if not (a.cv or a.lodo or a.register_clearml):
        ap.error("pass --cv and/or --lodo and/or --register-clearml")

    # init FIRST: everything below is the work whose runtime we want on record
    t0 = time.time()
    task = _init_task(a) if a.log_to_clearml else None
    if task is not None:
        print(f"[clearml] task {task.id}", flush=True)
    _start_heartbeat(t0)  # console-only; useful whether or not we are tracking

    df = pd.read_csv(OUT_CSV)
    print(f"public subset: {len(df):,} compounds x {len(df.columns)-1} tasks")
    for s, t in scopes(df).items():
        print(f"  {s:18s} {len(t)} tasks")

    if a.lodo:
        print("\n=== LODO (per scope) ===")
        lodo = build_lodo(df)
        with open(LODO_JSON, "w") as fh:
            json.dump(lodo, fh)
        meta = {k: v for k, v in lodo.items() if k != "scopes"}
        meta["scopes"] = {s: {"tasks": v["tasks"], "n_folds": v["n_folds"],
                              "folds": [{kk: f[kk] for kk in
                                         ("fold", "held_out_source", "n_train", "n_val",
                                          "n_test", "n_test_actives")} for f in v["folds"]]}
                          for s, v in lodo["scopes"].items()}
        with open(LODO_JSON.replace(".json", "_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"wrote {LODO_JSON} (+ _meta.json)")
        if task is not None:
            # attach outputs to the task that produced them -- ClearML does NOT auto-capture
            # plain file writes; output_uri only says where uploads land, not what to upload
            task.upload_artifact("lodo_splits", artifact_object=LODO_JSON)
            task.upload_artifact("lodo_splits_meta",
                                 artifact_object=LODO_JSON.replace(".json", "_meta.json"))

    if a.cv:
        sub = df if a.downsample is None else df.sample(
            n=min(a.downsample, len(df)), random_state=SEED).reset_index(drop=True)
        tag = "FULL" if a.downsample is None else f"downsampled {len(sub):,}"
        print(f"\n=== {a.n_repeats}x{N_SPLITS} BalanceSplit === ({tag})")
        out = build_cv(sub, n_jobs=a.n_jobs, n_repeats=a.n_repeats)
        # keep EVERY repeat: writing one column would silently discard n_repeats-1 of them,
        # and the repeats are the whole point of a repeated-measures design
        fold_cols = [c for c in out.columns if c == "Folds" or c.startswith("Folds_")]
        if not fold_cols:
            raise SystemExit(f"no fold column in BalanceSplit output: {list(out.columns)}")
        print(f"fold columns: {fold_cols}")
        print(out[fold_cols[0]].value_counts().sort_index().to_string())
        path = a.out_cv if a.downsample is None else a.out_cv.replace(".csv", f"_ds{len(sub)}.csv")
        out[[SMILES_COL, *fold_cols]].to_csv(path, index=False)
        print(f"wrote {path}  ({datetime.now().isoformat(timespec='seconds')})")
        if task is not None:
            task.upload_artifact(f"cv_{_CV_REPEATS}x{N_SPLITS}_split", artifact_object=path)
            fold_sizes = out[fold_cols[0]].value_counts().sort_index()
            task.get_logger().report_table("splits", "fold_sizes", iteration=0,
                                           table_plot=fold_sizes.to_frame("n_compounds"))

    if task is not None:
        elapsed_min = (time.time() - t0) / 60
        task.get_logger().report_single_value("total_runtime_min", elapsed_min)
        task.get_logger().report_single_value("n_compounds", len(df))
        print(f"\nTOTAL RUNTIME: {elapsed_min:.1f} min ({elapsed_min/60:.2f} h)", flush=True)

    if a.register_clearml:
        # Register only the kinds this run actually built, each as its own dataset.
        # That keeps every registered version traceable to the invocation that produced
        # it -- no chance of pairing a fresh split with a file left over from an earlier
        # run. To register an existing file without rebuilding, pass its build flag too
        # (e.g. --lodo --register-clearml re-derives LODO in seconds, then registers).
        kinds = [k for k, built in (("cv", a.cv), ("lodo", a.lodo)) if built]
        if not kinds:
            raise SystemExit("--register-clearml needs --cv and/or --lodo: it registers "
                             "what this run built, so versions stay traceable")
        for kind in kinds:
            register_clearml(kind, len(df))


if __name__ == "__main__":
    main()
