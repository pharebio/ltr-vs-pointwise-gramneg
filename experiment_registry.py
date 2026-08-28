"""Declarative registry of every run in the public benchmark (issue #3).

The registry is the artifact we REVIEW BEFORE LAUNCHING anything: it enumerates the full
matrix, resolves each cell's hyperparameters, and estimates cost, so the batch we commit
GPU-days to is inspected first rather than discovered afterwards.

Dimensions
    scope       S1_wt_ecoli / S2_ecoli_pmic / S3_multispecies   (homogeneous -> heterogeneous)
    split       cv (1x5 BalanceSplit, per-compound) / lodo (leave-one-data-source-out)
    arch        rf / mlp / xgb / gnn
    features    rdkit / morgan / chemeleon   (gnn learns its own -> "learned")
    loss        mse / ranknet / lambdaloss / lambdamart

Validity rules (why the matrix is not a full cross product)
    rf   supports only a pointwise objective -- there is no ranking-loss RandomForest, so
         it appears once per feature set as the classical QSAR reference.
    xgb  supports mse and lambdamart; RankNet has no GBDT analogue in xgboost.
    gnn  runs on its own learned representation (CheMeleon-initialised, unfrozen), so the
         fixed-feature axis does not apply -- it is the end-to-end arm.

Scoring is OOF-pooled by default (see SCORING below).

    micromamba run -n learn2rank python experiment_registry.py              # review table
    micromamba run -n learn2rank python experiment_registry.py --manifest results/registry.csv
    micromamba run -n learn2rank python experiment_registry.py --scope S1_wt_ecoli --arch gnn
"""
from __future__ import annotations

import argparse
import hashlib

from curate_public_data import PMIC_TASK

# --------------------------------------------------------------------------- #
# Scoring policy (Ian, 2026-08-09)
# --------------------------------------------------------------------------- #
# Per-fold active counts fall as low as 8 for the sparse assays, so computing BEDROC/EF@1%
# per fold and averaging is dominated by noise. Instead, concatenate out-of-fold predictions
# across all folds and score each assay ONCE on the full OOF vector (58-2,596 actives per
# task). Per-fold numbers are retained only for variance reporting. Metrics are always
# computed WITHIN an assay and then averaged across assays -- %GI and pMIC are never pooled
# numerically.
SCORING = {"pooling": "oof", "per_fold_for_variance": True,
           "metrics": ["bedroc", "ef@0.01", "ef@0.05", "ndcg", "spearman"],
           "bedroc_alpha": 20, "within_assay": True}

SCOPES = ["S1_wt_ecoli", "S2_ecoli_pmic", "S3_multispecies"]
SPLITS = ["cv", "lodo"]
FIXED_FEATURES = ["rdkit", "morgan", "chemeleon"]

# --------------------------------------------------------------------------- #
# Measured cost model (per-fold training sizes)
# --------------------------------------------------------------------------- #
# The hand-set per-arch estimates below were 3-6x low against the first real runs, which
# defeats the point of reviewing the matrix before committing GPU-days. These rates are
# MEASURED; the fold sizes come from results/fold_sizes.json.
#
#   gnn    2.60 min/epoch on 398,838 graphs  (ClearML 2a8b8ed5, gnn_mt S1 cv fold 1,
#          epochs 0-20 in 52m03s) -> 6.52e-6 min per graph-epoch
#   rf/mlp/xgb  timed on one real S1 cv fold, rdkit (482,819 rows x 200 features):
#          rf 0.47 min, mlp 1.30 min, xgb 0.12 min; and on the pool5000 subset
#          (20,343 rows): xgb 0.02, mlp 0.02. Expressed per row*feature because the
#          feature axis matters (rdkit 200 vs morgan/chemeleon 2048).
FEATURE_DIM = {"rdkit": 200, "morgan": 2048, "chemeleon": 2048, "learned": None}
GNN_MIN_PER_GRAPH_EPOCH = 6.52e-6
CPU_MIN_PER_ROW_FEAT = {           # (arch, is_ranking_loss) -> min per row*feature
    ("rf", False): 4.846e-09,
    ("mlp", False): 1.348e-08,
    ("mlp", True): 4.707e-09,
    ("xgb", False): 1.233e-09,
    ("xgb", True): 6.082e-09,
}
# Data load + feature alignment + OOF scoring, per CONFIG, on a warm ClearML cache
# (measured: wide csv 0.10, build_long 0.00, folds 0.10, features 0.09-0.72, scoring 0.01).
# A COLD cache adds the dataset download once per worker, which is what made the first
# xgb/chemeleon config take 30.6 min end to end.
FIXED_OVERHEAD_MIN = 1.0
EPOCH_BUDGET = 30
# Observed early-stop epochs on the first GNN config: 25 / 10 / 13 across three folds
# (best + patience 5). Reported alongside the worst case rather than replacing it --
# the pointwise arms have not been measured and MSE may not plateau like val NDCG.
OBSERVED_EARLY_STOP_EPOCHS = 16

# arch -> (losses, feature sets, device, est. minutes per config)
# Runtime estimates are per CONFIG (all folds), from observed runs: the D-MPNN CV ensemble
# did 5 folds in ~52 min; RF is the slowest fixed-feature model because it fits the full
# negative pool per fold.
ARCHS = {
    # Every pointwise arm gets "pointwise_norm" on the cross-scale scopes, for the reason
    # spelled out under "gnn" below. These arms pool all of a scope's assays into ONE
    # regression target with no task indicator, so on S2/S3 the pMIC head (log scale, ~4-10)
    # is regressed on the same axis as %GI (0-100) and is swamped by its variance. Shipping
    # those cells with only plain "mse" would compare ranking against a baseline E13 already
    # measured as scale-broken (0.098 vs 0.291 BEDROC on pMIC).
    # The fixed-feature models carry "pointwise_norm" (per-assay z-scored MSE) on EVERY
    # scope, not just the cross-scale ones. It is decisive on S2/S3 (below), and although
    # S1 is single-scale its four %GI heads still differ in spread (sd 4.8-9.3, ~1.9x), so
    # per-assay z is not a strict duplicate of plain "mse" there -- and the R1 ablation
    # showed even that 1.9x spread is enough to move selection. Running it on S1 too keeps
    # the pointwise baseline defined identically across scopes (no grey cells in the
    # method x representation grid). Cheap on CPU; the GPU `gnn` arm keeps it cross-scale
    # only to save GPU time.
    "rf":  {"losses": ["mse", "pointwise_norm"], "features": FIXED_FEATURES,
            "device": "cpu", "est_min": 40},
    "mlp": {"losses": ["mse", "pointwise_norm", "ranknet", "lambdaloss"],
            "features": FIXED_FEATURES, "device": "cpu", "est_min": 15},
    "xgb": {"losses": ["mse", "pointwise_norm", "lambdamart"], "features": FIXED_FEATURES,
            "device": "cpu", "est_min": 10},
    # "pointwise_norm" (per-assay z-scored MSE) is CROSS-SCALE ONLY -- see
    # losses_for_scope(). It is the steelman pointwise baseline when endpoints have
    # incomparable scales; plain "mse" z-scores globally, which E13 measured as crippled on
    # pMIC (BEDROC 0.098 vs 0.291 for per-assay-z). Shipping S2/S3 without it would pit
    # ranking (0.53) against a baseline we already know is scale-broken and overstate the
    # ranking advantage ~3x. On S1 the two coincide (one shared %GI scale), so it is
    # omitted there rather than run as a near-duplicate.
    "gnn": {"losses": ["mse", "ranknet", "lambdaloss"], "features": ["learned"],
            "device": "cuda", "est_min": 60, "cross_scale_losses": ["pointwise_norm"]},
    # Multi-task pointwise D-MPNN: one regression head PER ASSAY + masked MSE, i.e. the
    # production-style multi-task architecture, ranked by the in-scope head(s). The
    # single-head `gnn`
    # arm above holds capacity fixed and swaps only the loss (objective isolation); this
    # arm answers the different, practical question -- does a ranking model beat the
    # model we would actually deploy? Ranking losses have no multi-task analogue: they
    # emit ONE latent score by construction, which is the premise of the cross-scale
    # test, so this arm is pointwise-only.
    "gnn_mt": {"losses": ["mse"], "features": ["learned"],
               "device": "cuda", "est_min": 60},
}

# Best-known hyperparameters carried over from the Phase-2 tuning line, so the benchmark
# runs each objective at its best configuration rather than at a default we already know
# is worse. Sources: E5 (semi-hard negatives), E10 (continuous_gi relevance beat bins,
# 0.626 -> 0.709 BEDROC), losses.py reduction="sum" (Wang 2018 / allRank).
# NOTE on `neg_pool` -- it is a property of the LOSS, not of the run.
#
# Capping the inactive pool requires a BINARY CUTOFF to decide which rows are actives worth
# keeping. That cutoff is intrinsic to the ranking objectives: RankNet builds pairs and
# LambdaLoss/LambdaMART build lists by contrasting actives against sampled negatives, so
# negative sampling is part of the method itself.
#
# MSE needs no threshold at all. It consumes the continuous %GI magnitude directly, and can
# use every measured row. Forcing a cutoff + subsample on it would impose a binarization the
# method does not require and discard ~96% of the negatives -- removing exactly the advantage
# pointwise regression has in this regime (full magnitude, all data, low-variance fitting),
# which is what the Phase-1 verdict attributed its wins to. So the pointwise arms train on
# EVERYTHING and the comparison is each objective at its natural configuration.
LOSS_HPARAMS = {
    "mse":        {"neg_pool": None},
    # per-assay z-scored MSE: same objective, normalisation is the only difference, so it
    # inherits mse's "use every row, no binary cutoff" treatment
    "pointwise_norm": {"neg_pool": None},
    "ranknet":    {"neg_pool": 5000, "hard_neg_frac": 0.33,
                   "n_act_per_batch": 64, "n_neg": 8},
    "lambdaloss": {"neg_pool": 5000, "gain": "exp", "levels": 4, "reduction": "sum",
                   "list_act": 128, "list_neg": 256},
    "lambdamart": {"neg_pool": 5000, "levels": 4},
    # --- data-matched ranking ablations (S1 only; see ABLATIONS) --------------------- #
    # The headline comparison gives each objective its natural configuration, so pointwise
    # trains on every row (~482k/fold) while ranking sees ~20k. That is the honest default,
    # but it means a single cell cannot separate OBJECTIVE from DATA VOLUME -- and on
    # rdkit/S1 the same-library contrast is stark (xgb mse 0.666 vs lambdamart 0.382).
    # These two variants remove the confound, and they remove it differently because the
    # methods are limited by different things:
    #   lambdamart_full  XGBoost's TREES see every row, but its lambda gradients come from
    #                    SAMPLED PAIRS, budgeted by lambdarank_num_pair_per_sample -- a
    #                    fixed hyperparameter that does not scale with group size. A first
    #                    attempt set only neg_pool=None; growing each assay's query group
    #                    ~100x on a fixed pair budget DILUTED supervision ~100x and CV went
    #                    backwards (0.382 -> 0.309, EF@1% 21.3 -> 15.3). So the budget is
    #                    raised 8x here too, matching what n_neg does for RankNet. (Note the
    #                    match is approximate: under the default topk pair method this is a
    #                    truncation level, not a per-active negative count.)
    #   ranknet_full     Same principle, stated first: RankNet's cost is the PAIR budget
    #                    (n_actives x n_pos x n_neg), not the pool, so uncapping alone only
    #                    diversifies which negatives get sampled. n_neg 8 -> 64.
    "lambdamart_full": {"neg_pool": None, "levels": 4, "num_pair_per_sample": 64},
    "ranknet_full":    {"neg_pool": None, "hard_neg_frac": 0.33,
                        "n_act_per_batch": 64, "n_neg": 64},
    # --- NDCG@k: is the standard listwise objective simply the wrong target? --------- #
    # Full-list NDCG barely responds to early enrichment. Measured on real assay
    # geometry (n=97,450, 58 actives): random scores 0.2655 and a perfect ranking 1.0, so
    # two thirds of the range is unreachable, and across models spanning BEDROC 0.25-0.74
    # NDCG moves only 0.74-0.84. The cause is the log discount's fat tail -- weight 1.00 at
    # rank 1, still 0.06 at rank 100,000 -- so the bulk of the objective is about ordering
    # compounds nobody will ever screen. Both NDCG-optimising methods (lambdaloss,
    # lambdamart) sit at the bottom of every column, while pairwise RankNet is the
    # strongest ranking arm. Truncating the discount at k is the direct test of whether
    # that is causal. k=32 is a starting point, NOT tuned: it is roughly the top ~8% of a
    # 384-doc training list, in the spirit of BEDROC alpha=20 concentrating on the top 5%.
    #
    # RESULT (2026-08-10): truncation does NOT rescue LambdaLoss -- mean -0.011 BEDROC
    # over six rdkit cells, signs mixed. The hypothesis was mis-aimed: NDCG's insensitivity
    # was measured on EVALUATION lists (97,450 compounds, 0.06% actives), but LambdaLoss
    # trains on lists of list_act + list_neg = 384 documents that are ~33% relevant, where
    # NDCG is already top-sensitive. The objective was never in the pathological regime, so
    # there was nothing for k to fix. Kept as a recorded negative result.
    #
    # A lambdamart_k arm was also tried and REMOVED: XGBoost ignores
    # lambdarank_num_pair_per_sample under its default topk pair method (verified --
    # predictions bit-identical, max|diff| = 0.0), so all six runs were exact duplicates of
    # lambdamart and would have read as "truncation had no effect" when no truncation
    # occurred. LambdaMART is where the sparse-list hypothesis remains live, since it
    # trains on the whole query group rather than a short sampled list -- but testing it
    # needs pair_method="mean", which changes pair construction and truncation together.
    "lambdaloss_k":  {"neg_pool": 5000, "gain": "exp", "levels": 4, "reduction": "sum",
                      "list_act": 128, "list_neg": 256, "ndcg_k": 32},
}

# Explicit one-off configs appended to the matrix. Kept as a hand-written list rather than
# extra ARCHS entries so the ablation stays scoped to the cells where we already have the
# pointwise numbers to compare against, instead of multiplying across the whole grid.
ABLATIONS = [
    {"arch": "xgb", "features": "rdkit", "loss": "lambdamart_full",
     "scope": "S1_wt_ecoli", "split": split} for split in SPLITS
] + [
    # ranknet_full swept across BOTH representations and ALL scopes, to test whether the
    # full-pair-budget rescue (S1/rdkit: cv 0.661, disjoint-LODO 0.603 vs regular RankNet
    # ~0.60) generalises to the cross-scale scopes (S2/S3) and to the strongest
    # representation (CheMeleon), where pointwise's lead is cleanest. Same comprehension
    # style as lambdaloss_k below.
    {"arch": "mlp", "features": feat, "loss": "ranknet_full",
     "scope": scope, "split": split}
    for feat in ("rdkit", "chemeleon")
    for scope in SCOPES for split in SPLITS
] + [
    # NDCG@k, MLP only -- the xgb counterpart was removed as untestable via that knob
    # (see LOSS_HPARAMS). Retained after a negative result so the comparison stays in the
    # record rather than being quietly dropped.
    {"arch": "mlp", "features": "rdkit", "loss": "lambdaloss_k",
     "scope": scope, "split": split}
    for scope in SCOPES for split in SPLITS
]

# Per-loss; see LOSS_HPARAMS below for why. Evaluation is NEVER subsampled -- every held-out
# row is scored, so metrics are always computed against the true ~0.07% sparsity.
# (Historical note: a global 5,000 cap was once applied to every arm as a workaround for our
# own `dset.cache = True` in ranking_chemprop, which precomputes every MolGraph. That was not
# a chemprop limitation -- it trains on the full set routinely -- and the measured cache cost
# is 34.6 KB/molecule, ~21 GB for an S1 fold and ~35 GB for S3, well within the 124 GB box.)
SEEDS = [0]       # single seed for the first batch; 5x5 reserved for the key contrasts


def is_cross_scale(scope: str) -> bool:
    """True if the scope mixes endpoint scales (i.e. includes the log-scale pMIC head).

    Drives the relevance mode: a global scale collapses every pMIC row to the bottom grade
    (measured in E6/E13), so cross-scale scopes must use per-assay normalisation.
    """
    from curate_public_data import S1_WT_ECOLI, S2_ECOLI_PLUS_PMIC
    tasks = {"S1_wt_ecoli": S1_WT_ECOLI, "S2_ecoli_pmic": S2_ECOLI_PLUS_PMIC}.get(scope)
    return PMIC_TASK in tasks if tasks is not None else True  # S3 includes pMIC


def base_loss(loss: str) -> str:
    """Strip the variant suffix: 'lambdamart_full' and 'lambdaloss_k' behave as their base
    objective everywhere except in their hyperparameters. Distinct NAMES (rather than CLI
    overrides) are what keep run_id unique -- run_id hashes (scope, split, arch, features,
    loss, seed), so folding neg_pool or ndcg_k into that key would rewrite every existing
    id and orphan the results already on disk.

    Note this drives relevance_mode too: without stripping, 'lambdaloss_k' would fall
    through to None and the listwise arms would silently lose graded relevance.
    """
    for suffix in ("_full", "_k"):
        loss = loss.removesuffix(suffix)
    return loss


# Which LOSS_HPARAMS keys each arch's fit path actually consumes. The registry reads as
# the specification of what was run, so a key declared here and dropped by the runner makes
# it a lie -- and that lie propagates into results/registry.csv and the methods section.
# LambdaLoss had three: list_act was not even a parameter of fit_score_lambdaloss (every
# active went into every list), list_neg was passed on the MLP arm but not the GNN one
# (which used its own default of 384 against the declared 256), and reduction was never
# passed at all -- correct only because lambdaloss_ndcg happens to default to "sum".
#
# Keyed by (arch, loss) because the arms genuinely differ: hard_neg_frac is the GNN's
# semi-hard MIX (E5), while the MLP gate's only mining knob is hard_neg_rounds -- the naive
# full-replacement variant that Phase-1 measured as harmful. They are different mechanisms
# and must not be conflated, so hard_neg_frac is deliberately GNN-only rather than forced
# onto the MLP arm. Likewise n_act_per_batch exists only in the GNN's batch sampler.
CONSUMED_HPARAMS = {
    ("mlp", "ranknet"): {"n_neg"},
    ("mlp", "ranknet_full"): {"n_neg"},
    ("gnn", "ranknet"): {"hard_neg_frac", "n_neg", "n_act_per_batch"},
    ("mlp", "lambdaloss"): {"gain", "levels", "list_act", "list_neg", "reduction"},
    ("gnn", "lambdaloss"): {"gain", "levels", "list_act", "list_neg", "reduction"},
    ("mlp", "lambdaloss_k"): {"gain", "levels", "list_act", "list_neg", "reduction",
                              "ndcg_k"},
    ("xgb", "lambdamart"): {"levels"},
    ("xgb", "lambdamart_full"): {"levels", "num_pair_per_sample"},
}
# Declared for one arch but not another. Not a bug -- it is the asymmetry above -- but it
# must be visible, or the manifest implies every arm ran with every listed value.
ARCH_SPECIFIC = {("mlp", "ranknet"): {"hard_neg_frac", "n_act_per_batch"},
                 ("mlp", "ranknet_full"): {"hard_neg_frac", "n_act_per_batch"}}


def check_hparams_consumed() -> list[str]:
    """Declared-but-unconsumed hyperparameters as 'arch+loss.key'. Empty == consistent.

    The registry reads as the specification of what ran, so a key declared here and dropped
    by the runner makes results/registry.csv wrong -- and that feeds the methods section.
    `neg_pool` is always consumed (the runner's subsample_train applies it, not a fit fn).
    """
    bad = []
    for cfg in enumerate_experiments():
        key = (cfg["arch"], cfg["loss"])
        known = (CONSUMED_HPARAMS.get(key, set()) | ARCH_SPECIFIC.get(key, set())
                 | {"neg_pool"})
        for k in LOSS_HPARAMS[cfg["loss"]]:
            if k not in known:
                bad.append(f"{cfg['arch']}+{cfg['loss']}.{k}")
    return sorted(set(bad))


def uses_global_z(cfg: dict) -> bool:
    """True when this config regresses a MIXTURE of endpoint scales on one axis.

    Plain "mse" pools every assay in a scope into a single regression target with no task
    indicator (and, for the MLP/GNN, one global mean/sd). On a cross-scale scope that puts
    pMIC (~4-10) on the same axis as %GI (0-100), where its variance is swamped: measured
    at 0.181 vs 0.414 pMIC BEDROC against pointwise_norm on S2 cv rdkit.

    Such a run is NOT invalid -- it is the ablation that demonstrates the failure, and the
    contrast is a result worth publishing. But it must never be quoted as THE pointwise
    baseline on a cross-scale scope; pointwise_norm is. Derived from the config rather than
    stored, so it applies retroactively to results already on disk.
    """
    return bool(cfg.get("loss") == "mse" and cfg.get("cross_scale"))


def relevance_mode(scope: str, loss: str) -> str | None:
    """Relevance encoding for the listwise objectives; None where it does not apply."""
    if base_loss(loss) not in ("lambdaloss", "lambdamart"):
        return None
    return "continuous_local" if is_cross_scale(scope) else "continuous_gi"


def run_id(cfg: dict) -> str:
    """Deterministic short id, so a config maps to the same run name across invocations.

    The compound-disjoint LODO variant (F1 fix) gets a distinct id and a "-disjoint" tag,
    so its results sit beside the original (overlap) LODO rather than overwriting them --
    both are needed to report the overlap->disjoint delta. Configs without the flag are
    unchanged, so every existing id is stable.
    """
    key = "|".join(str(cfg[k]) for k in
                   ("scope", "split", "arch", "features", "loss", "seed"))
    tag = ""
    if cfg.get("disjoint_lodo"):
        key += "|disjoint"
        tag = "-disjoint"
    return f"{cfg['arch']}-{cfg['features']}-{cfg['loss']}-{cfg['split']}-" \
           f"{cfg['scope'].split('_')[0]}{tag}-{hashlib.sha1(key.encode()).hexdigest()[:6]}"


def _fold_sizes(path: str = "results/fold_sizes.json") -> dict:
    """Measured per-fold training sizes, or {} when the fold sizes have not been measured."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def estimate_minutes(arch: str, scope: str, split: str, loss: str, features: str,
                     sizes: dict | None = None, epochs: int = EPOCH_BUDGET) -> float:
    """Measured-throughput cost for one config, all folds. Falls back to the per-arch
    constant when results/fold_sizes.json is absent."""
    sizes = _fold_sizes() if sizes is None else sizes
    cell = sizes.get(f"{scope}|{split}")
    if not cell:
        return float(ARCHS[arch]["est_min"])

    pool = LOSS_HPARAMS[loss]["neg_pool"]
    tag = "full" if pool is None else f"pool{pool}"
    # gnn_mt dedups to unique COMPOUNDS (multi-task target matrix); every other arm
    # trains one example per measurement ROW
    unit = "compound" if arch == "gnn_mt" else "row"
    n = sum(f[tag][unit] for f in cell["folds"] if tag in f)

    if ARCHS[arch]["device"] == "cuda":
        return FIXED_OVERHEAD_MIN + n * GNN_MIN_PER_GRAPH_EPOCH * epochs
    # the measured CPU rates are keyed on whether the loss subsamples, so an uncapped
    # ranking ablation is costed at its own (ranking) rate over the full row count
    rate = CPU_MIN_PER_ROW_FEAT[(arch, base_loss(loss) not in ("mse", "pointwise_norm"))]
    return FIXED_OVERHEAD_MIN + n * FEATURE_DIM[features] * rate


def losses_for_scope(arch: str, scope: str) -> list[str]:
    """An arch's losses for a scope, plus any cross-scale-only additions."""
    spec = ARCHS[arch]
    extra = spec.get("cross_scale_losses", []) if is_cross_scale(scope) else []
    return [*spec["losses"], *extra]


def enumerate_experiments(scopes=None, splits=None, archs=None, seeds=None) -> list[dict]:
    """The full matrix, after validity rules. Order is stable and cheap-first."""
    scopes = scopes or SCOPES
    splits = splits or SPLITS
    archs = archs or list(ARCHS)
    seeds = seeds or SEEDS

    sizes = _fold_sizes()
    out = []
    for scope in scopes:
        for split in splits:
            for arch in archs:
                spec = ARCHS[arch]
                for feats in spec["features"]:
                    for loss in losses_for_scope(arch, scope):
                        for seed in seeds:
                            cfg = {"scope": scope, "split": split, "arch": arch,
                                   "features": feats, "loss": loss, "seed": seed,
                                   "device": spec["device"],
                                   "est_min": round(estimate_minutes(
                                       arch, scope, split, loss, feats, sizes), 1),
                                   "cross_scale": is_cross_scale(scope),
                                   "relevance": relevance_mode(scope, loss),
                                   **LOSS_HPARAMS[loss]}
                            cfg["run_id"] = run_id(cfg)
                            out.append(cfg)

    for ab in ABLATIONS:
        if (ab["scope"] not in scopes or ab["split"] not in splits
                or ab["arch"] not in archs):
            continue
        for seed in seeds:
            cfg = {**ab, "seed": seed, "device": ARCHS[ab["arch"]]["device"],
                   "est_min": round(estimate_minutes(
                       ab["arch"], ab["scope"], ab["split"], ab["loss"],
                       ab["features"], sizes), 1),
                   "cross_scale": is_cross_scale(ab["scope"]),
                   "relevance": relevance_mode(ab["scope"], ab["loss"]),
                   **LOSS_HPARAMS[ab["loss"]]}
            cfg["run_id"] = run_id(cfg)
            out.append(cfg)

    out.sort(key=lambda c: (c["est_min"], c["arch"], c["scope"], c["split"]))
    return out


def summarize(exps: list[dict]) -> str:
    from collections import Counter
    lines = []
    cpu = sum(e["est_min"] for e in exps if e["device"] == "cpu")
    gpu = sum(e["est_min"] for e in exps if e["device"] == "cuda")
    lines.append(f"total configs: {len(exps)}")
    for key in ("arch", "scope", "split", "loss", "features"):
        c = Counter(e[key] for e in exps)
        lines.append(f"  by {key:9s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    lines.append(f"estimated cost: CPU {cpu/60:.1f} h | GPU {gpu/60:.1f} h "
                 f"(serial; CPU arms parallelise across cores)")
    if _fold_sizes():
        frac = OBSERVED_EARLY_STOP_EPOCHS / EPOCH_BUDGET
        lines.append(
            f"  ^ measured throughput x {EPOCH_BUDGET} epochs (worst case: no early stop)."
            f" At the {OBSERVED_EARLY_STOP_EPOCHS} epochs observed on the first GNN config,"
            f" GPU falls to ~{gpu*frac/60:.1f} h.")
    else:
        lines.append("  ^ FALLBACK per-arch constants -- measure the fold sizes for "
                     "sizes (the constants were 3-6x low against real runs)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", action="append", choices=SCOPES)
    ap.add_argument("--split", action="append", choices=SPLITS)
    ap.add_argument("--arch", action="append", choices=list(ARCHS))
    ap.add_argument("--manifest", help="write the full matrix to a CSV for review")
    a = ap.parse_args()

    exps = enumerate_experiments(a.scope, a.split, a.arch)
    hdr = f"{'run_id':<44} {'arch':<4} {'features':<10} {'loss':<11} {'split':<5} " \
          f"{'scope':<16} {'relevance':<17} {'dev':<5} {'min':>4}"
    print(hdr)
    print("-" * len(hdr))
    for e in exps:
        print(f"{e['run_id']:<44} {e['arch']:<4} {e['features']:<10} {e['loss']:<11} "
              f"{e['split']:<5} {e['scope']:<16} {str(e['relevance']):<17} "
              f"{e['device']:<5} {e['est_min']:>4}")
    print()
    print(summarize(exps))
    print(f"\nscoring: {SCORING}")

    if a.manifest:
        import csv
        keys = sorted({k for e in exps for k in e})
        with open(a.manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(exps)
        print(f"\nwrote {a.manifest} ({len(exps)} rows)")


if __name__ == "__main__":
    main()
