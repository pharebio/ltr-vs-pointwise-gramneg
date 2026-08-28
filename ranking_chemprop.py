"""
Phase-2: end-to-end learn-to-rank on the ChemProp D-MPNN encoder.

This is the one regime Phase-1 never tested. On a *frozen* representation (RDKit
descriptors and frozen CheMeleon embeddings, LODO + CV) the ranking objective never
beat pointwise regression -- because rank metrics only need monotonicity and
pointwise already gives it, using the full %GI magnitude and all the data. The
untested, scorecard-comparable question is whether letting the ranking loss
backpropagate *through* message passing -- reshaping the representation itself --
changes that verdict.

``RankingMPNN`` subclasses ``chemprop.models.MPNN`` so:
  * it reuses ``MPNN.fingerprint`` (message passing -> aggregation -> the learned
    2048-dim representation) unchanged, and
  * the encoder is initialized from the **CheMeleon foundation** (same loader as
    ``extract_embeddings.py``) but **not frozen** -- every encoder parameter has
    ``requires_grad=True``, so the ranking gradient adapts message passing.
A separate ``score_head`` maps the fingerprint to a scalar ranking score.

We drive training with an explicit loop (``RankingChemPropTrainer``) rather than
Lightning's ``training_step`` + a custom ``Sampler``. The loop realizes the same
intent described above -- within-assay pairs/lists formed *inside* each
batch, then ``losses.ranknet_loss`` / ``losses.lambdaloss_ndcg`` -- with full control
over pair formation and none of the Lightning-DDP / ClearML-PyTorch-auto-binding
footguns (no second Lightning run in-process). Because the
model is still an ``MPNN`` subclass, ``fingerprint`` and a plain ``state_dict`` save
keep working.

Environment: chemprop + CheMeleon + rdkit. Single GPU only; the caller pins
``CUDA_VISIBLE_DEVICES`` before importing torch. ChemProp **Python API only** --
never the CLI.

Design notes
------------
* All three objectives (pairwise RankNet, listwise LambdaLoss-NDCG, pointwise MSE)
  share the SAME architecture (CheMeleon-init encoder + ``score_head``); only the
  loss differs. This isolates the objective from capacity/representation -- the same
  honest comparison ``run_ranknet_gate`` made for the MLP, now on the D-MPNN.
* Batches are **assay-homogeneous** (one task/assay per batch), so within-batch pairs
  and lists are automatically within-assay -- no cross-assay or cross-endpoint
  comparison is ever formed. The caller hands us a bounded training set (all actives +
  a capped inactive pool), so we can precompute the graph cache once per fit.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHEMELEON_CKPT = os.path.expanduser("~/.chemprop/chemeleon_mp.pt")

# Relative improvement an epoch must clear to count as progress and reset early-stopping
# patience. Relative rather than absolute so the SAME constant is correct for val NDCG
# (bounded [0,1], plateaus ~0.77) and val MSE (z-scored, starts ~1.0, decays to the noise
# floor); see RankingChemPropTrainer._val_setup. 0.1% is well above the observed
# epoch-to-epoch noise (~0.0002 on val NDCG) and far below any real gain.
MIN_DELTA_REL = 1e-3


# --------------------------------------------------------------------------- #
# Graded relevance for the listwise objective (mirrors run_ranknet_gate._graded_relevance;
# duplicated here to keep this training module free of the gate's feature/ClearML imports).
# --------------------------------------------------------------------------- #
def graded_relevance(y_act: np.ndarray, levels: int = 4, mode: str = "bins") -> np.ndarray:
    """Map active %GI values to graded relevance in [1, levels]; inactives get 0
    (added by the caller). ``lambdaloss_ndcg`` uses gains = 2**rel - 1, so relevance
    must be small bounded values (raw %GI would overflow 2**label).

    mode='bins'       : quartile rank bins (1..levels) -- coarse, discards magnitude.
    mode='continuous' : min-max scale the actives' %GI to [1, levels] -- keeps the
                        continuous potency signal without exponential-gain blow-up.
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


def list_relevance(y: np.ndarray, mode: str = "continuous_gi", levels: int = 4,
                   rel_max: float = 100.0, rel_lo=None, rel_hi=None) -> np.ndarray:
    """WHOLE-LIST graded relevance from raw activity values ``y`` (one single-assay list),
    for the non-binarized LambdaLoss variant: EVERY molecule is graded by its activity, with
    no relevance-0 floor at the hit threshold -- removing the artificial cliff at 80% GI where
    a 79% near-miss currently counts identically to a 2% true negative. Bounded to [0, levels]
    so ``lambdaloss_ndcg``'s ``2**rel`` gains stay finite.

    mode='continuous_gi' : scaled magnitude, ``levels * clip(y, 0, rel_max) / rel_max``. True
        negatives (y~0) map to ~0, hits to ~levels; smooth. Assumes a %GI-like scale where
        0 == inactive (rel_max = the %GI ceiling, default 100). A mixed-scale endpoint like
        pMIC (inactive floor != 0) would need its own per-assay scaling (cross-scale TODO).
    mode='rank_gi'       : within-list percentile rank of y mapped to [0, levels] -- removes
        ties entirely, but also assigns graded relevance to the (noisy) bottom of the list.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if mode == "rank_gi":
        if n <= 1:
            return np.full(n, float(levels))
        ranks = np.argsort(np.argsort(y))          # 0..n-1 ascending by activity
        return levels * ranks / (n - 1)
    if mode == "continuous_gi":
        return levels * np.clip(y, 0.0, rel_max) / rel_max
    if mode == "continuous_local":
        # per-ASSAY min-max magnitude: scale-adaptive across incomparable endpoints
        # (%GI ~[0,100] vs pMIC ~[4,10]). rel_lo/rel_hi = the assay's FIXED activity range
        # (passed by fit_listwise, computed once over the assay's full training set), so a
        # compound's relevance is intrinsic -- not dependent on which compounds were sampled
        # into its list. (Falls back to per-list min/max if not supplied.) Magnitude-preserving
        # -- the ranking analogue of pointwise_norm's per-assay z-score. Lists are single-assay.
        lo = float(np.min(y)) if rel_lo is None else float(rel_lo)
        hi = float(np.max(y)) if rel_hi is None else float(rel_hi)
        if hi - lo < 1e-9:
            return np.full(n, float(levels))
        return levels * np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    raise ValueError(f"unknown whole-list relevance mode {mode!r}")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_chemeleon_mp(ckpt_path: str = CHEMELEON_CKPT):
    """Build a ``BondMessagePassing`` initialized from the CheMeleon foundation.

    Same recipe as ``extract_embeddings.load_chemeleon`` (Zenodo 15460715 weights at
    ``~/.chemprop/chemeleon_mp.pt``) but returns just the message-passing block so we
    can wrap it in an MPNN subclass with a ranking head. Encoder is left trainable.
    """
    from chemprop import nn as cpnn

    ck = torch.load(ckpt_path, weights_only=True)
    mp = cpnn.BondMessagePassing(**ck["hyper_parameters"])
    mp.load_state_dict(ck["state_dict"])
    return mp, int(mp.output_dim)


class RankingMPNN:
    """Factory namespace -- see ``RankingMPNN.build`` / ``RankingMPNN.from_chemeleon``.

    (Implemented as a builder rather than a plain subclass constructor because the
    concrete subclass must be defined lazily: importing ``chemprop.models.MPNN`` at
    module import time would force a torch/chemprop import even for callers that only
    want ``graded_relevance``. The returned instance IS an ``chemprop.models.MPNN``.)
    """

    @staticmethod
    def build(mp, agg=None, head_hidden: int = 256, head_dropout: float = 0.1):
        from chemprop import models
        from chemprop import nn as cpnn

        agg = agg or cpnn.MeanAggregation()

        class _RankingMPNN(models.MPNN):
            """MPNN whose scalar output is a *ranking score* (not a per-task prediction)."""

            def __init__(self, mp, agg, head_hidden, head_dropout):
                # MPNN.__init__ requires a predictor; we never call it (fingerprint()
                # stops before the FFN). Kept only to satisfy the constructor.
                dummy = cpnn.RegressionFFN(input_dim=mp.output_dim)
                super().__init__(mp, agg, dummy, batch_norm=False)
                d = int(mp.output_dim)
                if head_hidden and head_hidden > 0:
                    self.score_head = nn.Sequential(
                        nn.Linear(d, head_hidden), nn.ReLU(),
                        nn.Dropout(head_dropout), nn.Linear(head_hidden, 1),
                    )
                else:
                    self.score_head = nn.Linear(d, 1)

            def score(self, bmg, V_d=None, X_d=None):
                """Per-molecule scalar ranking score from the D-MPNN fingerprint."""
                return self.score_head(self.fingerprint(bmg, V_d, X_d)).squeeze(-1)

        return _RankingMPNN(mp, agg, head_hidden, head_dropout)

    @staticmethod
    def from_chemeleon(ckpt_path: str = CHEMELEON_CKPT, head_hidden: int = 256,
                       head_dropout: float = 0.1):
        """CheMeleon-initialized (unfrozen) encoder + a fresh ranking head."""
        mp, _ = build_chemeleon_mp(ckpt_path)
        return RankingMPNN.build(mp, head_hidden=head_hidden, head_dropout=head_dropout)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class RankingChemPropTrainer:
    """Train a ``RankingMPNN`` with a pairwise / listwise / pointwise objective.

    The graph analogue of ``model.RankingTrainer``: the scorer is the D-MPNN, batches
    are collated from a cached ``MoleculeDataset``, and the loss backpropagates through
    message passing. One trainer instance == one scorer for one organism group.
    """

    def __init__(self, model, encoder_lr: float = 1e-5, head_lr: float = 1e-3,
                 weight_decay: float = 1e-5, warmup_frac: float = 0.1,
                 device: str | None = None):
        """Fine-tune a pretrained (CheMeleon) encoder with a ranking objective.

        Uses **differential learning rates** -- a low ``encoder_lr`` on the pretrained
        message-passing block and a 10-100x higher ``head_lr`` on the freshly-init
        ``score_head`` -- plus optional linear **warmup-then-decay** (``warmup_frac`` of
        total steps; 0 = constant LR). This is the standard recipe for fine-tuning a
        pretrained encoder (Howard & Ruder 2018 ULMFiT; Sun et al. 2019 show LR is a
        cliff for pretrained-encoder fine-tuning) and matches the sophistication of the
        ChemProp NoamLR schedule the pointwise baseline gets for free. To reproduce the
        old flat-LR behavior, pass ``encoder_lr == head_lr`` and ``warmup_frac=0``.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        enc_params, head_params = [], []
        for n, p in self.model.named_parameters():
            (head_params if n.startswith("score_head") else enc_params).append(p)
        self.opt = torch.optim.Adam(
            [{"params": enc_params, "lr": encoder_lr},
             {"params": head_params, "lr": head_lr}],
            weight_decay=weight_decay)
        self.warmup_frac = warmup_frac
        self.loss_history_: list[float] = []
        # Optional per-epoch callback: fn(tag, epoch, train_loss, val_metric_or_None).
        # Without it the curves are only readable AFTER a fold finishes, because
        # loss_history_ is reported in one shot by the caller. That is fine for the
        # ~4-min ranking folds but leaves the pointwise folds (~85 min each on S1)
        # showing nothing at all while they run -- no way to see progress or catch a
        # diverging loss. gnn_mt already streams live via a Lightning callback; this is
        # the equivalent hook for the arms that use this hand-rolled loop.
        self.on_epoch = None

    def _build_scheduler(self, total_steps: int, floor: float = 0.1):
        """Linear warmup (0->peak) over ``warmup_frac`` of steps, then linear decay to
        ``floor``x peak. Returns None (constant LR) when ``warmup_frac <= 0``. Scales
        both param groups' LRs proportionally, preserving the encoder/head ratio."""
        if self.warmup_frac <= 0 or total_steps <= 1:
            return None
        warmup = max(1, int(self.warmup_frac * total_steps))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return floor + (1.0 - floor) * (1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    # -- featurization / forward ------------------------------------------- #
    def _dataset(self, smiles, labels=None):
        """Build a cached ``MoleculeDataset`` (graphs precomputed once)."""
        from chemprop.data import MoleculeDatapoint, MoleculeDataset
        from rdkit import Chem

        dps = []
        for i, s in enumerate(smiles):
            y = np.array([float(labels[i]) if labels is not None else 0.0], dtype=float)
            dps.append(MoleculeDatapoint(mol=Chem.MolFromSmiles(s), y=y))
        dset = MoleculeDataset(dps)
        dset.cache = True  # precompute MolGraphs; we access indices repeatedly across epochs
        return dset

    def _forward_idx(self, dset, idx) -> torch.Tensor:
        """Collate ``dset[idx]``, move to device, return grad-enabled per-molecule scores."""
        from chemprop.data import collate_batch

        batch = collate_batch([dset[int(i)] for i in idx])
        batch.bmg.to(self.device)  # NB: BatchMolGraph.to() mutates in place, returns None
        V_d = batch.V_d.to(self.device) if batch.V_d is not None else None
        X_d = batch.X_d.to(self.device) if batch.X_d is not None else None
        return self.model.score(batch.bmg, V_d, X_d)

    @torch.no_grad()
    # -- validation: best-epoch selection ----------------------------------- #
    # Without this, whatever epoch the loop happens to end on is the model that gets
    # scored -- chemprop defines "best checkpoint" as best val metric, so with no val set
    # the concept does not exist. Selection criterion is per objective: the pointwise arm
    # uses val MSE, the ranking arms use val NDCG. Picking a pointwise model by a ranking
    # metric would build ranking-awareness into the baseline's own training procedure,
    # which is the bias the benchmark exists to measure.
    def _val_setup(self, val, criterion: str, patience: int = 5,
                   min_delta_rel: float = MIN_DELTA_REL):
        """``val`` = (smiles, targets, groups) or None. ``targets`` must already be in the
        space the criterion needs: z-scored for 'mse' (the model predicts z-scores), raw
        activity for 'ndcg' (used as graded relevance).

        ``min_delta_rel`` is RELATIVE, not absolute, because the two criteria live on
        different scales: val NDCG is bounded in [0,1] and plateaus near 0.77, while val
        MSE on z-scored targets starts near 1.0 and decays to the noise floor. Any
        absolute threshold meaningful for one is trivial or crippling for the other, so
        the test is "improved by more than min_delta_rel x |best|" in whichever direction
        the criterion improves.
        """
        self.val_history_ = []
        self._val = None
        if val is None:
            return
        smi, y, grp = val
        if criterion not in ("mse", "ndcg"):
            raise ValueError(f"unknown val criterion {criterion!r}")
        self._val = {"smiles": list(smi), "y": np.asarray(y, dtype=float),
                     "grp": np.asarray(grp), "criterion": criterion, "patience": patience,
                     "mode": "min" if criterion == "mse" else "max",
                     "min_delta_rel": float(min_delta_rel),
                     "best": None, "best_state": None, "best_epoch": None, "bad": 0}

    def _val_check(self, ep: int, verbose: bool = False, tag: str = "") -> bool:
        """Score val, track the best weights, return True when patience is exhausted."""
        # getattr, not self._val: _fit_scalar_target is also reached via
        # fit_pointwise_grouped, which never calls _val_setup
        v = getattr(self, "_val", None)
        if v is None:
            return False
        scores = self.score(v["smiles"])
        if v["criterion"] == "mse":
            metric = float(np.mean((scores - v["y"]) ** 2))
        else:
            from metrics import ndcg_at_k
            per = [ndcg_at_k(v["y"][m], scores[m])
                   for m in (v["grp"] == a for a in np.unique(v["grp"]))
                   if m.sum() >= 2 and not np.all(v["y"][m] == v["y"][m][0])]
            metric = float(np.mean(per)) if per else float("nan")
        self.val_history_.append(metric)

        # Two separate tests, deliberately. `improved` (any gain at all) keeps the best
        # weights, so we never discard a genuinely better epoch. `meaningful` (gain
        # exceeds the relative floor) is what resets patience -- otherwise a noise-level
        # gain buys another full patience window. Measured: an S1 fold improved val NDCG
        # by 0.0002 at epoch 19 and ran 5 more epochs for it, while a sibling fold on the
        # same config stopped at 9; the spread was noise, not convergence.
        if metric != metric:                                   # NaN: never an improvement
            improved = meaningful = False
        elif v["best"] is None:
            improved = meaningful = True
        else:
            gain = (v["best"] - metric) if v["mode"] == "min" else (metric - v["best"])
            improved = gain > 0
            meaningful = gain > v["min_delta_rel"] * abs(v["best"])

        if improved:
            v["best"], v["best_epoch"] = metric, ep
            v["best_state"] = {k: t.detach().cpu().clone()
                               for k, t in self.model.state_dict().items()}
        if meaningful:
            v["bad"] = 0
        else:
            v["bad"] += 1
        if verbose and (ep % 5 == 0 or v["bad"] >= v["patience"]):
            print(f"[dmpnn-{tag}] epoch {ep:3d}  val_{v['criterion']} {metric:.4f}"
                  f"{'  <- best' if v['bad'] == 0 else ''}", flush=True)
        return v["bad"] >= v["patience"]

    def _report_epoch(self, tag: str, ep: int) -> None:
        """Emit this epoch's train loss (and val metric, when there is one) to on_epoch.

        Called after _val_check so val_history_ is already appended. Never raises --
        telemetry must not kill a multi-hour fit.
        """
        if self.on_epoch is None:
            return
        v = getattr(self, "_val", None)
        vh = getattr(self, "val_history_", [])
        try:
            self.on_epoch(tag, ep, self.loss_history_[-1],
                          vh[-1] if (v is not None and vh) else None)
        except Exception:  # noqa: BLE001
            pass

    def _val_finish(self, tag: str = "", verbose: bool = False):
        """Restore the best epoch's weights (no-op when no val set was supplied)."""
        v = getattr(self, "_val", None)
        if v is None or v["best_state"] is None:
            return
        self.model.load_state_dict(v["best_state"])
        self.best_val_ = v["best"]
        self.best_epoch_ = v["best_epoch"]
        if verbose:
            print(f"[dmpnn-{tag}] restored best epoch {v['best_epoch']} "
                  f"(val_{v['criterion']} {v['best']:.4f})", flush=True)

    def _score_indices(self, dset, idx, batch_size: int = 512) -> np.ndarray:
        """Eval-mode scores for dataset indices, reusing the cached graphs in ``dset``.
        Used for hard-negative mining (rank inactives by current model score)."""
        was_training = self.model.training
        self.model.eval()
        idx = np.asarray(idx)
        out = [self._forward_idx(dset, idx[s:s + batch_size]).detach().cpu().numpy()
               for s in range(0, len(idx), batch_size)]
        if was_training:
            self.model.train()
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    @staticmethod
    def _select_batch_negatives(neg, n_neg_batch, hard_pool, sali_pool,
                                hard_frac, sali_frac, hard_pool_mult, sali_pool_mult, rng):
        """Split a batch's ``n_neg_batch`` negatives across three buckets and return the
        chosen GLOBAL inactive indices (deduped):

        * model-hard : top of ``hard_pool`` (current false positives; adaptive)
        * SALI-hard  : top of ``sali_pool`` (structurally near-active cliffs; static)
        * random     : the remainder, uniform over ``neg``

        Each hard bucket samples from the top ``pool_mult x n`` of its ranked pool (not a
        fixed top-K) so the selection stays *semi*-hard (diversity, no collapse). Pools are
        ranked global-index arrays or None; overlap between buckets is deduped, which can
        leave fewer than ``n_neg_batch`` negatives (fine -- higher quality)."""
        def _top_pool(pool, n, mult):
            k = min(len(pool), max(1, mult) * n)
            return rng.choice(pool[:k], size=min(n, k), replace=False)

        has_hard = hard_pool is not None and len(hard_pool)
        has_sali = sali_pool is not None and len(sali_pool)
        n_hard = int(round(hard_frac * n_neg_batch)) if has_hard else 0
        n_sali = int(round(sali_frac * n_neg_batch)) if has_sali else 0
        n_hard = min(n_hard, n_neg_batch)
        n_sali = min(n_sali, n_neg_batch - n_hard)
        chosen = []
        if n_hard > 0:
            chosen.append(_top_pool(hard_pool, n_hard, hard_pool_mult))
        if n_sali > 0:
            chosen.append(_top_pool(sali_pool, n_sali, sali_pool_mult))
        # Dedup the hard buckets first (model-hard and SALI-hard OVERLAP -- a near-active
        # inactive is often also a current false positive), then draw the random remainder
        # from inactives NOT already picked. This keeps the realized random : hard : SALI
        # proportions close to the request and hits n_neg_batch exactly, instead of letting
        # a final np.unique over (hard + sali + random-from-all) silently shrink/skew them.
        picked = np.unique(np.concatenate(chosen)) if chosen else neg[:0]
        n_rand = max(n_neg_batch - len(picked), 0)
        if n_rand > 0:
            rest = np.setdiff1d(neg, picked, assume_unique=True)
            if len(rest):
                picked = np.concatenate(
                    [picked, rng.choice(rest, size=min(n_rand, len(rest)), replace=False)])
        return picked if len(picked) else rng.choice(neg, size=min(n_neg_batch, len(neg)),
                                                      replace=False)

    # -- pairwise (RankNet) ------------------------------------------------- #
    def fit_pairwise(self, smiles, labels, groups, thr_map, epochs: int = 30,
                     n_act_per_batch: int = 64, neg_ratio: int = 3,
                     n_neg_per_active: int = 20, n_pos_per_active: int = 10,
                     pos_weight: float = 1.0, min_gap: float = 0.0,
                     hard_neg_frac: float = 0.0, mine_every: int = 5,
                     hard_pool_mult: int = 3, sali_neg_frac: float = 0.0,
                     sali_sim_lo: float = 0.3, sali_sim_hi: float = 0.9,
                     sali_pool_mult: int = 3, sali_radius: int = 2,
                     sali_n_bits: int = 2048, shuffle_schedule: bool = True,
                     iters_per_assay: int | None = None,
                     seed: int = 0, verbose: bool = False, val=None, patience: int = 5):
        """RankNet over within-assay, active-anchored pairs formed inside each batch.

        Each optimizer step consumes one assay-homogeneous batch = up to
        ``n_act_per_batch`` actives + ``neg_ratio``x that many sampled inactives.
        ``pairs.sample_pairs_imbalanced`` then forms active-over-inactive (+ optional
        active-over-active, ranked by %GI) pairs among the batch's molecules, and
        ``ranknet_loss`` is applied to the corresponding score differences. Bounding the
        batch keeps the D-MPNN forward/backward on the GPU tractable.

        **Semi-hard negative mining** (``hard_neg_frac > 0``): every ``mine_every``
        epochs the current model re-scores each assay's inactives; a fraction
        ``hard_neg_frac`` of each batch's negatives is then drawn from the top-scoring
        inactives (the model's current false positives) and the rest sampled randomly.
        Sampling from the top ``hard_pool_mult x n_hard`` (not a fixed top-K) keeps it
        *semi*-hard -- avoiding the diversity collapse of the frozen gate's naive
        full-replacement mining (Shen 2024 ACANet; Chen 2019). This is uniquely apt
        end-to-end, where the embedding moves under the gradient. ``0.0`` = pure random.

        **SALI-hard negatives** (``sali_neg_frac > 0``): a third, STATIC bucket of
        structurally near-active inactives -- activity cliffs (Guha & Van Drie 2008). Ranked
        once by max ECFP4-Tanimoto to any active in the assay (``sali.precompute_sali_ranks``)
        and restricted to a semi-hard band ``[sali_sim_lo, sali_sim_hi]`` (drop near-duplicate
        cliffs at sim->1 that are likely unlearnable/noise). Batch negatives then split
        random : model-hard : SALI-hard as ``(1 - hard - sali) : hard : sali``.

        The per-epoch step schedule is a flat list of assay tokens (one per optimizer step,
        ``ceil(n_act / n_act_per_batch)`` per assay) that is SHUFFLED each epoch when
        ``shuffle_schedule`` -- so assay updates interleave instead of running in fixed
        per-assay blocks (removes the within-epoch recency bias of the block order).
        """
        from losses import ranknet_loss
        from pairs import sample_pairs_imbalanced

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels, dtype=float)
        groups = np.asarray(groups)
        dset = self._dataset(smiles, labels)

        gdata = {}
        for g in np.unique(groups):
            gi = np.where(groups == g)[0]
            thr = thr_map[g]
            act, neg = gi[labels[gi] >= thr], gi[labels[gi] < thr]
            if len(act) and len(neg):
                gdata[g] = (act, neg, thr)
        if not gdata:
            return self

        # flat per-epoch schedule: one token per optimizer step (assay repeated by its
        # step count); shuffled each epoch so assay updates interleave (no block ordering).
        step_groups: list = []
        for g, (act, _neg, _thr) in gdata.items():
            n_iter = iters_per_assay or max(1, int(np.ceil(len(act) / n_act_per_batch)))
            step_groups.extend([g] * n_iter)
        steps_per_epoch = len(step_groups)
        sched = self._build_scheduler(epochs * steps_per_epoch)

        # SALI-hard negatives: static per-assay ranking of near-active inactives (once).
        sali_band: dict = {}
        if sali_neg_frac > 0:
            from sali import precompute_sali_ranks
            sali_band = precompute_sali_ranks(
                smiles, labels, groups, thr_map, sim_lo=sali_sim_lo, sim_hi=sali_sim_hi,
                radius=sali_radius, n_bits=sali_n_bits, verbose=verbose)
        # record how many inactives actually landed in the SALI band per assay, so a null
        # SALI result can be told apart from "SALI was barely applied" (empty/thin bands).
        self.sali_band_counts_ = {str(g): int(len(v)) for g, v in sali_band.items()}

        self.model.train()
        self.loss_history_ = []
        # ranking arm -> selected on val NDCG, with raw activity as graded relevance
        self._val_setup(val, "ndcg", patience)
        hard_rank: dict = {}
        for ep in range(epochs):
            if hard_neg_frac > 0 and ep % mine_every == 0:
                # re-score each assay's inactives; hardest (highest-scoring) first
                for g, (_act, neg, _thr) in gdata.items():
                    sc = self._score_indices(dset, neg)
                    hard_rank[g] = neg[np.argsort(-sc)]
            order = (rng.permutation(steps_per_epoch) if shuffle_schedule
                     else np.arange(steps_per_epoch))
            total, nsteps = 0.0, 0
            for si in order:
                g = step_groups[int(si)]
                act, neg, thr = gdata[g]
                a_sel = (rng.choice(act, size=n_act_per_batch, replace=False)
                         if len(act) > n_act_per_batch else act)
                n_neg_batch = min(len(neg), max(1, neg_ratio) * len(a_sel))
                n_sel = self._select_batch_negatives(
                    neg, n_neg_batch, hard_rank.get(g), sali_band.get(g),
                    hard_neg_frac, sali_neg_frac, hard_pool_mult, sali_pool_mult, rng)
                idx = np.concatenate([a_sel, n_sel])
                # pair indices are LOCAL to `idx` (single group), so they index `scores`
                y_local = labels[idx]
                pairs = sample_pairs_imbalanced(
                    y_local, np.zeros(len(idx)), active_threshold=thr,
                    n_neg_per_active=n_neg_per_active, n_pos_per_active=n_pos_per_active,
                    pos_weight=pos_weight, min_gap=min_gap, rng=rng)
                if not pairs:
                    continue
                scores = self._forward_idx(dset, idx)
                ii = torch.tensor([p.i for p in pairs], device=self.device)
                jj = torch.tensor([p.j for p in pairs], device=self.device)
                w = torch.tensor([p.weight for p in pairs], dtype=torch.float32,
                                 device=self.device)
                # pairs are constructed so i outranks j -> constant target labels
                li = torch.ones(len(pairs), device=self.device)
                lj = torch.zeros(len(pairs), device=self.device)
                self.opt.zero_grad()
                loss = ranknet_loss(scores[ii], scores[jj], li, lj, weight=w)
                loss.backward()
                self.opt.step()
                if sched is not None:
                    sched.step()
                total += loss.item()
                nsteps += 1
            self.loss_history_.append(total / max(nsteps, 1))
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[dmpnn-pairwise] epoch {ep:3d}  loss {self.loss_history_[-1]:.4f} "
                      f"({nsteps} steps)", flush=True)
            stop = self._val_check(ep, verbose, "pairwise")
            self._report_epoch("pairwise", ep)
            if stop:
                break
        self._val_finish("pairwise", verbose)
        return self

    # -- listwise (LambdaLoss-NDCG) ---------------------------------------- #
    def fit_listwise(self, smiles, labels, groups, thr_map, epochs: int = 30,
                     list_act: int = 128, list_neg: int = 384, lists_per_group: int = 20,
                     relevance_mode: str = "bins", rel_levels: int = 4,
                     rel_max: float = 100.0, gain: str = "exp",
                     hard_neg_frac: float = 0.0, mine_every: int = 5, hard_pool_mult: int = 3,
                     sali_neg_frac: float = 0.0, sali_sim_lo: float = 0.3,
                     sali_sim_hi: float = 0.9, sali_pool_mult: int = 3,
                     sali_radius: int = 2, sali_n_bits: int = 2048,
                     seed: int = 0, verbose: bool = False, val=None, patience: int = 5):
        """Listwise LambdaLoss-NDCG: per-assay lists, DeltaNDCG-weighted pairs.

        Each list = up to ``list_act`` actives + up to ``list_neg`` sampled inactives,
        resampled ``lists_per_group`` times per assay per epoch (caps bound the D-MPNN
        forward + the O(n^2) list matrix). Sampling is ALWAYS active-stratified (rare hits
        would vanish from a uniform sample). ``reduction="sum"`` matches Wang 2018 / allRank.

        Relevance labeling (``relevance_mode``):
          * "bins"/"continuous" : BINARIZED -- inactives get relevance 0, actives graded
            1..``rel_levels`` by %GI (quartile-rank bins / min-max). The classic setup.
          * "continuous_gi"/"rank_gi" : NON-binarized -- EVERY molecule graded from its raw
            activity (``list_relevance``), so there is no relevance cliff at the hit
            threshold; the listwise loss then has the non-degenerate graded labels it is
            designed to exploit. ``rel_levels`` (L) sets the relevance ceiling and ``rel_max``
            the %GI scale for continuous_gi. ``gain`` = "exp" (2^rel-1) or "linear" (rel).

        Negative composition (the ``list_neg`` inactive slots): random by default. With
        ``hard_neg_frac`` / ``sali_neg_frac`` the slots split random : model-hard : SALI-hard
        via ``_select_batch_negatives`` (same buckets as ``fit_pairwise``). Especially apt for
        LambdaLoss: BOTH its ΔNDCG weight and the logistic term vanish for easy negatives
        already ranked low, so a random-inactive list wastes most of its slots -- hard
        negatives put the list's mass in the contested top region (the IR retrieve-then-rerank
        "candidate set" regime). Model-hard inactives are re-mined every ``mine_every`` epochs;
        keep a random fraction for calibration (all-hard drifts off-distribution / saturates).
        """
        from losses import lambdaloss_ndcg

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels, dtype=float)
        groups = np.asarray(groups)
        dset = self._dataset(smiles, labels)

        gdata = {}
        for g in np.unique(groups):
            gi = np.where(groups == g)[0]
            thr = thr_map[g]
            act, neg = gi[labels[gi] >= thr], gi[labels[gi] < thr]
            if len(act) and len(neg):
                gdata[g] = (act, neg)
        if not gdata:
            return self

        # per-assay activity range (fixed scale) for continuous_local relevance, computed
        # once over each assay's FULL training set -- so a compound's relevance is intrinsic,
        # not dependent on which compounds land in its sampled list.
        assay_range = {g: (float(labels[np.concatenate([a, n])].min()),
                           float(labels[np.concatenate([a, n])].max()))
                       for g, (a, n) in gdata.items()}

        sched = self._build_scheduler(epochs * len(gdata) * lists_per_group)

        # SALI-hard negatives: static per-assay ranking of near-active inactives (once).
        sali_band: dict = {}
        if sali_neg_frac > 0:
            from sali import precompute_sali_ranks
            sali_band = precompute_sali_ranks(
                smiles, labels, groups, thr_map, sim_lo=sali_sim_lo, sim_hi=sali_sim_hi,
                radius=sali_radius, n_bits=sali_n_bits, verbose=verbose)
        self.sali_band_counts_ = {str(g): int(len(v)) for g, v in sali_band.items()}

        self.model.train()
        self.loss_history_ = []
        # ranking arm -> selected on val NDCG, with raw activity as graded relevance
        self._val_setup(val, "ndcg", patience)
        hard_rank: dict = {}
        for ep in range(epochs):
            if hard_neg_frac > 0 and ep % mine_every == 0:
                # re-score each assay's inactives; hardest (highest-scoring) first
                for g, (_act, neg) in gdata.items():
                    sc = self._score_indices(dset, neg)
                    hard_rank[g] = neg[np.argsort(-sc)]
            total, nlists = 0.0, 0
            for g, (act, neg) in gdata.items():
                for _ in range(lists_per_group):
                    a_sel = (act[rng.choice(len(act), size=list_act, replace=False)]
                             if len(act) > list_act else act)
                    if hard_neg_frac > 0 or sali_neg_frac > 0:
                        negs = self._select_batch_negatives(
                            neg, min(list_neg, len(neg)), hard_rank.get(g), sali_band.get(g),
                            hard_neg_frac, sali_neg_frac, hard_pool_mult, sali_pool_mult, rng)
                    else:
                        negs = (rng.choice(neg, size=list_neg, replace=False)
                                if len(neg) > list_neg else neg)
                    idx = np.concatenate([a_sel, negs])
                    if relevance_mode in ("continuous_gi", "rank_gi", "continuous_local"):
                        # non-binarized: grade EVERY molecule from its raw activity (no 0
                        # floor at the hit threshold); sampling above stays active-stratified.
                        # continuous_local uses the assay's FIXED range (not the sampled list's).
                        lo, hi = (assay_range[g] if relevance_mode == "continuous_local"
                                  else (None, None))
                        rel_np = list_relevance(labels[idx], mode=relevance_mode,
                                                levels=rel_levels, rel_max=rel_max,
                                                rel_lo=lo, rel_hi=hi)
                    else:
                        rel_act = graded_relevance(labels[a_sel], levels=rel_levels,
                                                   mode=relevance_mode)
                        rel_np = np.concatenate([rel_act, np.zeros(len(negs))])
                    rel = torch.tensor(rel_np, dtype=torch.float32, device=self.device)
                    scores = self._forward_idx(dset, idx)
                    self.opt.zero_grad()
                    loss = lambdaloss_ndcg(scores, rel, gain=gain)
                    loss.backward()
                    self.opt.step()
                    if sched is not None:
                        sched.step()
                    total += loss.item()
                    nlists += 1
            self.loss_history_.append(total / max(nlists, 1))
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[dmpnn-listwise] epoch {ep:3d}  loss {self.loss_history_[-1]:.4f} "
                      f"({nlists} lists)", flush=True)
            stop = self._val_check(ep, verbose, "listwise")
            self._report_epoch("listwise", ep)
            if stop:
                break
        self._val_finish("listwise", verbose)
        return self

    # -- pointwise (capacity/representation-matched MSE baseline) ----------- #
    def _fit_scalar_target(self, smiles, yz, epochs, batch_size, seed, tag, verbose):
        """Shared pointwise-MSE loop over an aligned, NaN-free ``(smiles, yz)`` set."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        yz = np.asarray(yz, dtype=np.float32)
        dset = self._dataset(smiles, yz)
        n = len(smiles)
        sched = self._build_scheduler(epochs * int(np.ceil(n / batch_size)))
        self.model.train()
        self.loss_history_ = []
        for ep in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                scores = self._forward_idx(dset, idx)
                tgt = torch.tensor(yz[idx], dtype=torch.float32, device=self.device)
                self.opt.zero_grad()
                loss = F.mse_loss(scores, tgt)
                loss.backward()
                self.opt.step()
                if sched is not None:
                    sched.step()
                total += loss.item() * len(idx)
            self.loss_history_.append(total / n)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[dmpnn-{tag}] epoch {ep:3d}  loss {self.loss_history_[-1]:.4f}", flush=True)
            stop = self._val_check(ep, verbose, tag)
            self._report_epoch(tag, ep)
            if stop:
                break
        self._val_finish(tag, verbose)
        return self

    def fit_pointwise(self, smiles, labels, epochs: int = 30, batch_size: int = 256,
                      seed: int = 0, verbose: bool = False, val=None, patience: int = 5):
        """Pointwise MSE on the SAME encoder + head, target z-scored on GLOBAL training
        stats (a single activity axis). The honest capacity/representation-matched
        comparator on a single-endpoint group; on a MIXED-endpoint set (%GI + pMIC) it is
        the naive/ill-posed baseline -- the dominant scale swamps the other -- so see
        ``fit_pointwise_grouped`` for the within-assay-normalized steelman. NaN labels dropped."""
        labels = np.asarray(labels, dtype=float)
        keep = ~np.isnan(labels)
        smiles = [s for s, k in zip(smiles, keep, strict=True) if k]
        y = labels[keep].astype(np.float32)
        mu, sigma = float(y.mean()), float(y.std())
        sd = sigma if sigma > 1e-8 else 1.0
        yz = (y - mu) / sd
        # val targets go into the SAME z-space -- the model predicts z-scores, so a raw-y
        # MSE would compare incommensurable quantities. Criterion is MSE (not NDCG): this
        # is the pointwise arm, selected the way production selects it.
        if val is not None:
            v_smi, v_y, v_grp = val
            v_y = (np.asarray(v_y, dtype=float) - mu) / sd
            val = (v_smi, v_y, v_grp)
        self._val_setup(val, "mse", patience)
        return self._fit_scalar_target(smiles, yz, epochs, batch_size, seed, "pointwise", verbose)

    def fit_pointwise_grouped(self, smiles, labels, groups, epochs: int = 30,
                              batch_size: int = 256, seed: int = 0, verbose: bool = False,
                              val=None, patience: int = 5):
        """Pointwise MSE with **per-assay** (within-group) target z-scoring -- the steelman
        comparator for the cross-scale (%GI + pMIC) test. Each assay's target is
        standardized on its own training mean/std, reconciling incomparable endpoint
        scales the way multitask heads / manual normalization would (vs the naive single
        global z-score in ``fit_pointwise``). Isolates whether ranking's edge is more than
        implicit within-assay normalization. NaN labels dropped."""
        labels = np.asarray(labels, dtype=float)
        groups = np.asarray(groups)
        keep = ~np.isnan(labels)
        smiles = [s for s, k in zip(smiles, keep, strict=True) if k]
        y = labels[keep].astype(np.float32)
        g = groups[keep]
        yz = np.empty_like(y)
        stats = {}
        for gg in np.unique(g):
            m = g == gg
            mu, sd = float(y[m].mean()), float(y[m].std())
            sd = sd if sd > 1e-8 else 1.0
            stats[gg] = (mu, sd)
            yz[m] = (y[m] - mu) / sd
        # val targets into the SAME per-assay z-space (train stats only). A val assay with
        # no training counterpart has no transform, so those rows are dropped rather than
        # scored against an arbitrary scale.
        if val is not None:
            v_smi, v_y, v_grp = val
            v_y, v_grp = np.asarray(v_y, dtype=float), np.asarray(v_grp)
            ok = np.array([gg in stats for gg in v_grp])
            v_z = np.array([(v_y[i] - stats[v_grp[i]][0]) / stats[v_grp[i]][1]
                            for i in np.flatnonzero(ok)])
            val = ([s for s, k in zip(v_smi, ok, strict=True) if k], v_z, v_grp[ok])
        self._val_setup(val, "mse", patience)
        return self._fit_scalar_target(smiles, yz, epochs, batch_size, seed,
                                       "pointwise_norm", verbose)

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def score(self, smiles, batch_size: int = 512) -> np.ndarray:
        """Row-aligned ranking scores for ``smiles`` (one forward pass, no shuffle)."""
        self.model.eval()
        dset = self._dataset(smiles)
        out = []
        for s in range(0, len(smiles), batch_size):
            idx = range(s, min(s + batch_size, len(smiles)))
            out.append(self._forward_idx(dset, idx).detach().cpu().numpy())
        self.model.train()
        return np.concatenate(out) if out else np.empty(0, dtype=float)
