"""
A minimal, dependency-light ranking model to prototype LTR on Phare Bio data.

Design choices:
    * The scorer is deliberately a small MLP over *precomputed features* (the
      200-dim normalized RDKit descriptors, or Morgan FPs). This lets you validate
      the LTR idea quickly WITHOUT touching ChemProp.
    * The same losses (losses.py) work unchanged if you later swap the MLP for the
      ChemProp D-MPNN encoder via its Python API (chemprop.nn.MPNN). See
      `ChemPropScorerAdapter` docstring for the integration point.
    * A single scorer f(x) -> R is trained with pairwise/listwise/contrastive
      objectives, giving one shared latent activity scale across all assays.

This is a scaffold: it is correct and runnable, but you will want to add
early stopping, checkpointing, and experiment logging for production use.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from losses import (
    lambdaloss_ndcg,
    listnet_loss,
    ranknet_loss,
    supcon_loss,
)
from pairs import Pair, pairs_to_arrays


class MLPScorer(nn.Module):
    """Shared scorer f: features -> scalar score. Also exposes an embedding."""

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)


class RankingTrainer:
    """
    Train a scorer with a configurable LTR objective.

    objective:
        "pairwise"    -> RankNet over provided pairs (recommended default)
        "listwise"    -> LambdaLoss-NDCG per group (best for early recognition)
        "listnet"     -> ListNet top-1 per group
        "contrastive" -> SupCon pretraining on coarse activity buckets, then
                         a short pairwise fine-tune (call fit twice).
    """

    def __init__(
        self,
        scorer: nn.Module,
        objective: str = "pairwise",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = scorer.to(self.device)
        self.objective = objective
        self.opt = torch.optim.Adam(scorer.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------ #
    def fit_pairwise(
        self,
        X: np.ndarray,
        pairs: list[Pair],
        epochs: int = 30,
        batch_size: int = 1024,
        verbose: bool = True,
    ):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        i, j, w = pairs_to_arrays(pairs)
        # labels only needed for the sign; recover from... we stored order i>j, so
        # synthesize labels_i > labels_j by construction:
        li = torch.ones(len(i), device=self.device)
        lj = torch.zeros(len(j), device=self.device)
        i_t = torch.tensor(i, device=self.device)
        j_t = torch.tensor(j, device=self.device)
        w_t = torch.tensor(w, device=self.device)

        n = len(i)
        self.loss_history_ = []
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for s in range(0, n, batch_size):
                b = perm[s : s + batch_size]
                self.opt.zero_grad()
                si = self.scorer(X_t[i_t[b]])
                sj = self.scorer(X_t[j_t[b]])
                loss = ranknet_loss(si, sj, li[b], lj[b], weight=w_t[b])
                loss.backward()
                self.opt.step()
                total += loss.item() * len(b)
            self.loss_history_.append(total / n)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[pairwise] epoch {ep:3d}  loss {total / n:.4f}")
        return self

    def fit_pointwise(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        epochs: int = 30,
        batch_size: int = 1024,
        verbose: bool = True,
    ):
        """
        Capacity-matched pointwise regression baseline: train the SAME MLPScorer
        backbone with plain MSE on the continuous activity value. This isolates the
        *objective* (pointwise vs pairwise) from model capacity -- the honest
        comparator for the RankNet go/no-go gate (a linear RidgeCV would confound
        the two). Rows with NaN labels are dropped.

        The target is z-scored using training statistics for stable optimization;
        since rank-aware metrics only depend on the ORDER of the predicted scores,
        this standardization does not affect evaluation and no inverse transform is
        needed at scoring time.
        """
        m = ~np.isnan(labels)
        X_t = torch.tensor(X[m], dtype=torch.float32, device=self.device)
        y = labels[m].astype(np.float32)
        mu, sigma = float(y.mean()), float(y.std())
        sigma = sigma if sigma > 1e-8 else 1.0
        y_t = torch.tensor((y - mu) / sigma, dtype=torch.float32, device=self.device)

        n = len(y_t)
        self.loss_history_ = []
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for s in range(0, n, batch_size):
                b = perm[s : s + batch_size]
                self.opt.zero_grad()
                pred = self.scorer(X_t[b])
                loss = torch.nn.functional.mse_loss(pred, y_t[b])
                loss.backward()
                self.opt.step()
                total += loss.item() * len(b)
            self.loss_history_.append(total / n)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[pointwise] epoch {ep:3d}  loss {total / n:.4f}")
        return self

    def fit_listwise(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        epochs: int = 40,
        use_lambdaloss: bool = True,
        verbose: bool = True,
    ):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(np.nan_to_num(labels), dtype=torch.float32, device=self.device)
        uniq = np.unique(groups)
        for ep in range(epochs):
            total = 0.0
            for g in uniq:
                idx = np.where((groups == g) & ~np.isnan(labels))[0]
                if len(idx) < 2:
                    continue
                idx_t = torch.tensor(idx, device=self.device)
                self.opt.zero_grad()
                s = self.scorer(X_t[idx_t])
                yy = y_t[idx_t]
                loss = lambdaloss_ndcg(s, yy) if use_lambdaloss else listnet_loss(s, yy)
                loss.backward()
                self.opt.step()
                total += loss.item()
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[listwise] epoch {ep:3d}  loss {total / len(uniq):.4f}")
        return self

    def pretrain_contrastive(
        self,
        X: np.ndarray,
        activity_bucket: np.ndarray,
        epochs: int = 20,
        batch_size: int = 512,
        verbose: bool = True,
    ):
        """SupCon warm-up on coarse activity buckets (e.g. deciles or active/inactive)."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(activity_bucket, dtype=torch.long, device=self.device)
        n = len(X)
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for s in range(0, n, batch_size):
                b = perm[s : s + batch_size]
                self.opt.zero_grad()
                z = self.scorer.embed(X_t[b])
                loss = supcon_loss(z, y_t[b])
                loss.backward()
                self.opt.step()
                total += loss.item() * len(b)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"[supcon]   epoch {ep:3d}  loss {total / n:.4f}")
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        self.scorer.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        out = self.scorer(X_t).cpu().numpy()
        self.scorer.train()
        return out


# --------------------------------------------------------------------------- #
class ChemPropScorerAdapter:
    """
    Integration point (not implemented here -- documented for the next step).

    To use the D-MPNN encoder instead of RDKit-descriptor MLP:

        from chemprop.nn import BondMessagePassing, MeanAggregation
        from chemprop.models import MPNN

        class DMPNNScorer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mp   = BondMessagePassing()
                self.agg  = MeanAggregation()
                self.head = nn.Linear(self.mp.output_dim, 1)
            def embed(self, batched_mol_graph):
                return self.agg(self.mp(batched_mol_graph), ...)
            def forward(self, batched_mol_graph):
                return self.head(self.embed(batched_mol_graph)).squeeze(-1)

    Then feed ChemProp `BatchMolGraph` objects (built from SMILES via
    chemprop.data) in place of the feature tensors above. The losses in losses.py
    are agnostic to how the score was produced, so RankingTrainer needs only a
    thin change to collate graphs instead of indexing a dense matrix.

    NOTE: driving ChemProp through the CLI (subprocess) does not expose the loss to
    autograd; for a custom ranking loss use the ChemProp *Python API* so the loss is
    part of the autograd graph. This is the single biggest integration task.
    """
    pass
