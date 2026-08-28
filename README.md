# Learn-to-rank vs. pointwise regression for gram-negative antibiotic discovery

Code and data to reproduce a benchmark comparing **learn-to-rank (LTR)** objectives
against **pointwise regression** for early-recognition virtual screening on a public
gram-negative antibacterial dataset.

**Headline result.** On a frozen molecular representation, pointwise regression (MSE /
per-assay z-scored MSE) matches or beats every learn-to-rank objective (RankNet,
LambdaLoss, LambdaMART) across all scopes, both evaluation splits, and every
early-recognition metric (BEDROC, EF@1%/5%, NDCG). The gap is widest under assay
transfer and under stricter early-recognition weighting (BEDROC α=100).

## What is compared

- **Architectures:** Random Forest, XGBoost, MLP, D-MPNN (message-passing GNN),
  multi-task D-MPNN.
- **Representations:** RDKit 2D descriptors, Morgan fingerprints, CheMeleon
  foundation-model embeddings (frozen), and the D-MPNN's own learned representation.
- **Objectives:** pointwise `mse` and per-assay z-scored `pointwise_norm`; ranking
  `ranknet`, `lambdaloss`, `lambdamart`.
- **Scopes:** `S1` WT *E. coli* (single %GI scale), `S2` *E. coli* + pMIC (cross-scale),
  `S3` multi-species (cross-scale).
- **Splits:** 5-fold chemistry-holdout CV, and leave-one-source-out (LODO, assay
  transfer). LODO is **compound-disjoint** — a held-out source's compounds are excluded
  from training entirely.

Metrics are computed by out-of-fold pooling and averaged within-assay (BEDROC α=20 is
the primary metric); %GI and pMIC endpoints are never pooled numerically.

## Data

The combined public activity matrix and the fold assignments ship under `data/`
(see [`data/README.md`](data/README.md)). Sources are all publicly available: PubChem
AID 710 / AID 1053175, the CO-ADD open-source subset, Gadiya et al. (pMIC + validation),
and Zahed et al. 2021. Feature tables are **not** shipped (they are large) and are
regenerated locally:

```bash
python compute_rdkit_features.py    # -> data/rdkit_features.parquet   (200-dim RDKit)
python extract_embeddings.py        # -> data/chemeleon_embeddings.parquet (CheMeleon; needs chemprop)
```

## Install

```bash
pip install -r requirements.txt
# install a torch build for your platform/CUDA from https://pytorch.org
python demo.py                      # synthetic end-to-end smoke test (no data needed)
```

The `metrics.py`, `pairs.py`, `splits.py` modules and the full test suite run without
torch; only the model/loss/training paths import it.

## Reproduce the benchmark

```bash
# run the full matrix locally (CPU arms on CPU; D-MPNN arms need --gpu N)
./run_public_benchmark.sh                       # or: python run_benchmark.py
./run_public_benchmark.sh --arch xgb --scope S1_wt_ecoli   # a subset
# rebuild the splits from the dataframe (they also ship prebuilt)
./run_public_cv_split.sh
```

Per-config results are written to `results/benchmark/`. Then regenerate the figures and
tables (these read the shipped result JSONs):

```bash
python make_benchmark_figures.py all            # verdict, heatmap, multimetric
python make_alpha_heatmap.py --alpha 100        # BEDROC α=100 heatmap
python make_factor_range_table.py               # factor-range decomposition
python make_benchmark_tables.py all             # design-matrix + task-overview tables
python make_dataset_figure.py                   # dataset diversity (UMAP + distributions)
python run_5x5_cv.py                            # 5x5 repeated-CV comparison
```

## Modules

| File | Purpose |
|---|---|
| `metrics.py` | Rank-aware + early-recognition metrics: Spearman, Kendall, NDCG, EF@k, BEDROC, RIE (pure numpy/scipy). |
| `losses.py` | PyTorch LTR losses: RankNet, ListNet, LambdaLoss-NDCG, contrastive. |
| `pairs.py` | Within-assay pair construction, sparsity-aware active-anchored sampling, hard-negative mining. |
| `splits.py` | Leave-one-source-out, grouped-scaffold, temporal splits. |
| `model.py` / `xgb_scorer.py` | MLP and XGBoost scorers + trainers. |
| `ranking_chemprop.py` | D-MPNN ranking trainer (CheMeleon-initialized; chemprop Python API). |
| `run_benchmark.py` | Enumerates and runs the config registry; OOF pooling + scoring. |
| `experiment_registry.py` | The config matrix and stable per-config `run_id`s. |
| `bootstrap_ci.py` | Bootstrap confidence intervals (within-assay compound resampling). |
| `curate_public_data.py` | Scope/task definitions and eligibility for S1/S2/S3. |
| `make_*` | Figure and table generators. |

## Tests

```bash
pytest        # pure-numpy tests, no torch required
ruff check .  # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push and PR.

## License

MIT — see [LICENSE](LICENSE).
