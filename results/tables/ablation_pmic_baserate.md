**Table. pMIC base-rate ablation (S2 · CV · mlp/rdkit).** pMIC actives subsampled 2,596 → 108 (matching coadd E. coli ATCC 25922), inactive pool fixed, 5 seeds. BEDROC α=20 on the pMIC head.

| Objective | Full base rate | Base-rate-matched (mean ± SD) | Δ |
|---|--:|--:|--:|
| MSE (global-z) | 0.171 | 0.111 ± 0.029 | -0.060 |
| z-MSE (per-assay) | 0.389 | 0.221 ± 0.052 | -0.168 |
| RankNet | 0.491 | 0.215 ± 0.024 | -0.277 |
| LambdaLoss | 0.233 | 0.138 ± 0.027 | -0.095 |

RankNet − z-MSE (steelman pointwise): **+0.102** (full) → **-0.006** (matched).
