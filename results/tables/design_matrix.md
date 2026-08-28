**Table 1. The benchmark matrix — 210 configurations (190 core + 20 loss ablations).**

| Dimension | Breakdown |
|---|---|
| Scope | S1 · WT E. coli 70 · S2 · E. coli+pMIC 70 · S3 · multi-species 70 |
| Split | 5-fold CV (chemistry holdout) 105 · leave-one-source-out (assay transfer) 105 |
| Architecture | Random Forest 36 · XGBoost 56 · MLP 90 · D-MPNN (single-head) 22 · D-MPNN (multi-task) 6 |
| Objective | MSE 66 · per-assay z-MSE 58 · RankNet 24 · LambdaLoss-NDCG 24 · LambdaMART 18 · lambdaloss_k 6 · ranknet_full 12 · lambdamart_full 2 |
| Representation | RDKit descriptors (200-d) 68 · Morgan FP (2048-d) 54 · CheMeleon embed (2048-d) 60 · learned D-MPNN 28 |

Ablations (20): lambdaloss_k 6, ranknet_full 12, lambdamart_full 2. Asymmetries: per-assay z-MSE exists only on the cross-scale scopes (S2/S3); gnn/gnn\_mt are D-MPNN, single-representation; gnn\_mt runs MSE only. Counts are therefore marginal, not a full product.