# Data

The public gram-negative benchmark inputs. Assay values are keyed by
`canonicalized_smiles` (RDKit standardization: metal disconnection, salt/solvent
stripping to the parent fragment, charge neutralization, canonical tautomer,
stereochemistry preserved).

Large CSVs are gzip-compressed (pandas reads `.csv.gz` transparently) to stay under
repo file-size limits.

| file | description |
|---|---|
| `public_gram_negative_wide_df.csv.gz` | combined public activity matrix — one row per compound, one column per assay endpoint (NaN where untested) |
| `public_1x5_split.csv.gz` | 5-fold chemistry-holdout CV assignment (BalanceSplit): `canonicalized_smiles`, `Folds` |
| `public_5x5_split.csv.gz` | 5×5 repeated CV assignments: `canonicalized_smiles`, `Folds_0..Folds_4` |
| `public_lodo_splits_meta.json` | leave-one-source-out fold order per scope (held-out source per fold). LODO folds are defined by source membership, so this is all the benchmark needs — the full per-compound LODO JSON is not shipped. |
| `organism_groups.json` | mapping of assay endpoints to organism groups |

## Public data sources

Only publicly available sources are included: PubChem AID 710 and AID 1053175,
the CO-ADD open-source subset, the Gadiya et al. pMIC and validation sets, and
Zahed et al. 2021 (tolC).

## Regenerating derived inputs

- **Splits** (already shipped) can be rebuilt from the wide dataframe:
  `python make_public_splits.py --cv --lodo`
- **Feature tables** are not shipped (they are large); regenerate them:
  - RDKit descriptors: `python compute_rdkit_features.py` → `rdkit_features.parquet`
  - CheMeleon embeddings: `python extract_embeddings.py` → `chemeleon_embeddings.parquet`
