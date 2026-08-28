#!/usr/bin/env bash
# Run the public gram-negative benchmark locally.
#
# run_benchmark.py loops the config registry itself, so this is a thin wrapper that
# passes filters straight through. Results are written per-config to results/benchmark/.
#
# Examples:
#   ./run_public_benchmark.sh --dry-run                    # list what would run
#   ./run_public_benchmark.sh --arch xgb --scope S1_wt_ecoli
#   ./run_public_benchmark.sh --features rdkit --split cv
#   ./run_public_benchmark.sh                              # the full matrix
#
# The CPU arms (rf / xgb / mlp) run on CPU. The D-MPNN arms (gnn / gnn_mt) need a GPU;
# pass --gpu N. Feature tables are regenerated first if absent (see compute_rdkit_features.py
# / extract_embeddings.py).
set -uo pipefail
cd "$(dirname "$0")"
python run_benchmark.py "$@"
