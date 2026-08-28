#!/usr/bin/env bash
# Regenerate the public benchmark's splits locally from the public dataframe.
#
# The splits ship with the repo under data/, so this is only needed to rebuild them.
# BalanceSplit is chemically balanced; the CV build is the slow part (~hours at
# ~828k compounds), the LODO folds are deterministic and take seconds.
#
#   ./run_public_cv_split.sh            # rebuild both CV (1x5) and LODO
#   ./run_public_cv_split.sh --cv       # CV only
#   ./run_public_cv_split.sh --lodo     # LODO only
set -uo pipefail
cd "$(dirname "$0")"
python make_public_splits.py "${@:---cv --lodo}"
