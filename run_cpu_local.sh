#!/usr/bin/env bash
# Run the CPU arms LOCALLY, in parallel, still tracked in ClearML.
#
# Why local rather than the queue: there is one agent on `default` and it serves the GPU
# sweep serially, so 108 CPU configs would sit behind ~100 GPU-hours despite needing no
# GPU at all. Locally they run alongside it. `--log_to_clearml` without
# `--execute_remotely` gives a normal tracked task per variant -- same tasks, same
# scalars, same artifacts, just executed here.
#
# Two constraints set the concurrency, and the binding one is NOT memory:
#
#   cores   32 on this box. rf/xgb default to n_jobs=-1 and torch grabs every core, so
#           N concurrent processes would each claim all 32 and thrash. --n-jobs caps it.
#           Keep JOBS x NJOBS <= ~32. (OMP_NUM_THREADS alone does NOT cap sklearn's RF:
#           its parallelism is joblib threads over trees, not BLAS.)
#   memory  peak per config, measured as feature matrix + the float32 train/test copies
#           sklearn makes (morgan's uint8 saving vanishes inside the fit):
#             rdkit      ~4-5 GB      morgan  ~9-14 GB      chemeleon ~12-22 GB
#           124 GB total on the box, and the GPU job's container plus other users need
#           headroom -- so the 2048-dim tiers are the ones that can actually OOM.
#
# Tiers exist because cost is wildly uneven: all 36 rdkit configs total ~2.1 h, while the
# 72 morgan/chemeleon configs are ~30.8 h -- 90% of the CPU budget. Running rdkit first
# exercises every arch/loss/scope combination cheaply before committing to the expensive
# tier.
#
# Usage:
#   ./run_cpu_local.sh rdkit                 # 36 configs, 8-way, ~20 min
#   ./run_cpu_local.sh morgan                # 36 configs, 3-way
#   ./run_cpu_local.sh chemeleon --dry-run
#   setsid nohup ./run_cpu_local.sh rdkit > cpu_rdkit.log 2>&1 &   # survives logout
set -uo pipefail
cd "$(dirname "$0")"

ENV=learn2rank
FEATS="${1:-}"
shift || true
DRY=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY=1; done

case "$FEATS" in
  rdkit)               JOBS=8; NJOBS=4  ;;   # ~5 GB each -> 40 GB, 32 threads
  morgan|chemeleon)    JOBS=3; NJOBS=10 ;;   # up to ~22 GB each -> ~65 GB, 30 threads
  *) echo "usage: $0 {rdkit|morgan|chemeleon} [--dry-run]" >&2; exit 2 ;;
esac

# The registry is the single source of truth for which configs exist and which are done.
mapfile -t RUN_IDS < <(
  micromamba run -n "$ENV" python run_benchmark.py --dry-run \
    --arch rf --arch mlp --arch xgb 2>/dev/null \
    | awk -v f="$FEATS" '$1 ~ /^[a-z]/ && $3 == f {print $1}'
)
if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
  echo "no $FEATS configs left to run (already done, or none match)"; exit 0
fi

echo "$FEATS: ${#RUN_IDS[@]} configs | ${JOBS}-way x --n-jobs $NJOBS"
if [[ $DRY -eq 1 ]]; then printf '  %s\n' "${RUN_IDS[@]}"; echo "(dry run)"; exit 0; fi

mkdir -p logs
run_one() {
  local rid="$1"
  # one tracked ClearML task per variant, executed here rather than on the queue
  if micromamba run -n "$ENV" python run_benchmark.py \
       --run-id "$rid" --log_to_clearml --n-jobs "$2" > "logs/${rid}.log" 2>&1; then
    echo "  ok      $rid"
  else
    echo "  FAILED  $rid  (see logs/${rid}.log)" >&2
  fi
}
export -f run_one
export ENV

printf '%s\n' "${RUN_IDS[@]}" \
  | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$NJOBS"

echo "$FEATS tier complete -> results in $(ls -d results/public_benchmark 2>/dev/null || echo results/)"
