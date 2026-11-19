#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
SEMANTIC="${1:?Usage: train_progressive_attribute_channel.sh SEMANTIC_FINAL_PT NEW_OUT [options]}"
OUT="${2:?Usage: train_progressive_attribute_channel.sh SEMANTIC_FINAL_PT NEW_OUT [options]}"
shift 2
INIT=()
if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU $CUDA_VISIBLE_DEVICES | frozen semantic checkpoint $SEMANTIC | output $OUT"
case " $* " in
  *" --three-tier-scratch "*)
    echo '8/24/48 cumulative complex symbols per Gaussian. Fresh channel weights and frozen private128 semantic model.'
    echo 'Default per layer is 8000 bootstrap + 250 render updates across both training scenes. Total 24750.'
    echo 'Bootstrap validation at 500, 2000, 4000 and 8000. Default neighbor steps 0.'
    ;;
  *)
    echo 'Eight cumulative budgets 8/16/24/32/40/48/56/64 complex symbols per Gaussian.'
    echo 'Default per layer is 500 bootstrap + 250 render updates, then 200 shared neighbor-only updates. Total 6200.'
    ;;
esac
echo 'Train the current channel layer only. Earlier layers stay frozen.'
echo '32 rotating training views; inherit original validation/test cameras and heldout scenes.'
echo 'Final delivery verifies LDPC4096/6144 + 16QAM side files; all failures remain charged.'
echo 'Download OUT/review.tar.gz. Resume: same arguments, RESUME=1 and same OUT.'
exec "${PYTHON_BIN:-python}" -u -m gaussian_jscc.progressive_attribute_experiment \
  --semantic-checkpoint "$SEMANTIC" --out "$OUT" \
  --digital-python "${DIGITAL_PYTHON:-$HOME/miniforge3/envs/gs_digital/bin/python}" \
  "${INIT[@]}" "$@"
