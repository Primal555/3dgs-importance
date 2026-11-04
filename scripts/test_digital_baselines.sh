#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
CODEC="${1:?Usage: test_digital_baselines.sh CODEC_RUN NEW_OUT [options]}"
OUT="${2:?Set output directory}"; shift 2
EVAL_PYTHON="${EVAL_PYTHON:-${PYTHON_BIN:-python}}"
ENV_PARENT="$(conda info --base)/envs"
DIGITAL_PYTHON="${DIGITAL_PYTHON:-${DIGITAL_ENV:-$ENV_PARENT/gs_digital}/bin/python}"
MESONGS_PYTHON="${MESONGS_PYTHON:-${MESONGS_ENV:-$ENV_PARENT/gs_mesongs}/bin/python}"
for interpreter in "$DIGITAL_PYTHON" "$MESONGS_PYTHON"; do
  [[ -x "$interpreter" ]] || { echo "Missing interpreter: $interpreter; run scripts/setup_comparison_baselines.sh" >&2; exit 1; }
done
INIT=(); if [[ "${RESUME:-0}" == 1 ]]; then INIT=(--resume); fi
echo "GPU: $CUDA_VISIBLE_DEVICES; source codec scene records: $CODEC; output: $OUT"
echo 'Same pruned PLY; zlib or MesonGS -> actual 5G LDPC/16QAM/AWGN -> archive-only receiver -> shared evaluation.'
echo 'Default: Truck, 10dB Es/N0, k4096/n6144, 4 trials, MesonGS c1 without rendering finetune.'
echo 'Progress is in the main log; download OUT/review.tar.gz after completion.'
exec "$EVAL_PYTHON" -u -m comparison_baselines.run \
  --run "$CODEC" --out "$OUT" --digital-python "$DIGITAL_PYTHON" \
  --mesongs-python "$MESONGS_PYTHON" --eval-python "$EVAL_PYTHON" "${INIT[@]}" "$@"
