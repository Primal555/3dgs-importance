#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
RUN="${1:?Usage: test_benefit_diagnostics.sh BENEFIT_PRETRAIN_RUN NEW_OUT [options]}"
OUT="${2:?Set a new diagnostic directory}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
echo "Benefit pretraining: $RUN; diagnostic output: $OUT"
echo 'Fixed training/validation replay; zero renders; source unchanged. Default: CPU, 1000 extra evaluator updates.'
exec "$PYTHON" -u -m gaussian_jscc.benefit_diagnostics --run "$RUN" --out "$OUT" "$@"
