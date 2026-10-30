#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
RUN="${1:?Usage: test_physical_benefit_audit.sh PRETRAIN_RUN NEW_AUDIT_OUT [options]}"
OUT="${2:?Set a new separate audit output}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
echo "Physical-benefit final audit; source: $RUN; output: $OUT"
echo 'CPU; existing indexed replay; sampling correction, fixed-pool fit, common-error oracles and physical consistency.'
exec "$PYTHON" -u -m gaussian_jscc.physical_benefit_audit --run "$RUN" --out "$OUT" "$@"
