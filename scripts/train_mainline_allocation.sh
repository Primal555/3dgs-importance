#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo 'Mainline allocation: calibrated first-order v2 (algorithm reference c1162c2).'
exec bash "$PROJECT/scripts/test_structured_evidence_allocation.sh" "$@"
