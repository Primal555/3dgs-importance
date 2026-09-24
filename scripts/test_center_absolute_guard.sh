#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry: new runs must not accidentally re-enable the rejected
# strict loss-approval experiment. Exact old resumes remain available via CLI.
if [[ -n "${STEP_GUARD+x}" ]]; then
  echo 'STEP_GUARD is retired. Unset it; this entry now runs continuous soft updates.' >&2
  exit 1
fi
echo 'Using continuous updates instead of the retired loss-rejection guard.'
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/test_center_absolute_soft.sh" "$@"
