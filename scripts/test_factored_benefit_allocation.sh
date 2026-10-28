#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo 'Factored evaluator: shared retention value + signed adjacent-tier increments; decision/ranking supervision.'
exec bash "$PROJECT/scripts/test_learned_benefit_allocation.sh" "$@" --benefit-variant factored
