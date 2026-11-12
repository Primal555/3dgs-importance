#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU}"
SOURCE="${1:?Usage: test_student_late_rd_controls.sh STUDENT_RUN NEW_OUT [options]}"
OUT="${2:?Set NEW output directory}"
shift 2
PYTHON="${PYTHON_BIN:-python}"
command -v "$PYTHON" >/dev/null || { echo 'Activate maskgs or set PYTHON_BIN' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "Use a new output directory: $OUT" >&2; exit 1; }
for CHECKPOINT in best latest; do
  [[ -f "$SOURCE/checkpoints/$CHECKPOINT/student.pt" ]] || { echo "Missing $CHECKPOINT checkpoint" >&2; exit 1; }
done
mkdir -p "$OUT"
echo "GPU: $CUDA_VISIBLE_DEVICES; completed student: $SOURCE; output: $OUT"
echo 'Best and latest: 3 independent one-step probes each, restored Adam and weights; no cumulative training.'
echo 'Original calibration plus32 diagnostic views; prior training overlap explicitly recorded.'
echo 'Separate positive upgrades/downgrades, keep changes, and evaluated/unseen candidate changes.'
for CHECKPOINT in best latest; do
  "$PYTHON" -u -m gaussian_jscc.student_rd_step_audit \
    --source "$SOURCE" --out "$OUT/$CHECKPOINT" --checkpoint "$CHECKPOINT" \
    --probes 3 --diagnostic-views 32 --direction-maps "$@"
done
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
import tarfile
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for label in ('best', 'latest'):
    report = json.loads((root/label/'review_summary.json').read_text(encoding='utf8'))
    summary[label] = dict(source_step=report['plan']['source_step'],
        diagnosis=report['diagnosis'], action_directions=report['action_directions'],
        probes=[dict(probe=r['probe'], transitions=r['transitions'],
            surrogate_change=r['soft_surrogate_change'],
            predicted_discrete_change=r['predicted_discrete_change'],
            actual_rd_changes={k:v['delta_vs_baseline'] for k,v in r['maps'].items()})
            for r in report['probes']])
(root/'review_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(root/'complete.json').write_text(json.dumps(dict(status='complete', checkpoints=['best','latest'])), encoding='utf8')
with tarfile.open(root/'review.tar.gz', 'w:gz') as archive:
    for label in ('best', 'latest'):
        for name in ('plan.json','complete.json','review_summary.json'):
            archive.add(root/label/name, arcname=f'{label}/{name}')
    for name in ('review_summary.json','complete.json'):
        archive.add(root/name, arcname=name)
print(f'Complete. Download {root / "review.tar.gz"}', flush=True)
PY
