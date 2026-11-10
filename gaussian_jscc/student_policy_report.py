"""Small deployment report; source features and optimization state stay server-side."""
from pathlib import Path
import tarfile
from .budgeted_allocation_state import read_log
from .modular_experiment import json_write


def finish_student_report(out, complete):
    out = Path(out)
    rows = {r['event']: r for r in read_log(out/'metrics/validation.jsonl')[0]
            if not r['event'].startswith('calibration_')}
    comparisons = {}
    for baseline in ('teacher', 'uniform_q2', 'uniform_q3'):
        if baseline not in rows: continue
        b = rows[baseline]; s = rows['selected']
        comparisons[baseline] = dict(source_psnr_difference=s['source_psnr']-b['source_psnr'],
            photo_psnr_difference=s['photo_psnr']-b['photo_psnr'],
            cost_saving=1-s['total_channel_uses']/b['total_channel_uses'],
            rd_difference=s['rd_score']-b['rd_score'])
        if baseline == 'uniform_q2':
            comparisons[baseline]['meets_q2_goal'] = s['source_psnr'] >= b['source_psnr'] and s['total_channel_uses'] < b['total_channel_uses']
    json_write(out/'review_summary.json', dict(completion=complete, rows=rows, comparisons=comparisons))
    temporary = out/'review.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(out.rglob('*')):
            relative = path.relative_to(out)
            if path.is_file() and relative.parts[0] in ('metrics', 'evaluation', 'deployment', 'plan.json',
                    'training.json', 'complete.json', 'progress.json', 'review_summary.json', 'execution_check.json'):
                if '.uncheckpointed_' not in path.name:
                    archive.add(path, arcname=relative.as_posix(), recursive=False)
    temporary.replace(out/'review.tar.gz')
