"""Camera identity and idempotent multi-scene evaluation snapshots."""
import hashlib
import json
import time
from pathlib import Path

import torch

from .modular_experiment import json_write


def camera_fingerprint(cameras):
    """Hash rendered-camera geometry and actual resized RGB, not just filenames.

    Hash CPU tensor buffers directly, avoiding another full image-byte copy.
    The test split is identity evidence only; it never supplies training loss.
    """
    digest = hashlib.sha256()
    fields = ('image_name','image_width','image_height','FoVx','FoVy','R','T',
              'world_view_transform','projection_matrix','full_proj_transform',
              'camera_center','original_image')
    for index,camera in enumerate(cameras):
        digest.update(f'camera:{index}'.encode())
        for name in fields:
            if not hasattr(camera,name): continue
            value = getattr(camera,name)
            digest.update(name.encode())
            if isinstance(value,(str,int,float,bool)) or value is None:
                digest.update(json.dumps(value,allow_nan=False).encode())
            else:
                tensor = torch.as_tensor(value).detach().cpu().contiguous()
                digest.update(json.dumps([str(tensor.dtype),list(tensor.shape)]).encode())
                digest.update(memoryview(tensor.numpy()).cast('B'))
    return digest.hexdigest()


def archive_snapshot(path, step, phase):
    """Remove only an incomplete snapshot, retaining an inspectable archive."""
    path = Path(path)
    if not path.exists(): return
    lines = path.read_text(encoding='utf-8').splitlines()
    retained, removed = [], []
    for line in lines:
        if not line.strip(): continue
        row = json.loads(line)
        (removed if row['step']==step and row['phase']==phase else retained).append(line)
    if not removed: return
    archive = path.with_name(f'{path.stem}.partial_{step}_{phase}_{time.time_ns()}.jsonl')
    archive.write_text('\n'.join(removed)+'\n',encoding='utf-8')
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text('\n'.join(retained)+('\n' if retained else ''),encoding='utf-8')
    temporary.replace(path)


class EvaluationJournal:
    """Metrics commit after ALL scenes + macro complete; pending work replays.

    This small journal is independent of the optimizer checkpoint so a crash
    between validation and the next save cannot lose completion information.
    Snapshot model identity prevents trusting markers for different weights.
    """
    def __init__(self, out, scenes, step):
        self.out, self.scenes = Path(out), list(scenes)
        self.path = self.out/'metrics/evaluations.json'
        state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        self.state = {k:v for k,v in state.items() if v['step']<=step}
        if state != self.state: json_write(self.path,self.state)

    def is_complete(self, step, phase, identity):
        saved = self.state.get(f'{step}:{phase}',{})
        return saved.get('status')=='complete' and saved.get('model_id')==identity

    def start(self, step, phase, identity):
        self.state[f'{step}:{phase}'] = dict(step=step,phase=phase,model_id=identity,status='pending')
        json_write(self.path,self.state)
        split = 'test' if phase=='test' else 'validation'
        for name in self.scenes:
            archive_snapshot(self.out/'scenes'/name/'metrics'/f'{split}.jsonl',step,phase)
        archive_snapshot(self.out/'metrics'/f'macro_{split}.jsonl',step,phase)

    def complete(self, step, phase):
        self.state[f'{step}:{phase}']['status'] = 'complete'
        json_write(self.path,self.state)
