"""Camera partitions and resumable per-scene view cycles for codec refinement."""
import copy
import hashlib

import torch

from .allocation_mirror import ViewWindowScheduler
from .modular_multiscene_state import camera_fingerprint
from .render_objective import spaced_indices
from .successive_multiscene_inputs import make_scene


def refinement_split(record, validation_views, views_per_step):
    train = list(record['train_view_names'])
    held = list(record['validation_view_names'])
    test = list(record['test_view_names'])
    if len(set(train+held+test)) != len(train+held+test):
        raise ValueError('source training/validation/test views overlap or repeat')
    extra = validation_views-len(held)
    if extra < 0 or views_per_step < 1 or len(train)-extra < views_per_step:
        raise ValueError('retain source validation views and leave enough distinct training views')
    added = [train[i] for i in spaced_indices(len(train), extra)] if extra else []
    added_set = set(added)
    return dict(train_view_names=[n for n in train if n not in added_set],
                validation_view_names=held+added, test_view_names=test,
                source_validation_view_names=held, added_validation_view_names=added)


def make_refinement_scene(spec, raw, degree, model, args, out, source_record, split):
    # Verify original geometry, camera fingerprints and numerical settings before
    # repartitioning the already loaded cameras. Images and blocks load once.
    original = copy.copy(args)
    original.views_per_step = source_record['views_per_step']
    original.validation_views = len(source_record['validation_view_names'])
    scene = make_scene(spec, raw, degree, model, original, out, old=source_record)
    lookup = {str(c.image_name): c for c in scene.cameras+scene.val_cameras}
    scene.cameras = [lookup[n] for n in split['train_view_names']]
    scene.val_cameras = [lookup[n] for n in split['validation_view_names']]
    scene.views_per_step = args.views_per_step
    scene.record = copy.deepcopy(scene.record)
    scene.record.update(train_view_names=split['train_view_names'],
                        validation_view_names=split['validation_view_names'],
                        views_per_step=args.views_per_step)
    scene.record['camera_fingerprints'].update(
        train=camera_fingerprint(scene.cameras), validation=camera_fingerprint(scene.val_cameras))
    return scene


class RefinementViewCycle:
    """Reuse student view rotation; persist actual visit counts and generator state."""
    def __init__(self, names, seed, scene, tier, state=None):
        key = f'{seed}:{scene}:q{tier}'.encode()
        local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
        self.scheduler = ViewWindowScheduler(names, local_seed)
        self.counts = torch.zeros(len(names), dtype=torch.long)
        self.updates = 0
        if state is not None:
            value = dict(state['scheduler'])
            value['rng'] = torch.tensor(value['rng'], dtype=torch.uint8)
            self.scheduler.load_state_dict(value)
            self.counts = torch.tensor(state['counts'], dtype=torch.long)
            self.updates = state['updates']
            if self.counts.shape != (len(names),) or (self.counts < 0).any():
                raise ValueError('invalid refinement camera coverage')

    def next(self, count):
        indices = self.scheduler.next(count)
        self.counts[indices] += 1
        self.updates += 1
        return indices

    def state_dict(self):
        value = self.scheduler.state_dict()
        value['rng'] = value['rng'].tolist()
        return dict(scheduler=value, counts=self.counts.tolist(), updates=self.updates)

    def coverage(self):
        return dict(pool=len(self.counts), covered=int((self.counts > 0).sum()),
                    total_view_uses=int(self.counts.sum()), updates=self.updates,
                    min_visits=int(self.counts.min()), max_visits=int(self.counts.max()),
                    mean_visits=float(self.counts.double().mean()))
