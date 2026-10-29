"""Disk-backed complete intervention groups and corrected sampling."""

import copy
import math
import os
import re
import uuid
from pathlib import Path

import torch


def _distribution(weights, size):
    values = torch.as_tensor(weights, dtype=torch.float64).flatten()
    if values.numel() != size or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("sampling weights must be finite, nonnegative, and match candidates")
    total = values.sum()
    return values / total if total > 0 else torch.full((size,), 1.0 / size, dtype=torch.float64)


def sample_sources(candidates, impact, difficulty, count, generator):
    """Draw with replacement and return exact proposal and uniform-target weights."""
    candidates = list(candidates)
    n = len(candidates)
    if n == 0 or count < 0:
        raise ValueError("candidates must be nonempty and count nonnegative")
    probabilities = (0.5 / n + 0.25 * _distribution(impact, n)
                     + 0.25 * _distribution(difficulty, n))
    indices = torch.multinomial(probabilities, count, replacement=True, generator=generator) if count else []
    return [{"source": candidates[int(i)], "proposal_probability": float(probabilities[int(i)]),
             "importance_weight": 1.0 / (n * float(probabilities[int(i)]))} for i in indices]


class PixelReplay:
    """Store groups in immutable shards; retain evicted shards for older checkpoints."""

    def __init__(self, root, seed=0, max_groups=1024, validation_groups=64):
        if max_groups <= 0 or validation_groups < 0:
            raise ValueError("invalid replay capacity")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_groups = int(max_groups)
        self.validation_groups = int(validation_groups)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.store_id = uuid.uuid4().hex
        self.training = []
        self.validation = []
        self.priorities = {}
        self.validation_frozen = False

    def add(self, record, validation=False):
        if validation and self.validation_frozen:
            raise ValueError("validation set is frozen")
        if validation and len(self.validation) >= self.validation_groups:
            return None
        if not isinstance(record, dict) or not {"source", "step", "views"} <= record.keys():
            raise ValueError("record must contain source, step, and views")
        group = dict(record)
        group["validation"] = bool(validation)
        record_id = uuid.uuid4().hex
        folder = self.root / ("validation" if validation else "training") / self.store_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (record_id + ".pt")
        temporary = folder / (record_id + ".tmp")
        try:
            torch.save(group, temporary)
            os.link(temporary, path)
            if not validation:
                with (folder / (record_id + '.owned')).open('x', encoding='ascii') as marker:
                    marker.write(record_id)
        finally:
            if temporary.exists():
                temporary.unlink()
        entry = {"id": record_id, "path": str(path.relative_to(self.root)),
                 "size": path.stat().st_size}
        if validation:
            self.validation.append(entry)
        else:
            self.training.append(entry)
            self.priorities[record_id] = 1.0
            if len(self.training) > self.max_groups:
                old = self.training.pop(0)
                del self.priorities[old["id"]]
        return record_id

    def freeze_validation(self):
        self.validation_frozen = True

    def _read(self, entry):
        path = self.root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"replay shard missing: {path}")
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"replay shard size mismatch: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def sample(self, count):
        if count < 0 or (count and not self.training):
            raise ValueError("invalid count or empty training replay")
        if count == 0:
            return []
        n = len(self.training)
        priorities = _distribution([self.priorities[e["id"]] for e in self.training], n)
        probabilities = 0.5 / n + 0.5 * priorities
        indices = torch.multinomial(probabilities, count, replacement=True, generator=self.generator)
        records = []
        for index in indices.tolist():
            entry = self.training[index]
            group = self._read(entry)
            p = float(probabilities[index])
            group.update(replay_record_id=entry["id"], replay_probability=p,
                         replay_importance_weight=1.0 / (n * p))
            records.append(group)
        return records

    def validation_records(self):
        for entry in self.validation:
            yield self._read(entry)

    def training_records(self):
        for entry in self.training:
            yield self._read(entry)

    def update_priority(self, record_id, error):
        value = float(error)
        if record_id not in self.priorities:
            raise KeyError(record_id)
        if not math.isfinite(value) or value < 0:
            raise ValueError("priority must be finite and nonnegative")
        self.priorities[record_id] = value

    def state_dict(self):
        return {"version": 1, "root": str(self.root), "store_id": self.store_id, "max_groups": self.max_groups,
                "validation_groups": self.validation_groups, "training": copy.deepcopy(self.training),
                "validation": copy.deepcopy(self.validation), "priorities": dict(self.priorities),
                "validation_frozen": self.validation_frozen, "rng": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if (state["version"] != 1 or state["root"] != str(self.root)
                or state["max_groups"] != self.max_groups
                or state["validation_groups"] != self.validation_groups):
            raise ValueError("replay identity or capacity mismatch")
        for entry in state["training"] + state["validation"]:
            path = self.root / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(f"replay shard missing: {path}")
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"replay shard size mismatch: {path}")
        self.training = copy.deepcopy(state["training"])
        self.validation = copy.deepcopy(state["validation"])
        self.priorities = dict(state["priorities"])
        self.validation_frozen = state["validation_frozen"]
        self.store_id = state['store_id']
        self.generator.set_state(state["rng"])

    def manifest(self):
        """Small JSON-serializable checkpoint reference list for safe collection."""
        return {'version': 1, 'root': str(self.root), 'store_id': self.store_id,
                'training': [entry['path'] for entry in self.training],
                'validation': [entry['path'] for entry in self.validation]}

    def garbage_collect(self, checkpoint_manifests):
        """Remove owned evicted training shards absent from all supplied manifests.

        Caller must supply manifests for every existing checkpoint, and must not
        collect while a checkpoint or its manifest is being replaced.
        """
        manifests = list(checkpoint_manifests)
        retained = {entry['path'] for entry in self.training}
        for manifest in manifests:
            if (not isinstance(manifest, dict) or manifest.get('version') != 1
                    or manifest.get('root') != str(self.root)
                    or not re.fullmatch(r'[0-9a-f]{32}', str(manifest.get('store_id', '')))
                    or not isinstance(manifest.get('training'), list)
                    or not isinstance(manifest.get('validation'), list)):
                raise ValueError('invalid checkpoint replay manifest')
            for name in ('training', 'validation'):
                for rel in manifest[name]:
                    if not isinstance(rel, str):
                        raise ValueError('invalid checkpoint shard path')
                    path = Path(rel)
                    if (len(path.parts) != 3 or path.parts[0] != name
                            or path.parts[1] != manifest['store_id']
                            or not re.fullmatch(r'[0-9a-f]{32}\.pt', path.name)):
                        raise ValueError('invalid checkpoint shard path')
                    if not (self.root / path).is_file():
                        raise FileNotFoundError(f'checkpoint replay shard missing: {self.root / path}')
                    if name == 'training':
                        retained.add(str(path))
        folder = self.root / 'training' / self.store_id
        if not folder.exists():
            return 0
        if folder.is_symlink() or folder.resolve() != self.root / 'training' / self.store_id:
            raise ValueError('invalid replay ownership directory')
        removed = 0
        for path in folder.glob('*.pt'):
            if not re.fullmatch(r'[0-9a-f]{32}\.pt', path.name) or path.is_symlink():
                continue
            marker = path.with_suffix('.owned')
            if (str(path.relative_to(self.root)) in retained or marker.is_symlink()
                    or not marker.is_file() or marker.read_text(encoding='ascii') != path.stem):
                continue
            path.unlink()
            marker.unlink()
            removed += 1
        return removed
