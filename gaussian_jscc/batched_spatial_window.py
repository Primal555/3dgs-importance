"""Execution-only batching for the frozen-codec spatial benefit allocator."""
import time
import torch

from .batched_spatial_features import project_candidates, spatial_inputs_batch
from .pixel_benefit_features import exact_values
from .spatial_benefit_features import SpatialWindow


class BatchedSpatialWindow(SpatialWindow):
    @property
    def candidate_batch_size(self):
        size = int(self.args.allocation_batch)
        if size < 1:
            raise ValueError('allocation batch must be positive')
        return size

    @property
    def execution_settings(self):
        return dict(mode='batched_spatial_v1', allocation_batch=self.candidate_batch_size)

    def candidate_views(self, evaluated, entries, captures, targets, labels, timings):
        timings['candidate_projection_calls'] = 0
        timings['spatial_feature_batches'] = 0
        for group, batch, loc, raw, tiers in self.candidate_batches(evaluated, entries, timings):
            sources = group.tolist()
            ids = self.scene.group_ids[batch]
            metadata = []; raws = []; tier_maps = []; partitions = {}
            for i, source in enumerate(sources):
                row, col = int(loc[i, 1]), int(loc[i, 2])
                valid = ids[row] >= 0
                block_ids = ids[row, valid].cpu().long()
                point = int(valid[:col].sum())
                changed = self.positions[block_ids, 3].to(raw.device)
                metadata.append((row, col, point, changed, tuple(block_ids.tolist())))
                raws.append(raw[i, :, valid]); tier_maps.append(tiers[i, :, valid])
                partitions.setdefault(len(block_ids), []).append(i)
            all_views = [[] for _ in sources]
            for tile, target in zip(captures, targets):
                started = time.perf_counter()
                projected = project_candidates(tile, raws, tier_maps)
                timings['projection_seconds'] += time.perf_counter()-started
                timings['candidate_projection_calls'] += 1
                inputs = [None]*len(sources)
                started = time.perf_counter()
                for indices in partitions.values():
                    changed = torch.stack([metadata[i][3] for i in indices])
                    points = torch.tensor([metadata[i][2] for i in indices], device=raw.device)
                    values = spatial_inputs_batch(self.static[group[indices]].to(raw),
                        tile._physical_stats[changed], torch.stack([raws[i] for i in indices]),
                        torch.stack([projected[i] for i in indices]), points, tile, target, changed,
                        self.reference['d_ref'], self.reference['lambda_s'], [metadata[i][4] for i in indices])
                    for i, value in zip(indices, values):
                        inputs[i] = value
                    timings['spatial_feature_batches'] += 1
                feature_seconds = (time.perf_counter()-started)/len(sources)
                for i, source in enumerate(sources):
                    view = dict(inputs=inputs[i], height=tile.height, width=tile.width,
                        d_ref=self.reference['d_ref'], tile_count=0, feature_seconds=feature_seconds)
                    if source in labels:
                        started = time.perf_counter()
                        changed = metadata[i][3]
                        view['values'] = exact_values(tile, changed, projected[i], target,
                            self.reference['lambda_s'], self.reference['d_ref']).cpu()
                        view['tile_count'] = len(tile.affected_tiles(changed, projected[i], halo_tiles=1))
                        view['teacher_seconds'] = time.perf_counter()-started
                    all_views[i].append(view)
            for i, source in enumerate(sources):
                row, col = metadata[i][:2]
                yield source, batch, row, col, all_views[i]
