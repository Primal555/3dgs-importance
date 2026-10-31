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


@torch.no_grad()
def verify_batched_execution(device):
    """Same-backend serial/batched check before resuming a production run."""
    from types import SimpleNamespace
    from .pixel_local_renderer import capture
    from .physical_benefit_features import visibility_statistics
    from .spatial_benefit_features import spatial_inputs
    device = torch.device(device)
    transform = torch.eye(4, device=device); transform[3, 2] = 3
    camera = SimpleNamespace(image_width=31, image_height=17, FoVx=1.2, FoVy=1.2,
        world_view_transform=transform, full_proj_transform=torch.eye(4, device=device),
        camera_center=torch.zeros(3, device=device))
    raw = torch.zeros(5, 14, device=device); raw[:, 7] = 1; raw[:, 4:7] = -2
    raw[:, 0] = torch.linspace(-1, 1, 5, device=device)
    tile = capture(raw, camera, 0, torch.ones(5, device=device), backend=device.type)
    target = tile.image*.9
    tile._physical_stats = visibility_statistics(tile, target)
    changed = [torch.arange(3, device=device), torch.arange(3, device=device), torch.arange(3, 5, device=device)]
    raws = [raw[x][None].repeat(4, 1, 1) for x in changed]
    tiers = [torch.ones(r.shape[:2], dtype=torch.long, device=device) for r in raws]
    raws[0][2, 0, 0] += 2; raws[1][3, 1, 11] += .2
    for t in tiers:
        t[0, 0] = 0
    serial = [torch.stack([tile.project(r[q], (t[q] > 0).to(raw)) for q in range(4)])
              for r, t in zip(raws, tiers)]
    batched = project_candidates(tile, raws, tiers)
    projection_error = feature_error = 0.
    for a, b in zip(serial, batched):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
        projection_error = max(projection_error, float((a-b).abs().max()))
    for indices in ([0, 1], [2]):
        ids = torch.stack([changed[i] for i in indices])
        stats = tile._physical_stats[ids]
        points = torch.tensor([i % len(changed[j]) for i, j in enumerate(indices)], device=device)
        static = torch.zeros(len(indices), 24, device=device)
        inputs = spatial_inputs_batch(static, stats, torch.stack([raws[i] for i in indices]),
            torch.stack([batched[i] for i in indices]), points, tile, target, ids,
            .01, .006, [tuple(changed[i].tolist()) for i in indices])
        for local, i in enumerate(indices):
            old = spatial_inputs(static[local], stats[local], raws[i], serial[i], int(points[local]),
                                  tile, target, ids[local], .01, .006)
            for key in old:
                torch.testing.assert_close(old[key], inputs[local][key], rtol=3e-5, atol=2e-6)
                feature_error = max(feature_error, float((old[key]-inputs[local][key]).abs().max()))
    return dict(batched_projection_max_abs=projection_error, batched_features_max_abs=feature_error,
                batched_execution_backend=device.type)
