"""Immutable two-resolution metric-space groups for the student policy."""
import math
import torch


def pooled_mean(values, cluster, count):
    result = values.new_zeros(count, values.shape[-1])
    result.index_add_(0, cluster, values)
    return result / torch.bincount(cluster, minlength=count).to(values).clamp_min(1)[:, None]


@torch.no_grad()
def build_graph(xyz, fine_points=16, coarse_points=4, neighbors=16):
    from scipy.spatial import cKDTree
    xyz = torch.as_tensor(xyz).detach().cpu().double()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not torch.isfinite(xyz).all():
        raise ValueError('nonempty finite original-row XYZ required')
    if min(fine_points, coarse_points, neighbors) < 1: raise ValueError('positive graph budgets required')
    unit = (xyz-xyz.amin(0))/float((xyz.amax(0)-xyz.amin(0)).max().clamp_min(1e-9))
    def group(points, target):
        quantized = (points*65535).floor().long()
        def labels(shift):
            keys = quantized >> shift
            packed = (keys[:, 0] << 32) | (keys[:, 1] << 16) | keys[:, 2]
            unique, inverse = torch.unique(packed, sorted=True, return_inverse=True)
            return inverse, len(unique)
        low, high = 0, 16
        while low < high:
            mid = (low+high)//2
            if labels(mid)[1] <= target: high = mid
            else: low = mid+1
        inverse, count = labels(low)
        return inverse, pooled_mean(points, inverse, count)
    cluster1, coords1 = group(unit, max(1, math.ceil(len(unit)/fine_points)))
    cluster2, coords2 = group(coords1, max(1, math.ceil(len(coords1)/coarse_points)))
    def near(points):
        _, ids = cKDTree(points.numpy()).query(points.numpy(), k=min(neighbors, len(points)), workers=1)
        return torch.as_tensor(ids).long().reshape(len(points), -1)
    return dict(cluster1=cluster1, cluster2=cluster2, coords1=coords1.float(), coords2=coords2.float(),
                neighbors1=near(coords1), neighbors2=near(coords2))


def graph_to(graph, device):
    return {k: v.to(device) for k, v in graph.items()}
