"""Normalized rendering and exact measured cost bridge for the historical ST path."""
import math
import torch
from .render_objective import MaskedMultiViewRenderTask


class NormalizedMaskedTask(MaskedMultiViewRenderTask):
    def __init__(self, *args, distortion, d_ref, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(d_ref) or d_ref <= 0: raise ValueError('positive d_ref required')
        self.distortion, self.d_ref = distortion, d_ref

    def _parts(self, scene):
        from .rendering import render
        for camera in self.cameras:
            image = render(scene[:, :-1], camera, self.degree, self.white_background, scene[:, -1])
            yield self.distortion.components(image, self.reference.get(camera, scene.device))

    def _stats(self, values):
        parts = torch.stack(values).detach()
        mean = parts.mean(0)
        self.stats = dict(view_mse=parts[:, 0].tolist(), views_per_step=len(parts),
            source_mse=float(mean[0]), source_structure_error=float(mean[1]),
            source_distortion=float(self.distortion.combine(mean)), lambda_s=self.distortion.lambda_s,
            normalized_distortion=float(self.distortion.combine(mean))/self.d_ref)

    def __call__(self, scene, retained_ids=None):
        parts = list(self._parts(scene)); self._stats(parts)
        return self.distortion.combine(torch.stack(parts).mean(0))/self.d_ref + scene.sum()*0

    def backward_scene(self, scene):
        if not scene.is_leaf or not scene.requires_grad: raise ValueError('differentiable scene leaf required')
        gradient, values = torch.zeros_like(scene), []
        for parts in self._parts(scene):
            term = self.distortion.combine(parts)/self.d_ref
            if not torch.isfinite(term): raise RuntimeError('nonfinite normalized image loss')
            gradient.add_(torch.autograd.grad(term+scene.sum()*0, scene)[0], alpha=1/len(self.cameras))
            values.append(parts.detach())
        scene.grad = gradient
        self._stats(values)
        return self.distortion.combine(torch.stack(values).mean(0))/self.d_ref


class DenseRateMeter:
    """Map packet rows back to PLY rows before exact reliable-stream accounting."""
    def __init__(self, exact, scene, reference):
        self.exact, self.scene = exact, scene
        self.count, self.bits = len(scene.raw), exact.bits
        self.normalizer = reference['c_ref']/self.count
        self.position_unit = reference['q3_cost']['coordinate_channel_uses']/self.count

    def details(self, packet_q):
        if packet_q.shape != (self.count,): raise ValueError('one tier per packet row required')
        original = torch.empty_like(packet_q, device='cpu')
        original[self.scene.order.cpu()] = packet_q.cpu()
        measured = self.exact.details(original)
        side = (measured['coordinate_channel_uses'] + measured['metadata_bytes']*8/self.bits
                + measured['scale_channel_uses'])
        return dict(position_channel_uses_estimate=measured['coordinate_channel_uses'],
            tier_map_proxy_bytes=measured['metadata_bytes'],
            allocation_side_uses_per_source_gaussian=side/self.count,
            allocation_uses_per_source_gaussian=measured['total_channel_uses']/self.count,
            scale_channel_uses=measured['scale_channel_uses'], scale_stream_bytes=measured['scale_stream_bytes'])
