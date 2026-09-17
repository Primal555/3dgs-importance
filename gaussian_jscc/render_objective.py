"""Render-first task, bounded-memory multiview backward and initialization only.

The image MSE is the distortion definition, not a weighted parameter proxy.
The source renderer is a training target only; it is never receiver input.
"""
import torch
from torch.nn import functional as F


def bootstrap_loss(pred, target):
    """Optional initialization: one SmoothL1 over codec-normalized features.

    XYZ uses the existing global bbox normalization and attributes the stored
    standardization. No extra XYZ scale, six attribute weights, or projection.
    This is an engineering initializer, NOT the communication quality metric.
    """
    return F.smooth_l1_loss(pred, target)


def image_distortion(image, target):
    # Do not clamp the prediction: out-of-range RGB still needs gradients.
    return (image - target.detach()).square().mean()


@torch.no_grad()
def image_metrics(image, target):
    """MSE/PSNR use unclipped renderer values; SSIM uses displayed RGB [0,1]."""
    from utils.loss_utils import ssim
    mse = image_distortion(image, target)
    return {'mse': float(mse), 'psnr': float(-10*mse.clamp_min(1e-12).log10()),
            'ssim': float(ssim(image.clamp(0, 1), target.clamp(0, 1))),
            'l1': float((image-target).abs().mean())}


class MultiViewRenderTask:
    """One noisy decoded scene, equally weighted cameras, no parameter target.

    __call__ supplies the normal autograd graph; backward_scene supplies the
    identical gradient one camera at a time for full_scene_step replay mode.
    """
    def __init__(self, cameras, reference, degree, white_background=False):
        if not cameras:
            raise ValueError('at least one training view is required')
        self.cameras, self.reference = cameras, reference
        self.degree, self.white_background = degree, white_background
        self.stats = {}

    def _terms(self, scene):
        from .rendering import render
        for camera in self.cameras:
            image = render(scene, camera, self.degree, self.white_background)
            yield image_distortion(image, self.reference.get(camera, scene.device))

    def __call__(self, scene, retained_ids=None):
        terms = list(self._terms(scene))
        self.stats = {'view_mse': [float(t.detach()) for t in terms],
                      'views_per_step': len(terms)}
        # Empty scenes still have a well-defined image loss (for q0 policies).
        return torch.stack(terms).mean() + scene.sum()*0

    def backward_scene(self, scene):
        if not scene.is_leaf or not scene.requires_grad:
            raise ValueError('streamed rendering requires a differentiable scene leaf')
        gradient = torch.zeros_like(scene)
        values = []
        for term in self._terms(scene):
            if not torch.isfinite(term):
                raise RuntimeError('nonfinite image distortion')
            values.append(float(term.detach()))
            gradient.add_(torch.autograd.grad(term + scene.sum()*0, scene)[0],
                          alpha=1/len(self.cameras))
        scene.grad = gradient
        self.stats = {'view_mse': values, 'views_per_step': len(values)}
        return scene.new_tensor(sum(values)/len(values))


def spaced_indices(size, count):
    if size < 1 or count < 1:
        raise ValueError('size/count must be positive')
    return torch.linspace(0, size-1, min(size, count)).round().long().tolist()


def split_cameras(cameras, validation_views, train_views=0):
    """Deterministic held-out views, spread across the camera list, never overlap."""
    if validation_views < 1 or len(cameras) <= validation_views or train_views < 0:
        raise ValueError('need training cameras in addition to held-out validation views')
    held = spaced_indices(len(cameras), validation_views)
    train = [i for i in range(len(cameras)) if i not in held]
    if train_views:
        train = [train[i] for i in spaced_indices(len(train), train_views)]
    return [cameras[i] for i in train], [cameras[i] for i in held], train, held
