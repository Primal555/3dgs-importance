"""Opt-in differentiable image losses for fixed-deployment codec finetuning.

Evaluation stays unchanged. MSE uses raw renderer RGB; SSIM/LPIPS use displayed
RGB [0,1]. The LPIPS backbone is frozen, NOT its gradient with respect to RGB.
"""
import math

import torch

from .perceptual_metrics import LPIPSEvaluator
from .render_objective import MultiViewRenderTask, image_distortion


PAPER_WEIGHTS = {'mse_ssim': 1/3, 'mse_lpips': 1.}
WEIGHT_REFERENCES = {
    'mse_ssim': dict(
        url='https://arxiv.org/html/2510.20067',
        section='III-C and IV; beta=.25 reconstruction-loss setting',
        source_formula='(1-beta)*MSE + beta*(1-SSIM)',
        conversion='relative SSIM/MSE weight .25/.75 = 1/3; MSE coefficient kept at 1'),
    'mse_lpips': dict(
        url='https://arxiv.org/html/2211.13772',
        section='V-B; DeepJSCC forward-operator training baseline (not a claim of all GenerativeJSCC settings)',
        source_formula='MSE + LPIPS',
        conversion='reference relative weight 1; our RGB MSE is a mean on raw renderer RGB'),
}


class PerceptualImageObjective(torch.nn.Module):
    def __init__(self, kind, weight, device):
        super().__init__()
        if kind not in ('mse_ssim', 'mse_lpips'):
            raise ValueError('expected mse_ssim or mse_lpips')
        if weight is None:
            weight = PAPER_WEIGHTS[kind]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('perceptual weight must be finite and positive')
        self.kind, self.weight = kind, float(weight)
        # The evaluator constructor already freezes weights and preserves RNG.
        # Do NOT call its evaluation-only __call__: it detaches the prediction.
        self.lpips_model = LPIPSEvaluator(device).model if kind == 'mse_lpips' else None

    def protocol(self):
        return dict(kind=self.kind, aux_weight=self.weight,
                    formula='MSE + weight * '+('(1-SSIM)' if self.kind == 'mse_ssim' else 'LPIPS'),
                    reference='source PLY rendering at the same training camera; never photo',
                    mse_input='unclipped renderer RGB', auxiliary_input='RGB clamped to [0,1]',
                    ssim='utils.loss_utils.ssim; window=11; sigma=1.5; RGB mean',
                    lpips='frozen pretrained VGG LPIPS v0.1; 2*x-1; full resolution; input gradients enabled',
                    weighting='paper-referenced relative coefficient; not automatically calibrated or a full-paper reproduction',
                    weight_reference=WEIGHT_REFERENCES[self.kind],
                    reference_default_weight=PAPER_WEIGHTS[self.kind],
                    prefix_anchor='same objective as fixed deployment; existing equal gradient mean')

    def forward(self, image, target):
        if (image.shape != target.shape or image.ndim != 3 or image.shape[0] != 3
                or not image.numel()):
            raise ValueError('image loss requires equally sized nonempty 3xHxW RGB')
        if not torch.isfinite(image).all() or not torch.isfinite(target).all():
            raise ValueError('image loss RGB must be finite')
        target = target.detach()
        mse = image_distortion(image, target)
        a, b = image.clamp(0, 1), target.clamp(0, 1)
        if self.kind == 'mse_ssim':
            from utils.loss_utils import ssim
            extra = 1-ssim(a, b)
            extra_key = 'image_ssim_loss'
        else:
            if min(image.shape[-2:]) < 16:
                raise ValueError('VGG LPIPS training requires image height and width >= 16')
            self.lpips_model.eval()
            extra = self.lpips_model(a.float().mul(2).sub(1)[None],
                                     b.float().mul(2).sub(1)[None])
            if extra.numel() != 1:
                raise RuntimeError('LPIPS training loss must be scalar')
            extra = extra.reshape(())
            extra_key = 'image_lpips_loss'
        weighted = self.weight*extra
        loss = mse+weighted
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite perceptual image loss')
        return loss, dict(image_mse=mse, **{extra_key:extra}, image_aux_weighted=weighted)


class PerceptualMultiViewRenderTask(MultiViewRenderTask):
    """Retain the existing streamed replay VJP, including one view at a time."""
    def __init__(self, cameras, reference, degree, white_background, objective):
        super().__init__(cameras, reference, degree, white_background)
        self.objective = objective
        self.components = []

    def _terms(self, scene):
        from .rendering import render
        self.components = []
        for camera in self.cameras:
            image = render(scene, camera, self.degree, self.white_background)
            loss, terms = self.objective(image, self.reference.get(camera, scene.device))
            self.components.append({key:float(value.detach()) for key,value in terms.items()})
            yield loss

    def _record_components(self):
        self.stats['view_objective'] = self.stats['view_mse']
        self.stats['view_mse'] = [row['image_mse'] for row in self.components]
        self.stats.update({key:sum(row[key] for row in self.components)/len(self.components)
                           for key in self.components[0]})
        self.stats.update(image_objective=self.objective.kind, image_aux_weight=self.objective.weight)
        self.stats['image_aux_to_mse_ratio'] = self.stats['image_aux_weighted']/max(self.stats['image_mse'],1e-12)

    def __call__(self, scene, retained_ids=None):
        loss = super().__call__(scene, retained_ids)
        self._record_components()
        return loss

    def backward_scene(self, scene):
        loss = super().backward_scene(scene)
        self._record_components()
        return loss
