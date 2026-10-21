"""Read-only gradient decomposition around the unchanged joint optimizer step.

Diagnostics must neither clip the gradient nor reset Adam. MSE/weighted SSIM
are separated at the same rendered scene; their exact VJPs plus latent MSE
are measured during the existing full-scene replay.
"""
import torch
from torch.nn import functional as F

from .cli import seed_all
from .data import to_raw
from .modular_training import latent_reconstruction_loss
from .optimization import clip_codec_gradients, preserved_rng, update_stats
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .position_delivery import delivered_positions
from .power_followup import joint_inputs, joint_update


RAW_GROUPS = dict(xyz=(0, 3), opacity=(3, 4), scale=(4, 7), rotation=(7, 11),
                  dc=(11, 14), sh=(14, None), color=(11, None), attributes=(3, None))
NORMALIZED_GROUPS = dict(opacity=(0, 1), scale=(1, 4), rotation=(4, 8), color=(8, None))


def gradient_profile(raw, gradient, source_ids, top_k=16):
    if raw.shape != gradient.shape or raw.ndim != 2 or len(source_ids) != len(raw) or raw.shape[1] < 14 or top_k < 1:
        raise ValueError('matching Gaussian rows, gradients and source IDs required')
    g = gradient.detach().double()
    if not torch.isfinite(g).all():
        raise RuntimeError('nonfinite raw scene gradient')
    groups = {}
    for name, (lo, hi) in RAW_GROUPS.items():
        part = g[:, lo:hi]
        groups[name] = dict(norm=float(part.norm()), max_abs=float(part.abs().max()) if part.numel() else 0.)
    scores = g[:, 3:].norm(dim=-1)
    indices = scores.topk(min(top_k, len(scores))).indices.cpu().tolist()
    top = []
    for index in indices:
        row = raw[index].detach().cpu()
        top.append(dict(source_id=int(source_ids[index]), row_index=index, gradient_norm=float(scores[index]),
            xyz=row[:3].tolist(), opacity_logit=float(row[3]), opacity=float(row[3].sigmoid()),
            log_scale=row[4:7].tolist(), scale=row[4:7].exp().tolist(),
            axis_ratio=float((row[4:7].max()-row[4:7].min()).exp()), rotation=row[7:11].tolist(),
            gradients={name: float(g[index, lo:hi].norm()) for name, (lo, hi) in RAW_GROUPS.items()}))
    return dict(groups=groups, top_points=top)


class GradientRenderTask(PerceptualMultiViewRenderTask):
    def __init__(self, *args, source_ids, top_k):
        super().__init__(*args)
        self.source_ids, self.top_k = source_ids, top_k

    def backward_scene(self, scene):
        from .rendering import render
        self.components = []; self.image_gradient_norms = []
        self.mse_gradient = torch.zeros_like(scene); self.ssim_gradient = torch.zeros_like(scene)
        combined = torch.zeros_like(scene); values = []
        for camera in self.cameras:
            image = render(scene, camera, self.degree, self.white_background)
            loss, terms = self.objective(image, self.reference.get(camera, scene.device))
            row = {}
            for key, destination in (('image_mse', self.mse_gradient), ('image_aux_weighted', self.ssim_gradient)):
                scene_grad, image_grad = torch.autograd.grad(terms[key]+scene.sum()*0, (scene, image), retain_graph=True)
                destination.add_(scene_grad, alpha=1/len(self.cameras))
                row[key] = float(image_grad.detach().norm())
            # Use the original combined backward, not a substitute sum of the
            # two diagnostic CUDA backwards. Diagnostics never write .grad.
            combined.add_(torch.autograd.grad(loss+scene.sum()*0, scene)[0], alpha=1/len(self.cameras))
            self.image_gradient_norms.append(dict(image_name=str(camera.image_name), **row))
            self.components.append({key: float(value.detach()) for key, value in terms.items()})
            values.append(float(loss.detach()))
        scene.grad = combined
        self.stats = dict(view_mse=values, views_per_step=len(values))
        self._record_components()
        self.raw_profile = gradient_profile(scene, combined, self.source_ids, self.top_k)
        self.raw_components = {key: gradient_profile(scene, gradient, self.source_ids, self.top_k)
            for key, gradient in (('mse', self.mse_gradient), ('ssim', self.ssim_gradient))}
        # Camera-space depths for the highest-attribute-gradient Gaussians.
        for point in self.raw_profile['top_points']:
            point['camera_depths'] = {}
            for camera in self.cameras:
                if hasattr(camera, 'world_view_transform'):
                    xyz = scene.new_tensor(point['xyz']+[1.])
                    point['camera_depths'][str(camera.image_name)] = float((xyz@camera.world_view_transform.to(scene))[2])
        return scene.new_tensor(sum(values)/len(values))


class ParameterDecomposition:
    def __init__(self, model):
        self.model = model
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.parameters = [p for _, p in self.named]
        count = sum(p.numel() for p in self.parameters)
        self.sums = {k: next(model.parameters()).new_zeros(count) for k in ('mse', 'ssim', 'latent')}
        self.boundary = {}; self.normalized_squared = {k: {name: 0. for name in NORMALIZED_GROUPS} for k in self.sums}
        self.pre_rotation_min = float('inf'); self.pre_rotation_near_zero = 0
        self.task = None
        self.hooks = [model.semantic_decoder.register_forward_hook(self._semantic_boundary)]

    def _semantic_boundary(self, module, inputs, output):
        if output.requires_grad:
            self.boundary['attributes'] = output
            self.boundary['active'] = inputs[1]

    def close(self):
        for hook in self.hooks: hook.remove()

    def __call__(self, raw, weighted_aux, offset):
        task = self.task; count = len(raw)
        objectives = dict(mse=(raw*task.mse_gradient[offset:offset+count]).sum(),
            ssim=(raw*task.ssim_gradient[offset:offset+count]).sum(), latent=weighted_aux)
        attributes = self.boundary['attributes']
        physical = attributes.detach()*self.model.attr_std+self.model.attr_mean
        norms = physical[..., 4:8][self.boundary['active']].norm(dim=-1)
        self.pre_rotation_min = min(self.pre_rotation_min, float(norms.min()))
        self.pre_rotation_near_zero += int((norms < 1e-4).sum())
        for name, objective in objectives.items():
            gradients = torch.autograd.grad(objective, self.parameters+[attributes], allow_unused=True, retain_graph=True)
            flat = torch.cat([(gradient.detach().reshape(-1) if gradient is not None else parameter.new_zeros(parameter.numel()))
                              for parameter, gradient in zip(self.parameters, gradients[:-1])])
            self.sums[name].add_(flat)
            boundary = gradients[-1]
            if boundary is not None:
                for group, (lo, hi) in NORMALIZED_GROUPS.items():
                    self.normalized_squared[name][group] += float(boundary[..., lo:hi].detach().double().square().sum())

    def report(self):
        sums = {k: v.detach().double().cpu() for k, v in self.sums.items()}
        actual = torch.cat([(p.grad.detach().flatten() if p.grad is not None else p.new_zeros(p.numel())) for p in self.parameters]).double().cpu()
        total = sum(sums.values())
        groups = {}; offset = 0
        for name, parameter in self.named:
            groups.setdefault(name.split('.')[0], []).append(slice(offset, offset+parameter.numel()))
            offset += parameter.numel()
        norms = {key: {group: float(torch.cat([value[s] for s in slices]).norm()) for group, slices in groups.items()}
                 for key, value in sums.items()}
        cosine = {}
        for left, right in (('mse', 'ssim'), ('mse', 'latent'), ('ssim', 'latent')):
            denominator = sums[left].norm()*sums[right].norm()
            cosine[left+'__'+right] = float(torch.dot(sums[left], sums[right])/denominator) if denominator > 0 else None
        return dict(parameter_norms=norms, parameter_cosine=cosine,
            gradient_sum_relative_error=float((total-actual).norm()/actual.norm().clamp_min(1e-30)),
            normalized_attribute_gradient_norms={k: {g: v**.5 for g, v in row.items()} for k, row in self.normalized_squared.items()},
            pre_normalization_rotation_min_norm=self.pre_rotation_min,
            pre_normalization_rotation_below_1e_minus4=self.pre_rotation_near_zero)


@torch.no_grad()
def measure_same_input(model, scene, step, phase_step, args):
    from .rendering import render
    device = next(model.parameters()).device
    batches, indices, _ = joint_inputs(scene, step, phase_step, args)
    objective = PerceptualImageObjective('mse_ssim', 1/3, device)
    rows = []; auxiliary = 0.; count = 0
    with preserved_rng(device):
        seed_all(args.seed+500009*step)
        for features, q in batches:
            f, q = features.to(device), q.to(device)
            y, decoded, _ = model.forward_channel_batches(f, f[..., :3], F.one_hot(q, len(model.cfg.rates)).to(f), 10., 'awgn')
            latent, _ = latent_reconstruction_loss(y, decoded, q, len(model.cfg.rates))
            pred = model.decode_representation(decoded, q, delivered_positions(f[..., :3], q, model.cfg))
            raw = to_raw(pred[q > 0], scene.geometry, model)
            rows.append(raw); auxiliary += float(latent)*len(raw); count += len(raw)
        raw = torch.cat(rows); values = []
        for index in indices:
            camera = scene.cameras[index]
            image = render(raw, camera, scene.degree, scene.white_background)
            loss, terms = objective(image, scene.reference.get(camera, device))
            values.append(dict(loss=float(loss), **{k: float(v) for k, v in terms.items()}))
    means = {k: sum(r[k] for r in values)/len(values) for k in values[0]}
    means['latent_mse'] = auxiliary/count
    means['loss'] += means['latent_mse']
    return means


def audit_joint_step(model, optimizer, scene, step, phase_step, args, *, instrumented):
    device = next(model.parameters()).device
    model.train(); model.semantic_encoder.eval(); optimizer.zero_grad(set_to_none=True)
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    decomposition = ParameterDecomposition(model) if instrumented else None
    task_ref = {}
    def factory(*values):
        batches, _, _ = joint_inputs(scene, step, phase_step, args)
        source_ids = torch.cat([ids[q > 0] for ids, (_, q) in zip(scene.group_ids, batches)])
        task = GradientRenderTask(*values, source_ids=source_ids, top_k=args.top_k)
        decomposition.task = task; task_ref['task'] = task
        return task
    try:
        loss, stats = joint_update(model, scene, step, phase_step, args,
            task_factory=factory if instrumented else None, replay_observer=decomposition)
        norm, gradient = clip_codec_gradients(model, mode='none')
        record = dict(step=step, phase_step=phase_step, layout=stats['layout'],
            training_view_indices=stats['training_view_indices'], grad_norm=float(norm), **gradient,
            before=dict(loss=float(loss), image_mse=stats['image_mse'],
                image_ssim_loss=stats['image_ssim_loss'], latent_mse=stats['latent_mse']),
            scene_gradient_norms=stats['scene_gradient_norms'])
        if instrumented:
            task = task_ref['task']
            record.update(decomposition=decomposition.report(), raw_render_gradient=task.raw_profile,
                          raw_component_gradients=task.raw_components, image_gradient_norms=task.image_gradient_norms)
        optimizer.step()
        record['updates'] = update_stats(model, snapshot)
        try:
            record['after'] = measure_same_input(model, scene, step, phase_step, args)
            record['same_input_loss_change'] = record['after']['loss']-record['before']['loss']
        except (RuntimeError, ValueError) as exc:
            # Keep the pre-update explosion evidence even if this deliberate
            # unprotected diagnostic step makes subsequent rendering invalid.
            record['after'] = None; record['post_update_error'] = str(exc)
        return record
    finally:
        if decomposition is not None: decomposition.close()
