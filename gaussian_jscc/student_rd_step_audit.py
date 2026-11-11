"""Independent paired one-step probes of a saved student RD optimizer."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import torch
from . import student_policy_experiment as entry
from .student_policy_state import FAMILY, cpu_copy
from .student_policy_model import StudentConfig, StudentPolicy, hard_tiers, teacher_loss
from .student_policy_soft import load_soft_teacher, soft_teacher_loss
from .student_policy_inputs import load_teacher, split_student_cameras
from .student_policy_features import digest, VERSION as FEATURE_VERSION
from .student_policy_graph import graph_to
from .student_policy_feedback import CurrentPolicy, UniformFeedback, policy_surrogate, actual_tier_keep_cost
from .student_rd_audit_tools import (split_action_maps, transition_stats, take_step, paired_evaluate,
                                     discrete_prediction)
from .modular_checkpoint import capture_rng, restore_rng
from .optimization import preserved_rng
from .modular_experiment import json_write
from .transport import model_id
from .budgeted_allocation_evaluation import ExactDeploymentCost, tier_hash


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p.add_argument('--checkpoint', choices=('latest', 'distilled', 'best'), default='latest')
    p.add_argument('--probes', type=int, default=3); p.add_argument('--small-step', type=float, default=.1)
    p.add_argument('--device', default='cuda'); p.add_argument('--cpu-threads', type=int)
    return p


def cost_proxy(q, rates, side):
    return float(torch.tensor(rates)[q].sum())+int((q > 0).sum())*(side['xyz_per_retained']+
        side.get('scale_per_retained', 0.))+side['fixed_uses']


def make_proxy(cost, q, bits, reference):
    kept = int((q > 0).sum())
    return dict(xyz_per_retained=cost['coordinate_channel_uses']/kept if kept else
            reference['q3_cost']['coordinate_channel_uses']/len(q),
        scale_per_retained=cost['scale_content_bits']/bits/kept if kept else 0.,
        fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/bits,
        bits_per_use=bits)


def surrogate_terms(heads, keep_costs, ids, costs, inverse):
    keep, tier = heads; indices = ids.to(tier.device)
    signal = (keep_costs[:, 1]-keep_costs[:, 0]).detach().to(keep)
    a = (keep.softmax(-1)[:, 1]*signal).sum()
    b = (tier[indices].softmax(-1)*costs.detach().to(tier)*inverse.to(tier)[:, None]).sum() if len(ids) else tier.sum()*0
    expected, _ = policy_surrogate(keep, tier, keep_costs, ids, costs, inverse)
    torch.testing.assert_close(a+b, expected, rtol=0, atol=0)
    return a, b


@torch.no_grad()
def diagnostic_surrogate(heads, keep_costs, ids, costs, inverse):
    return float(sum(surrogate_terms(tuple(h.detach().cpu().double() for h in heads),
        keep_costs.detach().cpu().double(), ids.cpu(), costs.detach().cpu().double(), inverse.cpu().double())))


def diagnosis(rows):
    def delta(row, name, group):
        return row['maps'][name]['evaluation'][group]['rd']-row['maps']['baseline']['evaluation'][group]['rd']
    return dict(probes=len(rows),
        same_window_rd_worsened=sum(delta(r, 'full', 'training') > 1e-7 for r in rows),
        calibration_rd_worsened=sum(delta(r, 'full', 'calibration') > 1e-7 for r in rows),
        training_improved_calibration_worsened=sum(delta(r, 'full', 'training') < -1e-7 and
            delta(r, 'full', 'calibration') > 1e-7 for r in rows),
        surrogate_improved_hard_rd_worsened=sum(r['soft_surrogate_change']['full'] < -1e-10 and
            delta(r, 'full', 'training') > 1e-7 for r in rows),
        smaller_step_better_training_rd=sum(delta(r, 'small_step', 'training') < delta(r, 'full', 'training')-1e-7 for r in rows),
        unseen_positive_only_worsened=sum(delta(r, 'positive_unseen', 'training') > 1e-7 for r in rows),
        restored_anchor_better_training_rd=sum('anchor_restored' in r['maps'] and
            delta(r, 'anchor_restored', 'training') < delta(r, 'full', 'training')-1e-7 for r in rows),
        interpretation='paired independent one-step evidence from one checkpoint; no cumulative training or test-view assessment')


def main(cli):
    source, out = Path(cli.source).resolve(), Path(cli.out).resolve()
    if cli.probes < 1 or not 0 < cli.small_step < 1 or (cli.cpu_threads is not None and cli.cpu_threads < 1):
        raise ValueError('invalid probe budget')
    if out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('audit output must be separate from source')
    if out.exists(): raise FileExistsError('use a new audit output')
    checkpoint_path = source/'checkpoints'/cli.checkpoint/'student.pt'
    blob = checkpoint_path.read_bytes()
    state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    if state.get('family') != FAMILY or state['progress']['phase'] != 'rd':
        raise ValueError('use a student checkpoint after distillation, with its RD Adam state')
    plan = state['plan']; settings = plan['settings']; reference = plan['reference']
    args = entry.parser().parse_args(['--run', settings['run'], '--scene', settings['scene'], '--out', str(out)])
    vars(args).update(settings); args.device = cli.device; args.out = str(out); args.resume = False
    if cli.cpu_threads is not None: args.cpu_threads = cli.cpu_threads
    args.lpips = False
    entry.validate(args)
    device = entry.device_for(args.device); torch.set_num_threads(args.cpu_threads)
    saved, cfg, spec, old, options, source_info = entry.load_student_source(args)
    model = entry.build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.]); codec = model_id(model)
    if codec != plan['codec_id']: raise ValueError('source codec differs from student checkpoint')
    raw, degree = entry.read_ply(spec['ply'])
    scene = entry.make_scene(spec, raw, degree, model, options, out, old=old); del raw
    model.set_scene_geometry(scene.geometry)
    if scene.record != plan['scene'] or source_info != plan['source']:
        raise ValueError('source scene/cameras/codec provenance changed')
    teacher_q, loaded_ref, teacher = load_teacher(args.teacher_run, scene, codec, args.beta, cfg.to_dict(),
        seed=args.seed, snr=args.snr, trials=args.trials)
    teacher_soft = None
    if args.distill_target == 'soft_deployment':
        teacher_soft, soft_record = load_soft_teacher(args.teacher_run, teacher_q, out,
            seed=args.seed, required_cap=args.required_teacher_tier_cap)
        teacher['soft_target'] = soft_record; teacher_soft = teacher_soft.to(device)
    if teacher != plan['teacher'] or reference != loaded_ref: raise ValueError('student teacher/reference changed')
    training, calibration, preparation = split_student_cameras(scene, teacher, args.preparation_views)
    feature_path = source/'preparation/features.pt'
    features = torch.load(feature_path, map_location='cpu', weights_only=True)
    expected_protocol = dict(version=FEATURE_VERSION,
        identity=dict(codec_id=codec, scene=scene.record, source=source_info),
        camera_names=[str(c.image_name) for c in preparation], fine_points=args.fine_points,
        coarse_points=args.coarse_points, neighbors=args.neighbors)
    if features['protocol'] != expected_protocol or features['signature'] != digest(expected_protocol) or features['signature'] != plan['feature_signature']:
        raise ValueError('source feature cache identity mismatch')
    student = StudentPolicy(StudentConfig(**state['config'])).to(device)
    graph = graph_to(features['graph'], device); controls = entry.controls_for(args, reference, len(scene.raw), device)
    def forward(): return student(features['attributes'], features['latent'], features['visibility'], graph, controls)
    @torch.no_grad()
    def heads():
        student.eval(); return tuple(v.detach().cpu() for v in forward())
    def imitation(h): return soft_teacher_loss(*h, teacher_soft)[0] if teacher_soft is not None else teacher_loss(*h, teacher_q)[0]
    sampler = UniformFeedback(scene.group_ids, args.candidate_rounds, args.finite_samples, args.seed+1103)
    sampler.load_state_dict(state['sampler'])
    views = entry.ViewWindowScheduler(plan['train_view_names'], args.seed+9001); views.load_state_dict(state['views'])
    meter = ExactDeploymentCost(model, scene, codec, args.snr, options.net_bits_per_use)
    distortion = entry.ImageDistortion('mse_ssim', reference['lambda_s'])
    execution = entry.FastFirstOrderExecution(args.candidate_batch); execution.verify(model, scene, args.snr, out)
    student.load_state_dict(state['student']); before = heads(); base_q = hard_tiers(*before)
    original_id = model_id(student)
    step = state['progress']['rd_step']+1
    side = copy.deepcopy(state['progress'].get('side_proxy'))
    if side is None or 'control' in state['progress'].get('pending', []):
        side = make_proxy(meter.details(base_q), base_q, options.net_bits_per_use, reference)
    weight = entry.teacher_weight_at(args, step)
    audit_plan = dict(version='student_rd_paired_step_audit_v1', source=str(source),
        source_checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
        source_step=state['progress']['step'], source_rd_step=state['progress']['rd_step'],
        probes=cli.probes, small_step_fraction=cli.small_step, settings=settings,
        reference=reference, teacher_weight=weight, initial_teacher_weight=args.teacher_weight,
        feature_signature=features['signature'], codec_id=codec, source_model_hash=original_id,
        design='same saved weights/Adam per probe; rotating training windows, paired AWGN per map; calibration evaluation only')
    json_write(out/'plan.json', audit_plan)
    print(f'Pinned RD checkpoint: {checkpoint_path}; RD step={step-1}; teacher weight={weight:.5g}; probes={cli.probes}', flush=True)
    rows = []; next_rng = state['rng']
    for probe in range(1, cli.probes+1):
        started = time.perf_counter()
        student.load_state_dict(state['student']); student.eval()
        optimizer = entry.optimizer_for(student, args, 'rd'); optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        indices = views.next(args.window_views); selected_views = [training[i] for i in indices]
        cameras = dict(training=selected_views, calibration=calibration)
        restore_rng(next_rng); paired_rng = capture_rng()
        task = entry.ComponentRenderTask(selected_views, scene.reference, scene.degree,
            scene.white_background, distortion=distortion)
        adapter = CurrentPolicy(*[x.to(device) for x in before])
        feedback = entry.budgeted_mask_step(model, adapter, scene.groups, scene.group_ids, scene.geometry,
            task, sampler, args.snr, 'awgn', d_ref=reference['d_ref'], price=args.beta,
            rate_normalizer=reference['c_ref'], side_proxy=dict(side, step=step), timings=None, execution=execution)
        next_rng = capture_rng()
        if not torch.equal(feedback['q'], base_q): raise RuntimeError('probe baseline hard policy mismatch')
        keep_costs = actual_tier_keep_cost(feedback, adapter, cfg.rates, args.beta, reference['c_ref'])
        ids, costs = feedback['tier_ids'], feedback['tier_costs']; inverse = sampler.inverse_inclusion[ids].clone()
        student.train(); h = forward(); a, b = surrogate_terms(h, keep_costs, ids, costs, inverse)
        aux = imitation(h); initial_surrogate = diagnostic_surrogate(before, keep_costs, ids, costs, inverse)
        update = take_step(student, optimizer, dict(keep=a, positive=b, teacher=weight*aux), args.grad_clip)
        del h, a, b, aux
        after = heads(); full_model_hash = model_id(student); full_state = cpu_copy(student.state_dict())
        maps = split_action_maps(before, after, ids)
        predicted = dict(full=discrete_prediction(base_q, maps['full'], keep_costs, ids, costs, inverse))
        with torch.no_grad():
            full_surrogate = diagnostic_surrogate(after, keep_costs, ids, costs, inverse)
            for key, value in full_state.items():
                origin = state['student'][key]
                value.copy_(origin+cli.small_step*(value-origin) if value.is_floating_point() else origin)
            student.load_state_dict(full_state)
        smaller = heads(); maps['small_step'] = hard_tiers(*smaller)
        predicted['small_step'] = discrete_prediction(base_q, maps['small_step'], keep_costs, ids, costs, inverse)
        small_surrogate = diagnostic_surrogate(smaller, keep_costs, ids, costs, inverse)
        anchor = None
        if weight < args.teacher_weight:
            student.load_state_dict(state['student']); student.train()
            optimizer = entry.optimizer_for(student, args, 'rd'); optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            h = forward(); a, b = surrogate_terms(h, keep_costs, ids, costs, inverse); aux = imitation(h)
            anchor = take_step(student, optimizer, dict(keep=a, positive=b, teacher=args.teacher_weight*aux), args.grad_clip)
            del h, a, b, aux
            maps['anchor_restored'] = hard_tiers(*heads())
        cache = {}; results = {}
        for label, q in maps.items():
            key = tier_hash(q)
            if key not in cache:
                cache[key] = paired_evaluate(model, scene, q, cameras, args.snr, paired_rng, distortion, meter, reference)
            result = copy.deepcopy(cache[key]); result['tier_sha256'] = key
            result['tier_counts'] = torch.bincount(q, minlength=4).tolist()
            result['cost_proxy'] = cost_proxy(q, cfg.rates, side)
            result['proxy_cost_error'] = result['cost_proxy']-result['cost']['total_channel_uses']
            results[label] = result
            if label == 'baseline':
                error = abs(result['evaluation']['training']['distortion']-feedback['source_distortion'])
                if error > max(1e-8, abs(feedback['source_distortion'])*1e-5):
                    raise RuntimeError(f'paired baseline does not match gradient collection: {error}')
            base = results['baseline']
            result['delta_vs_baseline'] = {group: dict(
                rd=value['rd']-base['evaluation'][group]['rd'],
                distortion_term=value['normalized_distortion']-base['evaluation'][group]['normalized_distortion'],
                cost_term=reference['beta']*(value['normalized_cost']-base['evaluation'][group]['normalized_cost']),
                source_psnr=value['source_psnr']-base['evaluation'][group]['source_psnr'])
                for group, value in result['evaluation'].items()}
            print(f'probe {probe}/{cli.probes} {label}: train RD={result["evaluation"]["training"]["rd"]:.7g}, '
                f'cal RD={result["evaluation"]["calibration"]["rd"]:.7g}, cost={result["cost"]["total_channel_uses"]/1e6:.5f}M', flush=True)
        interactions = {}
        for group in cameras:
            rd = {k: v['evaluation'][group]['rd'] for k, v in results.items()}
            interactions[group] = dict(keep_positive_nonadditivity=rd['full']-rd['keep_only']-rd['positive_only']+rd['baseline'],
                evaluated_unseen_nonadditivity=rd['positive_only']-rd['positive_evaluated']-rd['positive_unseen']+rd['baseline'])
        record = dict(probe=probe, source_weights_reloaded=True, source_optimizer_reloaded=True,
            source_model_hash=original_id, full_model_hash=full_model_hash, next_rd_step=step, teacher_weight=weight,
            training_views=[str(c.image_name) for c in selected_views], evaluated_positive_points=len(ids),
            update=update, anchor_restored_update=anchor,
            transitions=transition_stats(before, after, ids), small_step_transitions=transition_stats(before, smaller, ids),
            soft_surrogate_change=dict(full=full_surrogate-initial_surrogate, small_step=small_surrogate-initial_surrogate),
            predicted_discrete_change=predicted, maps=results, interactions=interactions,
            baseline_gradient_render_error=error, elapsed_seconds=time.perf_counter()-started,
            exact_corrections=feedback['audit'])
        rows.append(record); json_write(out/'probes'/f'{probe:03d}.json', record)
        json_write(out/'review_summary.json', dict(plan=audit_plan, diagnosis=diagnosis(rows), probes=rows))
        del feedback, task, keep_costs, costs, full_state, cache, adapter
    if model_id(model) != codec: raise RuntimeError('frozen codec changed')
    complete = dict(status='complete', probes=cli.probes, source_checkpoint_sha256=audit_plan['checkpoint_sha256'],
        frozen_codec_verified=True, independent_updates=True, diagnosis=diagnosis(rows))
    json_write(out/'complete.json', complete)
    with tarfile.open(out/'review.tar.gz', 'w:gz') as archive:
        for path in (out/'plan.json', out/'complete.json', out/'review_summary.json'):
            archive.add(path, arcname=path.name)
    print(json.dumps(complete), flush=True); return complete


if __name__ == '__main__': main(parser().parse_args())
