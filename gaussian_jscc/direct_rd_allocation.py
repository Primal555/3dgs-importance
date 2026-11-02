"""Frozen spatial benefit prediction with directly audited deployment changes."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .allocation_distortion import ImageDistortion
from .allocation_mirror import BlockVisitScheduler, ViewWindowScheduler
from .batched_spatial_window import BatchedSpatialWindow
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_state import repair, upsert_jsonl
from .calibrated_spatial_benefit import DIRECTION_OBJECTIVE
from .calibrated_spatial_benefit_allocation import FAMILY as BENEFIT_FAMILY, batched_preflight
from .cli import device_for, seed_all
from .data import read_ply
from .direct_rd_decisions import HardTierMap, ProposalBank, audit_proposals
from .model_factory import build_codec
from .modular_checkpoint import capture_rng, restore_rng
from .modular_experiment import json_write
from .optimization import preserved_rng
from .render_objective import spaced_indices
from .response_value_window import ResponseScheduler
from .spatial_benefit import SpatialBenefitNetwork, SpatialLearner
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY = 'frozen_benefit_direct_rd_v1'


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FrozenBenefit(SpatialLearner):
    def __init__(self, inference, device):
        self.device = torch.device(device)
        self.network = SpatialBenefitNetwork(inference['hidden'], inference['dropout']).to(self.device)
        self.network.load_state_dict(inference['network'])
        self.network.eval().requires_grad_(False)
        self.scales = inference['scales']; self.updates = inference['updates']
        if self.scales is None: raise ValueError('benefit checkpoint has no fitted scales')

    def train_updates(self, *args, **kwargs):
        raise RuntimeError('direct RD experiment freezes the benefit network')


class NoReplay:
    validation_groups = 0
    validation = ()
    validation_frozen = True

    def add(self, *args, **kwargs):
        raise RuntimeError('direct RD inference must not write replay')


class FrozenCandidateScheduler(ResponseScheduler):
    def state_dict(self):
        return dict(coverage=BlockVisitScheduler.state_dict(self), rng=self.generator.get_state())

    def load_state_dict(self, state):
        BlockVisitScheduler.load_state_dict(self, state['coverage'])
        self.generator.set_state(state['rng'].cpu())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'benefit-run', 'scene', 'out'): p.add_argument('--'+name, required=True)
    p.add_argument('--benefit-checkpoint', help='Explicit state.pt; default final, or latest for a pretrain-only run')
    p.add_argument('--manifest'); p.add_argument('--device', default='cuda')
    p.add_argument('--beta', type=float, default=.1)
    for name, default in (('allocation-steps', 2000), ('allocation-batch', 128), ('audit-every', 10),
                          ('accept-trials', 2), ('max-backtracks', 2), ('calibration-views', 4),
                          ('window-views', 4), ('test-views', 32), ('trials', 4), ('image-views', 2),
                          ('save-every', 10), ('cpu-threads', 4), ('seed', 42)):
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--min-rd-gain', type=float, default=1e-6)
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--dry-run', action='store_true')
    return p


def validate(args):
    for name in ('allocation_steps', 'allocation_batch', 'audit_every', 'accept_trials', 'calibration_views',
                 'window_views', 'trials', 'image_views', 'save_every', 'cpu_threads'):
        if getattr(args, name) < 1: raise ValueError('invalid '+name)
    if args.max_backtracks < 0 or args.test_views < 0 or args.snr != 10.:
        raise ValueError('invalid backtracking/views/SNR')
    if any(not math.isfinite(v) or v < 0 for v in (args.beta, args.min_rd_gain)):
        raise ValueError('invalid beta or RD threshold')
    run, benefit, out = (Path(getattr(args, k)).resolve() for k in ('run', 'benefit_run', 'out'))
    if any(out.is_relative_to(src) or src.is_relative_to(out) for src in (run, benefit)):
        raise ValueError('source and output must be separate')


def source_benefit(args, codec_state, source_identity):
    root = Path(args.benefit_run).resolve()
    if args.benefit_checkpoint:
        path = Path(args.benefit_checkpoint).resolve()
    else:
        path = root/'checkpoints/final/state.pt'
        if not path.is_file(): path = root/'checkpoints/latest/state.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('family') != BENEFIT_FAMILY or state['plan']['source'] != source_identity
            or state['learner'].get('direction_objective') != DIRECTION_OBJECTIVE
            or state['learner']['scales'] is None
            or tensor_digest(state['state_dict']) != tensor_digest(codec_state)):
        raise ValueError('benefit source/codec/objective mismatch or unfitted network')
    settings = state['plan']['settings']
    inference = dict(network=state['learner']['network'], scales=state['learner']['scales'],
        updates=state['learner']['updates'], hidden=settings['hidden'], dropout=settings['dropout'])
    descriptor = dict(checkpoint=str(path), step=state['progress']['step'],
                      network_updates=inference['updates'], network_sha256=tensor_digest(inference['network']))
    return dict(inference=inference, descriptor=descriptor, window_settings=settings,
                reference=state['objective_reference'], scene_record=state['scene_record'])


def refresh_side(reference, cost, q, bits):
    kept = int((q > 0).sum())
    reference['side_proxy'] = dict(
        xyz_per_retained=cost['coordinate_channel_uses']/kept if kept else
            reference['q3_cost']['coordinate_channel_uses']/len(q),
        scale_per_retained=cost['scale_content_bits']/bits/kept if kept else 0.,
        fixed_uses=cost['metadata_channel_uses']+cost['scale_framing_bytes']*8/bits, bits_per_use=bits)


def main(args, max_updates=None):
    validate(args)
    saved, cfg, spec, old, options, source = load_inputs(args)
    if tuple(cfg.rates) != (0, 8, 24, 48): raise ValueError('requires 0/8/24/48')
    out = Path(args.out).resolve(); latest = out/'checkpoints/latest/state.pt'
    if not args.dry_run and out.exists() and not args.resume: raise FileExistsError('use NEW output or --resume')
    state = torch.load(latest, map_location='cpu', weights_only=True) if args.resume else None
    settings = {k: v for k, v in vars(args).items() if k not in
                ('out', 'resume', 'dry_run', 'device', 'allocation_batch', 'allocation_image_objective')}
    plan = json.loads(json.dumps(dict(version=FAMILY, source=source, spec=spec, settings=settings)))
    if state is not None and (state.get('family') != FAMILY or state['plan'] != plan):
        raise ValueError('direct RD resume source/settings mismatch')
    frozen = state['frozen'] if state is not None else source_benefit(args, saved['state_dict'], source)
    if args.dry_run:
        result = dict(plan, frozen_benefit=frozen['descriptor'], writes_output=False)
        print(json.dumps(result, indent=2)); return result
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.])
    codec_id = model_id(model)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, options, out, old=old); del raw
    model.set_scene_geometry(scene.geometry)
    if any(frozen['scene_record'].get(k) != v for k, v in scene.record.items()):
        raise ValueError('benefit scene geometry/grouping/camera mismatch')
    chosen = set(spaced_indices(len(scene.cameras), args.calibration_views))
    calibration_views = [c for i, c in enumerate(scene.cameras) if i in chosen]
    training_views = [c for i, c in enumerate(scene.cameras) if i not in chosen]
    scene_record = dict(scene.record, calibration_view_names=[str(c.image_name) for c in calibration_views],
                        allocation_train_view_names=[str(c.image_name) for c in training_views])
    if state is not None and (state['scene_record'] != scene_record or state['codec_id'] != codec_id):
        raise ValueError('direct RD resume scene/codec mismatch')
    learner = FrozenBenefit(frozen['inference'], device)
    reference = copy.deepcopy(state['reference'] if state is not None else frozen['reference'])
    args.allocation_image_objective = ImageDistortion('mse_ssim' if reference['lambda_s'] else 'mse', reference['lambda_s'])
    window_args = SimpleNamespace(**dict(frozen['window_settings'], allocation_batch=args.allocation_batch,
        label_groups=0, beta=args.beta, snr=args.snr, cpu_threads=args.cpu_threads,
        renderer_backend=device.type))
    q = state['q'].long() if state is not None else torch.full((len(scene.raw),), 2, dtype=torch.long)
    table = HardTierMap(q, device)
    scheduler = FrozenCandidateScheduler(scene.group_ids, seed=args.seed)
    views = ViewWindowScheduler(scene_record['allocation_train_view_names'], args.seed+9001)
    bank = ProposalBank(len(q)); meter = ExactDeploymentCost(model, scene, codec_id, args.snr, options.net_bits_per_use)
    window = BatchedSpatialWindow(model, scene, table, scheduler, reference, window_args)
    block_ids = torch.empty(len(q), dtype=torch.long); offset = 0
    for ids in scene.group_ids:
        rows, cols = (ids >= 0).nonzero(as_tuple=True)
        block_ids[ids[rows, cols].cpu()] = rows.cpu()+offset; offset += len(ids)
    progress = dict(step=0, initial_done=False, final_done=False, accepted_batches=0, accepted_changes=0)
    baseline = None
    if state is not None:
        progress = state['progress']; baseline = state['calibration']
        scheduler.load_state_dict(state['scheduler']); views.load_state_dict(state['views']); bank.load_state_dict(state['bank'])
        repair(out, progress['step']); restore_rng(state['rng'])
    out.mkdir(parents=True, exist_ok=True)
    json_write(out/'plan.json', plan)
    json_write(out/'training.json', dict(plan, **scene_record, objective_reference=reference,
        benefit_source=frozen['descriptor'], initial_map='uniform_q2', deployment='direct_integer_map',
        codec_frozen=True, benefit_frozen=True))
    with preserved_rng(device):
        preflight = batched_preflight(model, scene, training_views[0], window_args, reference)
    json_write(out/'preflight.json', preflight)

    def measure(tiers, label, cameras, *, images=False, acceptance=False):
        options_eval = copy.copy(args)
        if acceptance: options_eval.trials = args.accept_trials; options_eval.lpips = False
        result = evaluate_map(model, scene, tiers, cameras, options_eval, save_images=images,
            folder=None if acceptance else out/'evaluation'/label, label=label)
        cost = meter.details(tiers)
        return dict(result, **cost, normalized_distortion=result['source_distortion']/reference['d_ref'],
            normalized_cost=cost['total_channel_uses']/reference['c_ref'],
            rd_score=result['source_distortion']/reference['d_ref']+args.beta*cost['total_channel_uses']/reference['c_ref'],
            beta=args.beta, d_ref=reference['d_ref'], c_ref=reference['c_ref'])

    def record(label, row):
        upsert_jsonl(out/'metrics/validation.jsonl', dict(row, step=progress['step'], event=label))

    def save():
        checkpoint = dict(family=FAMILY, plan=plan, frozen=frozen, codec_id=codec_id, scene_record=scene_record,
            reference=reference, q=q.to(torch.uint8), progress=dict(progress), calibration=baseline,
            bank=bank.state_dict(), scheduler=scheduler.state_dict(), views=views.state_dict(), rng=capture_rng())
        latest.parent.mkdir(parents=True, exist_ok=True)
        temporary = latest.with_suffix('.pt.tmp'); torch.save(checkpoint, temporary); temporary.replace(latest)

    if not progress['initial_done']:
        initial = measure(q, 'initial', scene.test_cameras, images=True)
        record('initial', initial); record('uniform_q2', initial)
        baseline = measure(q, 'acceptance', calibration_views, acceptance=True)
        refresh_side(reference, baseline, q, options.net_bits_per_use)
        progress['initial_done'] = True; save()
    updates = 0
    while progress['step'] < args.allocation_steps:
        step = progress['step']+1; seed_all(args.seed+3000003+step)
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        indices = views.next(args.window_views)
        result = window.collect(step, [training_views[i] for i in indices], learner, NoReplay(), False)
        if not torch.equal(result['q'].cpu(), q): raise RuntimeError('candidate background differs from deployed map')
        bank.add(result['tier_ids'], result['tier_costs']); progress['step'] = step
        audit_seconds = 0.
        if step % args.audit_every == 0 or step == args.allocation_steps:
            audit_start = time.perf_counter(); proposal = bank.propose(q, block_ids)
            before_rd = baseline['rd_score']
            audited = audit_proposals(q, proposal, baseline,
                lambda candidate: measure(candidate, 'acceptance', calibration_views, acceptance=True),
                max_backtracks=args.max_backtracks, min_gain=args.min_rd_gain)
            q, baseline = audited['q'], audited['measurement']
            if audited['accepted']:
                progress['accepted_batches'] += 1; progress['accepted_changes'] += audited['changed']
                table.set_map(q); refresh_side(reference, baseline, q, options.net_bits_per_use)
            bank.clear(); audit_seconds = time.perf_counter()-audit_start
            audit_row = dict(step=step, event='audit', before_rd=before_rd, after_rd=baseline['rd_score'],
                accepted=audited['accepted'], changed=audited['changed'], proposed=len(proposal['ids']),
                attempts=audited['attempts'], total_channel_uses=baseline['total_channel_uses'],
                source_psnr=baseline['source_psnr'], tier_counts=baseline['tier_counts'], audit_seconds=audit_seconds)
            upsert_jsonl(out/'metrics/audits.jsonl', audit_row)
            print(f"audit {step}: accepted={audited['accepted']}, changed={audited['changed']}, "
                  f"RD={before_rd:.7f}->{baseline['rd_score']:.7f}", flush=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        row = dict(result['metrics'], step=step, event='update', phase='direct_allocation',
            step_seconds=time.perf_counter()-started, audit_seconds=audit_seconds, allocation_batch=args.allocation_batch,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
            benefit_updates=learner.updates, accepted_batches=progress['accepted_batches'],
            tier_counts=torch.bincount(q, minlength=4).tolist())
        upsert_jsonl(out/'metrics/loss.jsonl', row)
        if step % args.save_every == 0 or step % args.audit_every == 0 or step == args.allocation_steps: save()
        if step == 1 or step % 10 == 0:
            print(f"direct {step}/{args.allocation_steps}: evaluated={row.get('evaluated_groups')}, "
                  f"tiers={row['tier_counts']}, sec={row['step_seconds']:.2f}, "
                  f"benefit_updates={learner.updates} (frozen)", flush=True)
        updates += 1
        if max_updates is not None and updates >= max_updates:
            save(); return dict(status='interrupted', step=step)
    final = measure(q, 'latest', scene.test_cameras, images=True); record('latest', final)
    record('uniform_q3', measure(torch.full_like(q, 3), 'uniform_q3', scene.test_cameras))
    frozen_codec = model_id(model) == codec_id
    frozen_benefit = tensor_digest(learner.network.state_dict()) == frozen['descriptor']['network_sha256']
    if not frozen_codec or not frozen_benefit: raise RuntimeError('frozen network changed')
    deployment = out/'deployment'; deployment.mkdir(exist_ok=True)
    np.save(deployment/'latest_tiers.npy', q.numpy().astype(np.uint8), allow_pickle=False)
    progress['final_done'] = True; save()
    completion = dict(status='complete', family=FAMILY, **progress, tier_sha256=tier_hash(q),
        frozen_codec_verified=frozen_codec, frozen_benefit_verified=frozen_benefit,
        benefit_source=frozen['descriptor'], calibration_rd=baseline['rd_score'])
    json_write(out/'complete.json', completion)
    from .direct_rd_artifacts import summarize
    summarize(out)
    print(json.dumps(completion), flush=True)
    return completion


if __name__ == '__main__':
    main(parser().parse_args())
