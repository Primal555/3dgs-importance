"""Exact reliable-side accounting and fixed-map, RNG-isolated evaluation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .cli import seed_all
from .modular_evaluation import decode_path, conditioning_pair
from .modular_experiment import json_write
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator, image_metric_protocol
from .position_delivery import encode_positions
from .render_objective import image_metrics, spaced_indices
from .render_validation import save_panel
from .transport import encode_metadata, model_id


def tier_hash(q):
    return hashlib.sha256(q.detach().cpu().to(torch.uint8).numpy().tobytes()).hexdigest()


class ExactDeploymentCost:
    def __init__(self, model, scene, codec_id, snr, bits):
        self.model, self.scene, self.codec_id = model, scene, codec_id
        if not math.isfinite(bits) or bits <= 0: raise ValueError('explicit reliable bit/use required')
        self.snr, self.bits = snr, bits
        from .position_delivery import PositionCostMeter
        self.positions = PositionCostMeter(model.cfg, scene.geometry.normalize(scene.raw[:, :3]))

    def details(self, original_q):
        q = original_q.detach().cpu()
        if q.dtype != torch.long or q.shape != (len(self.scene.raw),) or ((q<0)|(q>3)).any():
            raise ValueError('tiers must be original-row int64 [N] in 0..3')
        q = q[self.scene.order]
        size = self.positions.stream_bytes(q)
        if size is None: size = len(encode_positions(self.positions.unit, q, self.model.cfg))
        tx, rx = conditioning_pair(self.model, self.snr)
        header = dict(version=2, config=self.model.cfg.to_dict(), model_id=self.codec_id,
            count=int((q>0).sum()), source_count=len(q), geometry=self.scene.geometry.to_dict(),
            snr_db=self.snr, channel='awgn', tx_condition_snr_db=tx, rx_condition_snr_db=rx)
        meta_bytes = len(encode_metadata(header, q.numpy()))
        payload = int(torch.tensor(self.model.cfg.rates)[q].sum())
        xyz_uses, meta_uses = math.ceil(8*size/self.bits), math.ceil(8*meta_bytes/self.bits)
        scale = self.model.scale_rate_record(self.scene, q, self.bits)
        return dict(payload_complex_symbols=payload, xyz_bytes=size, metadata_bytes=meta_bytes,
            coordinate_channel_uses=xyz_uses, metadata_channel_uses=meta_uses, **scale,
            total_channel_uses=payload+xyz_uses+meta_uses+scale['scale_channel_uses'],
            net_bits_per_use_assumption=self.bits, xyz_cache_hit=self.positions.last_cache_hit,
            rate_scope='analogue payload + retained XYZ + complete metadata + binary16 scale frames; shared weights excluded',
            reliability='reliable-side assumption; no FEC/packet errors simulated')


@torch.no_grad()
def evaluate_map(model, scene, q, cameras, args, *, save_images=False, folder=None, label='map'):
    from .rendering import render
    from .tier_visualization import render_tier_map, save_tier_panel
    if not cameras: raise ValueError('evaluation cameras cannot be empty')
    device = next(model.parameters()).device
    identity = dict(codec_id=model_id(model), tier_sha256=tier_hash(q), label=label,
        views=[str(c.image_name) for c in cameras], trials=args.trials, seed=args.seed,
        snr=args.snr, lpips=args.lpips, images=save_images)
    objective = getattr(args, 'allocation_image_objective', None)
    if objective is not None: identity['image_objective'] = objective.identity()
    folder = Path(folder) if folder is not None else None
    if folder is not None and (folder/'metrics.json').exists():
        cached = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        if cached['identity'] != identity: raise ValueError('saved evaluation identity mismatch')
        return cached['metrics']
    observations = []
    with preserved_rng(device):
        model.eval()
        perceptual = LPIPSEvaluator(device) if args.lpips else None
        qs = [torch.where(ids>=0, q[ids.clamp_min(0)], 0) for ids in scene.group_ids]
        packet_q = q[scene.order]
        images = set(spaced_indices(len(cameras), args.image_views))
        for trial in range(args.trials):
            seed_all(args.seed+1000+trial)
            received = decode_path(model, scene, qs, 'awgn', args.snr)[0] if (q>0).any() else scene.raw[:0].to(device)
            for index, camera in enumerate(cameras):
                image = render(received, camera, scene.degree, scene.white_background)
                source = scene.reference.get(camera, device); photo = camera.original_image[:3].to(device)
                row = dict(trial=trial, view=str(camera.image_name))
                if objective is not None:
                    components = objective.components(image, source)
                    row.update(source_objective_mse=float(components[0]), source_structure_error=float(components[1]),
                               source_distortion=float(objective.combine(components)))
                for prefix, target in (('source', source), ('photo', photo)):
                    row.update({prefix+'_'+k: v for k,v in image_metrics(image, target).items()})
                    row[prefix+'_lpips'] = perceptual(image, target) if perceptual is not None else None
                observations.append(row)
                if save_images and trial == 0 and index in images:
                    if folder is None: raise ValueError('image output folder required')
                    stem = folder/'images'/f'view{index:04d}'
                    save_panel(stem.with_suffix('.png'), photo, source, image, label)
                    colored, labels = render_tier_map(received, packet_q[packet_q>0], camera, scene.degree)
                    save_tier_panel(stem.with_name(stem.name+'_tiers.png'), photo, source, image, colored,
                                    model.cfg.rates, torch.bincount(q, minlength=4).tolist())
                    from PIL import Image
                    Image.fromarray(labels.cpu().numpy().astype(np.uint8)).save(stem.with_name(stem.name+'_tier_labels.png'))
    result = dict(views=len(cameras), trials=args.trials, per_view_trial=observations,
        tier_counts=torch.bincount(q, minlength=4).tolist(), codec_id=identity['codec_id'],
        tier_sha256=identity['tier_sha256'], protocol=image_metric_protocol(args.lpips))
    result['tier_shares'] = [x/len(q) for x in result['tier_counts']]
    for key in ('source_mse','source_psnr','source_ssim','source_lpips','photo_mse','photo_psnr','photo_ssim','photo_lpips'):
        values = [r[key] for r in observations]
        result[key] = float(np.mean(values)) if values[0] is not None else None
    if objective is not None:
        for key in ('source_objective_mse', 'source_structure_error', 'source_distortion'):
            result[key] = float(np.mean([r[key] for r in observations]))
        result['lambda_s'] = objective.lambda_s
    if folder is not None: json_write(folder/'metrics.json', dict(identity=identity, metrics=result))
    return result
