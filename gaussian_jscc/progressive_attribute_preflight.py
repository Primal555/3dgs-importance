"""Small real-device prefix/packet checks before any optimizer updates."""
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .optimization import preserved_rng
from .progressive_attribute_evaluation import decode_cached
from .progressive_attribute_packet import export_packet, receive_packet


@torch.no_grad()
def verify_execution(model, scene, cached, args, out):
    device = next(model.parameters()).device
    model.eval(); model.set_scene_geometry(scene.geometry)
    data = cached[0][:2]
    ids = scene.group_ids[0][:2]
    active = ids >= 0
    q = ((torch.arange(active.numel()).reshape_as(active) % 8)+1)*active
    # Even a tiny fixture must exercise the highest stage.
    q[active] = torch.arange(int(active.sum())) % 8+1
    q[active.nonzero(as_tuple=True)[0][-1], active.nonzero(as_tuple=True)[1][-1]] = 8
    sample = SimpleNamespace(groups=[data[..., :59]], group_ids=[ids], geometry=scene.geometry,
                             raw=scene.raw[:int(active.sum())], name=scene.name)
    y = data[..., 59:].to(device); q_device = q.to(device)
    errors = []
    with preserved_rng(device):
        full, full_scales = model.encode(y, q_device, args.snr)
        for layer in range(1, 9):
            z, scales = model.encode(y, q_device.clamp_max(layer), args.snr)
            torch.testing.assert_close(z, full[..., :16*layer], atol=2e-6, rtol=1e-6)
            torch.testing.assert_close(scales, full_scales[:, :layer], atol=0, rtol=0)
            errors.append(float((z-full[..., :16*layer]).abs().max()))
        with tempfile.TemporaryDirectory(prefix='execution-', dir=out) as directory:
            folder = Path(directory)/'packet'
            _, cost = export_packet(model, sample, [data], [q], args.snr, args.seed+991, folder)
            recovered = receive_packet(model, folder)
            original, _ = decode_cached(model, sample, [data], [q], args.snr, args.seed+991)
            torch.testing.assert_close(recovered, original, atol=2e-5, rtol=1e-5)
    return dict(status='passed', device=str(device), layers=8,
        maximum_prefix_error=max(errors), packet_max_error=float((recovered-original).abs().max()),
        actual_packet_complex_uses=cost['total_channel_uses'],
        scope='same fixed mixed delivery plan; cap received prefix; source-free received-file reconstruction')
