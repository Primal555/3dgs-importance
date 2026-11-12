from types import SimpleNamespace
import shutil
import numpy as np
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from test_gaussian_jscc import fixture
from gaussian_jscc.data import prepare, to_features
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.model_factory import build_codec
from gaussian_jscc.modular_evaluation import decode_path
from gaussian_jscc.successive_delivery_packet import export_packet, receive_packet, SIDE_FILES


def scene_model(count=19, block_size=8, blocks_per_batch=2, degree=1):
    torch.set_num_threads(1); torch.manual_seed(8)
    raw, _ = fixture(count, degree)
    cfg = ModularCodecConfig(sh_degree=degree, hidden=16, grid_dim=4, levels=(2,), block_size=block_size,
        channel_variant='successive', rates=(0,8,24,48), channel_hidden=16, context_variant='both',
        channel_power_mode='block', channel_prefix_mode='full_prefix', channel_neighbor_mode='plain',
        channel_neighbor_neighbors=3,
        context_centers=(4, 2), context_neighbors=(3, 3))
    model = build_codec(cfg).eval(); model.set_training_snrs([10.])
    model.channel_decoder.neighbor_enabled.fill_(True)
    order = torch.randperm(len(raw))
    # Packet ordering may differ arbitrarily from original PLY order.
    ordered = raw[order]; _, geom, _ = prepare(raw, cfg.morton_bits)
    f = to_features(ordered, geom, model)[0]
    blocks = list(f.split(block_size)); ids = list(order.split(block_size))
    groups = [pad_sequence(blocks[i:i+blocks_per_batch], batch_first=True) for i in range(0, len(blocks), blocks_per_batch)]
    gids = [pad_sequence(ids[i:i+blocks_per_batch], batch_first=True, padding_value=-1) for i in range(0, len(ids), blocks_per_batch)]
    return model, SimpleNamespace(raw=ordered, order=order, geometry=geom, groups=groups, group_ids=gids)


@pytest.mark.parametrize('noiseless', [False, True])
@pytest.mark.parametrize('all_drop', [False, True])
def test_receiver_packet_matches_original_batched_codec_without_encoders(tmp_path, noiseless, all_drop):
    model, scene = scene_model(); q = torch.arange(len(scene.raw))%4
    if all_drop: q.zero_()
    qs = [torch.where(ids>=0, q[ids.clamp_min(0)], 0) for ids in scene.group_ids]
    torch.manual_seed(42)
    expected = decode_path(model, scene, qs, 'finite_noiseless' if noiseless else 'awgn', 10.)[0] if not all_drop else scene.raw[:0]
    sender = tmp_path/'sender'
    record = export_packet(model, scene, q, 10., 42, sender, noiseless=noiseless)
    assert record['payload_complex_symbols'] == sum(model.cfg.rates[v] for v in q)
    measured = model.scale_rate_record(scene,q[scene.order],2.)
    assert record['scale_stream_bytes']==measured['scale_stream_bytes']
    receiver = tmp_path/'receiver'; shutil.copytree(sender, receiver); shutil.rmtree(sender)
    model.semantic_encoder.forward = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('receiver used encoder'))
    model.channel_encoder.forward = model.semantic_encoder.forward
    del scene
    got, packet_q = receive_packet(model, receiver)
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=1e-5)
    for name in SIDE_FILES:
        path = receiver/name; saved = path.read_bytes(); path.unlink()
        with pytest.raises(FileNotFoundError): receive_packet(model, receiver)
        path.write_bytes(saved)
    scales = receiver/'scales.bin'; scales.write_bytes(scales.read_bytes()+b'X')
    with pytest.raises(ValueError, match='framing'): receive_packet(model, receiver)


def test_server_shaped_batch_and_partial_tail_preserve_xyz_rows(tmp_path):
    from gaussian_jscc.position_delivery import decode_positions, delivered_positions
    model, scene = scene_model(count=64*256+17, block_size=256, blocks_per_batch=64, degree=3)
    assert scene.groups[0].shape == (64,256,59)
    q = torch.arange(len(scene.raw))%4
    packet = tmp_path/'packet'
    export_packet(model,scene,q,10.,42,packet,noiseless=True)
    actual = decode_positions((packet/'xyz.bin').read_bytes(),q[scene.order],model.cfg)
    units = torch.cat([f[..., :3][ids>=0] for f,ids in zip(scene.groups,scene.group_ids)])
    expected = delivered_positions(units,q[scene.order],model.cfg)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    decoded, packet_q = receive_packet(model,packet)
    assert len(decoded)==int((q>0).sum()) and torch.isfinite(decoded).all()
