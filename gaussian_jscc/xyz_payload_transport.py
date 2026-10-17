"""Self-contained simulated XYZ-payload packet; no reliable coordinate stream."""
import json
import hashlib
import math
import struct
import tempfile
import zlib
from pathlib import Path
import numpy as np
import torch
from .codec import channel,validate_tiers
from .data import Geometry,prepare,to_features,to_raw
from .optimization import preserved_rng
from .transport import encode_metadata,decode_metadata,model_id
from .xyz_payload_config import XYZPayloadConfig


def encode_packet_metadata(header,q,digest=bytes(32)):
    """Fixed 40-byte integrity envelope; checksum bytes are not compressed.

    Outer CRC binds header/framing and expected SHA256 together. SHA256 detects
    corruption of stored simulated received values, NOT real channel errors/FEC.
    """
    if len(digest)!=32: raise ValueError('invalid payload checksum')
    body=digest+encode_metadata(header,q)
    return b'XYP1'+struct.pack('<I',zlib.crc32(body))+body


def decode_packet_metadata(data):
    if len(data)<48 or data[:4]!=b'XYP1': raise ValueError('invalid XYZ packet integrity framing')
    body=data[8:]
    if zlib.crc32(body)!=struct.unpack('<I',data[4:8])[0]: raise ValueError('packet metadata CRC failure')
    header,q=decode_metadata(body[32:])
    return header,q,body[:32]


def packet_header(model,geometry,q,snr,kind,identity=None):
    return dict(experiment='xyz_payload_jscc_v1',version=1,config=model.cfg.to_dict(),
                model_id=identity or model_id(model),geometry=geometry.to_dict(),
                count=int((q>0).sum()),source_count=len(q),snr_db=float(snr),channel=kind,
                xyz_delivery='in_payload',coordinate_bytes=0,ordering='sender_morton')


def rate_record(model,geometry,q,snr,kind,net_bits_per_use=2.,identity=None):
    if not math.isfinite(net_bits_per_use) or net_bits_per_use<=0:
        raise ValueError('net bits/use must be positive finite')
    q=q.detach().cpu()
    header=packet_header(model,geometry,q,snr,kind,identity)
    metadata=encode_packet_metadata(header,q.numpy())
    payload=int(torch.tensor(model.cfg.rates)[q].sum())
    overhead=math.ceil(8*len(metadata)/net_bits_per_use)
    return dict(payload_complex_symbols=payload,symbols_per_gaussian=payload/max(len(q),1),
                xyz_bytes=0,coordinate_channel_uses=0,metadata_bytes=len(metadata),packet_integrity_bytes=40,
                metadata_channel_uses=overhead,total_channel_uses=payload+overhead,
                net_bits_per_use_assumption=float(net_bits_per_use),
                reliability='global metadata assumed reliable; no FEC or packet errors simulated',
                rate_scope='payload includes XYZ+attributes; measured global metadata/tier framing; shared model excluded')


@torch.no_grad()
def export_packet(model,raw,q,snr,kind,seed,out,net_bits_per_use=2.):
    from .cli import seed_all
    validate_tiers(q,len(raw),len(model.cfg.rates))
    if not (q>0).any() or kind not in ('none','awgn') or not math.isfinite(float(snr)):
        raise ValueError('nonempty payload and finite supported channel required')
    if not math.isfinite(net_bits_per_use) or net_bits_per_use<=0: raise ValueError('invalid net bits/use')
    out=Path(out)
    if out.exists(): raise FileExistsError('packet output must be new')
    device=next(model.parameters()).device; was_training=model.training
    raw,geometry,q=prepare(raw,model.cfg.morton_bits,q)
    model.set_scene_geometry(geometry)
    received=[]
    try:
        with preserved_rng(device):
            seed_all(seed); model.eval()
            for start in range(0,len(raw),model.cfg.block_size):
                f,xyz=to_features(raw[start:start+model.cfg.block_size].to(device),geometry,model)
                tiers=q[start:start+len(f)].to(device)
                z=model.encode(f,xyz,tiers,snr)
                received.append(channel(z,snr,kind).cpu())
    finally: model.train(was_training)
    header=packet_header(model,geometry,q,snr,kind)
    values=torch.cat(received).numpy()
    metadata=encode_packet_metadata(header,q.numpy(),hashlib.sha256(values.tobytes()).digest())
    cost=rate_record(model,geometry,q,snr,kind,net_bits_per_use,header['model_id'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.xyz-packet-',dir=out.parent) as temporary:
        staged=Path(temporary)/'packet'; staged.mkdir()
        (staged/'metadata.bin').write_bytes(metadata)
        np.save(staged/'received.npy',values,allow_pickle=False)
        (staged/'cost.json').write_text(json.dumps(cost,indent=2,allow_nan=False),encoding='utf-8')
        staged.rename(out)
    return cost


@torch.no_grad()
def receive_packet(model,folder):
    folder=Path(folder)
    if (folder/'xyz.bin').exists(): raise ValueError('coordinate side stream forbidden')
    header,q,digest=decode_packet_metadata((folder/'metadata.bin').read_bytes())
    if (header.get('experiment')!='xyz_payload_jscc_v1' or header.get('version')!=1
            or header.get('xyz_delivery')!='in_payload' or header.get('coordinate_bytes')!=0):
        raise ValueError('not a source-free XYZ payload packet')
    if XYZPayloadConfig.from_dict(header['config'])!=model.cfg or header['model_id']!=model_id(model):
        raise ValueError('packet model/config mismatch')
    if header['channel'] not in ('none','awgn') or not math.isfinite(float(header['snr_db'])):
        raise ValueError('invalid packet channel')
    expected=int(torch.tensor(model.cfg.rates)[q].sum())
    values=np.load(folder/'received.npy',allow_pickle=False)
    if values.shape!=(expected,2) or values.dtype!=np.float32 or not np.isfinite(values).all():
        raise ValueError('invalid received symbol length/values')
    if hashlib.sha256(values.tobytes()).digest()!=digest:
        raise ValueError('received payload integrity checksum mismatch')
    geometry=Geometry(**header['geometry']); model.set_scene_geometry(geometry)
    device=next(model.parameters()).device; was_training=model.training
    rows=[]; offset=0
    try:
        model.eval()
        for start in range(0,len(q),model.cfg.block_size):
            tiers=q[start:start+model.cfg.block_size].to(device)
            count=int(torch.tensor(model.cfg.rates,device=device)[tiers].sum())
            packet=torch.from_numpy(values[offset:offset+count]).to(device)
            pred=model.decode(packet,tiers,header['snr_db'])
            rows.append(to_raw(pred[tiers>0],geometry,model)); offset+=count
    finally: model.train(was_training)
    if offset!=expected or not rows: raise ValueError('invalid packet framing')
    raw=torch.cat(rows)
    if len(raw)!=header['count'] or not torch.isfinite(raw).all(): raise ValueError('invalid decoded scene')
    return raw,header
