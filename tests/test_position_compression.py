import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.position_delivery import (delivered_positions, encode_positions,
    decode_positions, position_cost, PositionCostMeter)
from gaussian_jscc.transport import transmit, receive, save_checkpoint, load_checkpoint, model_id
from gaussian_jscc.data import prepare
from test_learned_joint import setup


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, bits=16):
        return CodecConfig(position_delivery='quantized', position_bits=bits,
                           position_compression='delta_zlib')

    def test_bit_exact_order_duplicates_padding_empty(self):
        xyz=torch.tensor([[1.,0.,.5],[0.,1.,0.],[0.,1.,0.],[.2,.8,.9],[1.,1.,1.],[0.,0.,0.]])
        for bits in (1,12,14,16):
            cfg=self.config(bits)
            for q in (torch.tensor([3,1,2,0,3,1]),torch.zeros(6,dtype=torch.long)):
                data=encode_positions(xyz,q,cfg)
                restored=decode_positions(data,q,cfg)
                torch.testing.assert_close(restored,delivered_positions(xyz,q,cfg),rtol=0,atol=0)
                batched=decode_positions(encode_positions(xyz.reshape(2,3,3),q.reshape(2,3),cfg),q.reshape(2,3),cfg)
                torch.testing.assert_close(batched.reshape(-1,3),restored,rtol=0,atol=0)
                self.assertEqual(position_cost(cfg,int((q>0).sum()),len(data))['position_stream_bytes'],len(data))

    def test_corruption_bounds_and_trailing_streams(self):
        cfg=self.config();q=torch.ones(2,dtype=torch.long);xyz=torch.tensor([[0.,1.,0.],[1.,0.,1.]])
        data=encode_positions(xyz,q,cfg)
        def reframe(payload):
            body=data[8:18]+payload
            return b'GXYZ'+struct.pack('<I',zlib.crc32(body))+body
        for broken in (data[:-1],data+b'x',reframe(data[18:]+b'x'),
                       reframe(zlib.compress(b'\0'*25)),reframe(data[18:-2]),
                       reframe(zlib.compress(b'\xff'*24))):
            with self.assertRaises(ValueError):decode_positions(broken,q,cfg)
        with self.assertRaises(ValueError):decode_positions(data,torch.zeros_like(q),cfg)
        with self.assertRaises(ValueError):decode_positions(data,q,CodecConfig(position_delivery='quantized',position_bits=16))
        with self.assertRaises(ValueError):position_cost(cfg,2)
        with self.assertRaises(ValueError):CodecConfig(position_compression='delta_zlib')

    def test_meter_cache_counts_retention_not_tier(self):
        cfg=self.config();unit=torch.rand(40,3);meter=PositionCostMeter(cfg,unit)
        with patch('gaussian_jscc.position_delivery.encode_positions',wraps=encode_positions) as spy:
            a=meter.stream_bytes(torch.ones(40,dtype=torch.long))
            b=meter.stream_bytes(torch.full((40,),3,dtype=torch.long))
            self.assertEqual(a,b);self.assertEqual(spy.call_count,1)
            q=torch.ones(40,dtype=torch.long);q[0]=0
            meter.stream_bytes(q);self.assertEqual(spy.call_count,2)

    def test_packet_only_receiver_accounting_checkpoint_and_attribute_identity(self):
        raw,_,f,base=setup(17)
        cfg=CodecConfig(**(base.cfg.to_dict()|dict(position_delivery='quantized',position_bits=16,position_compression='delta_zlib')))
        model=GaussianCodec(cfg);model.load_state_dict(base.state_dict())
        plain_cfg=CodecConfig(**(cfg.to_dict()|{'position_compression':'none'}))
        plain=GaussianCodec(plain_cfg);plain.load_state_dict(base.state_dict())
        q=torch.arange(17)%4
        torch.manual_seed(2);pred=model(f,f[:,:3],q,10,'awgn')
        torch.manual_seed(2);expected=plain(f,f[:,:3],q,10,'awgn')
        torch.testing.assert_close(pred,expected,rtol=0,atol=0)
        pred[:,3:].square().mean().backward()
        self.assertIsNone(model.learned.heads['xyz'].weight.grad)
        self.assertGreater(float(model.learned.heads['dc'].weight.grad.norm()),0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stats=transmit(model,raw,q,10,'none',42,root/'packet',.5,4)
            save_checkpoint(root/'codec.pt',model,0);fresh=load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(model_id(model),model_id(fresh))
            restored=receive(fresh,root/'packet')
            ordered,g,oq=prepare(raw,16,q)
            target=g.denormalize(delivered_positions(g.normalize(ordered[:,:3]),oq,cfg)[oq>0])
            torch.testing.assert_close(restored[:,:3],target,rtol=0,atol=0)
            size=(root/'packet/xyz.bin').stat().st_size
            self.assertEqual(stats['position_stream_bytes'],size)
            self.assertEqual(stats['position_channel_uses'],size*4)
            self.assertEqual(stats['total_channel_uses'],stats['payload_complex_symbols']+stats['metadata_channel_uses']+size*4)

    def test_training_and_validation_use_measured_cached_bytes(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.data import write_ply
        from test_render_first import synthetic_render, RenderFirstTests
        raw,_,_,_=setup(16)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--source','mock','--device','cuda','--render-steps','2','--validation-views','1',
                  '--validation-trials','1','--validate-every','1','--hidden','16','--depth','1',
                  '--grid-dim','4','--levels','2','--block-size','8','--decoder-window','4',
                  '--position-delivery','quantized','--position-bits','16','--position-compression','delta_zlib']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=RenderFirstTests().cameras()*2), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),patch('gaussian_jscc.plots.safe_plot'), \
                 patch('gaussian_jscc.position_delivery.encode_positions',wraps=encode_positions) as spy:
                main()
                self.assertEqual(spy.call_count,1)
            model=load_checkpoint(root/'run/codec.pt','cpu')
            ordered,g,q=prepare(raw,16,torch.ones(16,dtype=torch.long))
            size=len(encode_positions(g.normalize(ordered[:,:3]),q,model.cfg))
            rows=[json.loads(x) for x in (root/'run/loss.jsonl').read_text().splitlines()]
            for row in rows:
                self.assertEqual(row['position_stream_bytes'],size)
                self.assertEqual(row['position_channel_uses_estimate'],size*4)
            for row in [json.loads(x) for x in (root/'run/validation.jsonl').read_text().splitlines()]:
                for entry in row['layouts']:
                    self.assertEqual(entry['position_stream_bytes'],size)


if __name__=='__main__':unittest.main()
