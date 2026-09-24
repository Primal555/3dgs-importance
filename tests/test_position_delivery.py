"""CPU protocol/autograd checks. Synthetic rendering is NOT a fidelity test."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from test_learned_joint import setup
from test_render_first import synthetic_render, Reference
import test_render_first as render_fixture
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import prepare, to_raw, write_ply
from gaussian_jscc.position_delivery import (delivered_positions, encode_positions, decode_positions,
                                            position_cost, training_position_cost)
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, transmit, receive, model_id
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.render_objective import MultiViewRenderTask


def explicit_model(base, mode='quantized', bits=12):
    cfg = CodecConfig(**(base.cfg.to_dict() | {'position_delivery':mode,'position_bits':bits}))
    model = GaussianCodec(cfg)
    model.load_state_dict(base.state_dict())
    return model


class PositionDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_configuration(self):
        for kwargs in ({'position_delivery':'silent_oracle'}, {'position_bits':17}, {'position_bits':0}):
            with self.assertRaises(ValueError):
                CodecConfig(**kwargs)

    def test_same_random_weights_all_modes(self):
        states=[]
        for mode in ('learned','float32','quantized'):
            torch.manual_seed(3)
            states.append(GaussianCodec(CodecConfig(position_delivery=mode)).state_dict())
        for key in states[0]:
            torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
            torch.testing.assert_close(states[0][key],states[2][key],rtol=0,atol=0)

    def test_bitstream_roundtrip_all_precisions_and_empty(self):
        _,_,f,base=setup(17)
        for mode,bits in [('float32',12)]+[('quantized',b) for b in (1,8,10,12,16)]:
            model=explicit_model(base,mode,bits)
            for q in (torch.arange(17)%4,torch.zeros(17,dtype=torch.long)):
                blob=encode_positions(f[:,:3],q,model.cfg)
                decoded=decode_positions(blob,q,model.cfg)
                expected=delivered_positions(f[:,:3],q,model.cfg)
                torch.testing.assert_close(decoded,expected,rtol=0,atol=0)
                self.assertEqual(len(blob),position_cost(model.cfg,int((q>0).sum()))['position_stream_bytes'])
                broken=bytearray(blob); broken[-1]^=1
                with self.assertRaisesRegex(ValueError,'CRC'):
                    decode_positions(bytes(broken),q,model.cfg)
                with self.assertRaises(ValueError):
                    decode_positions(blob[:-1],q,model.cfg)
                if q.any():
                    with self.assertRaisesRegex(ValueError,'count'):
                        decode_positions(blob,torch.zeros_like(q),model.cfg)

    def test_forward_replaces_only_xyz_and_preserves_noise_rng(self):
        _,_,f,base=setup(8)
        q=torch.tensor([0,1,2,3,1,2,0,3])
        for mode in ('float32','quantized'):
            model=explicit_model(base,mode)
            torch.manual_seed(7)
            baseline=base(f,f[:,:3],q,10,'awgn')
            end_state=torch.get_rng_state().clone()
            torch.manual_seed(7)
            pred=model(f,f[:,:3],q,10,'awgn')
            self.assertTrue(torch.equal(torch.get_rng_state(),end_state))
            torch.testing.assert_close(pred[:,3:],baseline[:,3:],rtol=0,atol=0)
            torch.testing.assert_close(pred[:,:3],delivered_positions(f[:,:3],q,model.cfg))
            torch.manual_seed(7)
            batched=model.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float()[None],10,'awgn')[0][0]
            torch.testing.assert_close(pred,batched)
            pred[:,3:].square().mean().backward()
            self.assertIsNone(model.learned.heads['xyz'].weight.grad)
            self.assertGreater(float(model.learned.heads['dc'].weight.grad.norm()),0)
            self.assertGreater(float(model.learned.symbol_head.weight.grad.norm()),0)
            with self.assertRaisesRegex(ValueError,'requires delivered'):
                model.decode(model.encode(f,f[:,:3],q,10),q,10)

    def test_receiver_needs_only_packet_and_checkpoint_and_counts_cost(self):
        raw,_,_,base=setup(17)
        q=torch.arange(17)%4
        for mode in ('float32','quantized'):
            model=explicit_model(base,mode)
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                stats=transmit(model,raw,q,10,'none',42,root/'packet',.5,4)
                save_checkpoint(root/'codec.pt',model,0)
                fresh=load_checkpoint(root/'codec.pt','cpu')
                self.assertEqual(model_id(model),model_id(fresh))
                recovered=receive(fresh,root/'packet')
                ordered,g,oq=prepare(raw,16,q)
                expected=g.denormalize(delivered_positions(g.normalize(ordered[:,:3]),oq,model.cfg)[oq>0])
                torch.testing.assert_close(recovered[:,:3],expected,rtol=0,atol=0)
                stream_size=(root/'packet'/'xyz.bin').stat().st_size
                self.assertEqual(stats['position_stream_bytes'],stream_size)
                self.assertEqual(stats['position_channel_uses'],stream_size*4)
                self.assertEqual(stats['total_channel_uses'],stats['payload_complex_symbols']+
                                 stats['metadata_channel_uses']+stats['position_channel_uses'])
                train_cost=training_position_cost(model.cfg,int((q>0).sum()),len(q),stats['payload_complex_symbols'],2)
                self.assertEqual(train_cost['position_stream_bytes'],stream_size)
                self.assertEqual(train_cost['position_channel_uses_estimate'],stats['position_channel_uses'])
                (root/'packet'/'xyz.bin').unlink()
                with self.assertRaises(FileNotFoundError):
                    receive(fresh,root/'packet')

    def test_replay_matches_checkpoint_and_has_attribute_gradients(self):
        _,g,f,base=setup(16)
        batches=[(f.reshape(2,8,-1),torch.tensor([[1,0,2,3,1,2,3,0],[3,2,1,3,0,2,1,2]]))]
        first=explicit_model(base)
        second=copy.deepcopy(first)
        cameras=render_fixture.RenderFirstTests().cameras()
        losses=[]
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for mode,model in [('replay',first),('checkpoint',second)]:
                torch.manual_seed(6)
                task=MultiViewRenderTask(cameras,Reference(),0)
                loss,_=full_scene_step(model,batches,g,10,'awgn',task,attr_weight=0,mode=mode)
                losses.append(loss)
        torch.testing.assert_close(*losses)
        for (name,p),(_,v) in zip(first.named_parameters(),second.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,v.grad,atol=2e-6,rtol=2e-4,msg=name)
        self.assertGreater(float(first.learned.heads['dc'].weight.grad.norm()),0)
        self.assertIsNone(first.learned.heads['xyz'].weight.grad)

    def test_three_mode_short_training_and_summary_mock_renderer(self):
        from gaussian_jscc.cli import main
        from scripts.summarize_position_delivery import summarize
        raw,_,_,_=setup(16)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            for mode in ('learned','float32','quantized'):
                argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/mode),
                      '--source','mock','--device','cuda','--render-steps','2','--validation-views','1',
                      '--validation-trials','1','--validate-every','1','--hidden','16','--depth','1',
                      '--grid-dim','4','--levels','2','--block-size','8','--decoder-window','4',
                      '--position-delivery',mode]
                with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                     patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()*2), \
                     patch('gaussian_jscc.rendering.render',side_effect=synthetic_render),patch('gaussian_jscc.plots.safe_plot'):
                    main()
                loaded=load_checkpoint(root/mode/'codec.pt','cpu')
                self.assertEqual(loaded.cfg.position_delivery,mode)
                rows=[json.loads(line) for line in (root/mode/'loss.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['aux_loss']==0 for row in rows))
            result=summarize(root)
            self.assertEqual(set(result['runs']),{'learned','float32','quantized'})
            self.assertTrue((root/'validation_comparison.csv').exists())
            self.assertGreater(result['runs']['float32']['last_layouts'][0]['position_stream_bits'],0)


if __name__=='__main__':
    unittest.main()
