import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_modular_training import setup
from test_modular_evaluation import scene_fixture
from test_render_first import synthetic_render


class ModularSNRTests(unittest.TestCase):
    def test_supported_condition_does_not_change_physical_noise(self):
        m,g,f,q=setup(); m.training_snrs=(10.,)
        choices=torch.nn.functional.one_hot(q,4).float()
        captured=[]
        hook=m.channel_decoder.register_forward_pre_hook(lambda module,args:captured.append((args[0].detach().clone(),args[3])))
        try:
            torch.manual_seed(77)
            _,high,_=m.forward_channel_batches(f,f[...,:3],choices,20,'awgn',paired_noise=True)
            high_signal=captured[-1][0]
            torch.manual_seed(77)
            _,explicit,_=m.forward_channel_batches(f,f[...,:3],choices,20,'awgn',paired_noise=True,
                                                  tx_condition_snr=10,rx_condition_snr=10)
            _,clean,_=m.forward_channel_batches(f,f[...,:3],choices,20,'none',paired_noise=True)
            clean_signal=captured[-1][0]
            torch.manual_seed(77)
            m.forward_channel_batches(f,f[...,:3],choices,0,'awgn',paired_noise=True)
            low_signal=captured[-1][0]
        finally: hook.remove()
        torch.testing.assert_close(high,explicit,rtol=0,atol=0)
        torch.testing.assert_close(high_signal-clean_signal,(low_signal-clean_signal)/10,atol=1e-7,rtol=1e-5)
        self.assertEqual([snr for received,snr in captured],[10.,10.,10.,10.])
        self.assertFalse(torch.equal(high,clean))

    def test_explicit_sender_receiver_conditions_are_independent(self):
        m,g,f,q=setup(); m.training_snrs=(10.,)
        calls=[]
        hooks=[m.channel_encoder.register_forward_pre_hook(lambda module,args:calls.append(('tx',args[2]))),
               m.channel_decoder.register_forward_pre_hook(lambda module,args:calls.append(('rx',args[3])))]
        try:
            m.forward_channel_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),20,'none',
                                      tx_condition_snr=10,rx_condition_snr=20)
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(calls,[('tx',10.),('rx',20.)])

    def test_training_range_and_invalid_values(self):
        m,g,f,q=setup()
        self.assertEqual(m.resolve_condition_snr(20),20.)
        m.training_snrs=(5.,10.,15.)
        self.assertEqual([m.resolve_condition_snr(x) for x in (0,8,20)],[5.,8.,15.])
        self.assertEqual(m.resolve_condition_snr(20,20),20.)
        for actual,condition in [(float('nan'),10),(10,float('inf'))]:
            with self.assertRaises(ValueError): m.resolve_condition_snr(actual,condition)

    def test_evaluation_records_actual_and_both_conditions(self):
        from gaussian_jscc.modular_evaluation import evaluate_paths
        m,g,f,q=setup(); m.training_snrs=(10.,); scene=scene_fixture(m,g,f)
        with tempfile.TemporaryDirectory() as temp,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            fixed=evaluate_paths(m,scene,step=8,phase='test',snrs=[20],trials=1,seed=42,save_images=False,out=Path(temp))
            matched=evaluate_paths(m,scene,step=8,phase='test',snrs=[20],trials=1,seed=42,save_images=False,out=Path(temp),condition_policy='matched')
        for records,condition in [(fixed,10.),(matched,20.)]:
            awgn=[r for r in records if r['path']=='awgn']
            self.assertEqual(len(awgn),4)
            self.assertTrue(all(r['actual_snr']==20 and r['tx_condition_snr']==condition and r['rx_condition_snr']==condition for r in awgn))

    def test_saved_support_and_packet_condition_survive_receiver_load(self):
        from test_gaussian_jscc import fixture
        from gaussian_jscc.transport import save_checkpoint,load_checkpoint,transmit,receive,decode_metadata
        m,g,f,q=setup(); m.training_snrs=(10.,)
        raw,_=fixture(16,1); tiers=torch.full((16,),3,dtype=torch.long)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); ckpt=root/'codec.pt'
            save_checkpoint(ckpt,m,8,training={'progress':{'train_snrs':[10.]}})
            receiver=load_checkpoint(ckpt,'cpu')
            self.assertEqual(receiver.training_snrs,(10.,))
            transmit(m,raw,tiers,20,'none',42,root/'high')
            transmit(m,raw,tiers,10,'none',42,root/'reference')
            header,_=decode_metadata((root/'high/metadata.bin').read_bytes())
            self.assertEqual((header['snr_db'],header['tx_condition_snr_db'],header['rx_condition_snr_db']),(20.,10.,10.))
            receiver.training_snrs=()
            torch.testing.assert_close(receive(receiver,root/'high'),receive(receiver,root/'reference'),rtol=0,atol=0)

    def test_legacy_packet_uses_its_original_condition_with_guarded_receiver(self):
        from test_gaussian_jscc import fixture
        from gaussian_jscc.transport import transmit,receive,decode_metadata,encode_metadata
        m,g,f,q=setup(); raw,_=fixture(16,1); tiers=torch.full((16,),3,dtype=torch.long)
        with tempfile.TemporaryDirectory() as temp:
            packet=Path(temp)/'packet'
            transmit(m,raw,tiers,20,'none',42,packet)
            reference=receive(m,packet)
            header,tiers=decode_metadata((packet/'metadata.bin').read_bytes())
            header.pop('tx_condition_snr_db'); header.pop('rx_condition_snr_db')
            (packet/'metadata.bin').write_bytes(encode_metadata(header,tiers.numpy()))
            m.training_snrs=(10.,)
            torch.testing.assert_close(receive(m,packet),reference,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
