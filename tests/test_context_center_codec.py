"""Context-informed centroid: same initial function, additional learned inputs."""
import unittest
from unittest.mock import patch
import torch
import test_block_center_codec as centered
from test_block_center_codec import setup as old_setup
import test_q3_noiseless as q3
import test_position_path_diagnosis as diagnosis
from gaussian_jscc.codec import CodecConfig,GaussianCodec


def setup(n=17):
    raw,g,f,old=old_setup(n)
    cfg=old.cfg.to_dict();cfg['xyz_decoder']='context_center'
    m=GaussianCodec(CodecConfig.from_dict(cfg))
    m.attr_mean.copy_(old.attr_mean);m.attr_std.copy_(old.attr_std)
    return raw,g,f,m


class ContextCenterCLI(q3.Q3NoiselessTests):
    xyz_decoder='context_center'


class ContextCenterDiagnosis(diagnosis.PositionPathTests):
    xyz_decoder='context_center'


class ContextCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_initial_function_but_context_centroid_weights_get_gradients(self):
        _,_,f,old=centered.setup(17)
        cfg=old.cfg.to_dict()
        torch.manual_seed(42);a=GaussianCodec(CodecConfig.from_dict(cfg))
        cfg['xyz_decoder']='context_center'
        torch.manual_seed(42);b=GaussianCodec(CodecConfig.from_dict(cfg))
        q=torch.arange(17)%4
        pa=a(f,f[:,:3],q,10,'none');pb=b(f,f[:,:3],q,10,'none')
        torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-6)
        pb[:,:3].sum().backward()
        weights=b.learned.dec_xyz_center[0]
        self.assertEqual(float(weights.weight[:,b.cfg.hidden:].detach().norm()),0.)
        self.assertGreater(float(weights.weight.grad[:,b.cfg.hidden:].norm()),0.)

    def test_centering_and_transport_contracts(self):
        for name in ('test_center_is_received_only_and_context_cannot_translate',
                     'test_transport_gradients_fit_and_all_backward_modes'):
            with self.subTest(name=name),patch('test_block_center_codec.setup',setup):
                getattr(centered.BlockCenterTests(name),name)()


if __name__=='__main__':
    unittest.main()
