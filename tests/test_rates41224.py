"""4/12/24 protocol, cost-scale and actual Bash launcher checks; CPU only."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from gaussian_jscc.allocation_diagnostics import AllocationCostMeter
from gaussian_jscc.codec import GaussianCodec, pack
from gaussian_jscc.position_delivery import PositionCostMeter
from gaussian_jscc.transport import transmit, receive
from test_progressive_prefix import progressive
import test_training_launcher as launcher


class Rates41224Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        raw,g,f,old=progressive(16)
        model=GaussianCodec(replace(old.cfg,rates=(0,4,12,24)))
        model.attr_mean.copy_(old.attr_mean)
        model.attr_std.copy_(old.attr_std)
        return raw,g,f,old,model

    def test_prefix_lengths_power_and_packet_roundtrip(self):
        raw,_,f,_,model=self.model()
        q=torch.arange(len(raw))%4
        active=q>0
        full=model.encode_full(f,f[:,:3],active,10)
        self.assertEqual(full.shape,(16,48))
        for start,end in zip(model.cfg.rates[:-1],model.cfg.rates[1:]):
            energy=full[active,2*start:2*end].square().sum(-1)/(end-start)
            self.assertTrue((energy<=1+1e-6).all())
        torch.testing.assert_close(model.encode(f,f[:,:3],q,10),pack(full,q,model.cfg.rates),rtol=0,atol=0)
        for tier in (1,2,3):
            self.assertEqual(len(model.encode(f,f[:,:3],active.long()*tier,10)),int(active.sum())*model.cfg.rates[tier])
        with tempfile.TemporaryDirectory() as temp:
            stats=transmit(model,raw,q,10,'none',42,Path(temp)/'packet',code_rate=1,modulation_bits=2)
            actual=receive(model,Path(temp)/'packet')
            self.assertEqual(stats['payload_complex_symbols'],160)
            self.assertEqual(stats['tier_id_bits'],2)
            self.assertEqual(len(actual),12)
            self.assertTrue(torch.isfinite(actual).all())

    def test_penalty_normalizes_by_actual_full24_cost(self):
        raw,g,_,old,new=self.model()
        unit=g.normalize(raw[:,:3])
        before=AllocationCostMeter(old.cfg,PositionCostMeter(old.cfg,unit),len(raw),2.)
        default=AllocationCostMeter(new.cfg,PositionCostMeter(new.cfg,unit),len(raw),2.)
        full=default.details(torch.full((len(raw),),3))
        self.assertEqual(default.normalizer,full['allocation_uses_per_source_gaussian'])
        self.assertAlmostEqual(default.normalizer,24+full['allocation_side_uses_per_source_gaussian'])
        self.assertAlmostEqual(before.normalizer-default.normalizer,8)

    def test_full_launcher_defaults_random_and_four_stages(self):
        result=launcher.LauncherTests().launch({'CUDA_VISIBLE_DEVICES':'2','INIT':'obsolete32.pt','ALLOCATION_INIT':'obsolete.pt'},
                                     script='scripts/train_progressive16_rates41224_full.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for text in ('--rates 0 4 12 24','--bootstrap-steps 5000','--render-steps 5000',
                     '--joint-steps 3000','--mask-only-steps 2000',
                     '--beta 0.01','--joint-validate-every 100','--position-compression-level 6',
                     '--keep-lr 0.01','--mask-adam-eps 1e-15','--render-backward replay',
                     'codec_end_allocation.pt','route2_end_allocation.pt','--tiers 1 2 3',
                     'test_best_uniform','test_best_mask','--metadata-code-rate 1 --metadata-modulation-bits 2'):
            self.assertIn(text,result.stdout)
        self.assertIn('ignoring inherited INIT/ALLOCATION_INIT',result.stdout)
        self.assertNotIn('--init ',result.stdout)
        self.assertNotIn('--allocation-init ',result.stdout)
        self.assertNotIn('--rate-reference-symbols',result.stdout)

    def test_launcher_stage_overrides_and_missing_gpu(self):
        result=launcher.LauncherTests().launch({'CUDA_VISIBLE_DEVICES':'0','BOOTSTRAP_STEPS':'4','RENDER_STEPS':'8',
                                     'ALLOCATION_STEPS':'7','JOINT_FINETUNE_STEPS':'3','FINAL_EVAL':'0'},
                                     script='scripts/train_progressive16_rates41224_full.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for text in ('--bootstrap-steps 4','--render-steps 8','--joint-steps 10','--mask-only-steps 7'):
            self.assertIn(text,result.stdout)
        self.assertNotIn('-m gaussian_jscc evaluate',result.stdout)
        self.assertNotEqual(launcher.LauncherTests().launch(script='scripts/train_progressive16_rates41224_full.sh').returncode,0)
        invalid=launcher.LauncherTests().launch({'CUDA_VISIBLE_DEVICES':'0','ALLOCATION_STEPS':'0'},
                                      script='scripts/train_progressive16_rates41224_full.sh')
        self.assertNotEqual(invalid.returncode,0)


if __name__=='__main__':
    unittest.main()
