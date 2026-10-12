import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_modular_training import setup


class ModularCheckpointTests(unittest.TestCase):
    def test_resume_preserves_adam_rng_and_schedule(self):
        from gaussian_jscc.modular_checkpoint import capture_rng,restore_rng,save_training_checkpoint,restore_training_checkpoint
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step
        m,g,f,q=setup(); opt=configure_phase(m,'semantic_bootstrap',{})
        random.seed(6); np.random.seed(6); torch.manual_seed(6)
        def update(net,optim):
            snr=random.choice([0,10,20]); np.random.rand()
            optim.zero_grad(); bootstrap_step(net,f,q,g,snr,'none','semantic_bootstrap'); optim.step()
            return snr
        for _ in range(4): update(m,opt)
        progress={'phase':'semantic_bootstrap','phase_step':4,'step':4,'module_lrs':{},'completed':{},'channel_trained':False}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'checkpoint.pt'
            save_training_checkpoint(p,m,opt,progress,capture_rng(),{'fingerprint':'abc','validation_indices':[0]})
            snrs=[update(m,opt) for _ in range(4)]
            other,oo,pp,rng,record=restore_training_checkpoint(p,'cpu')
            self.assertEqual(pp,progress); restore_rng(rng)
            self.assertEqual(snrs,[update(other,oo) for _ in range(4)])
            for a,b in zip(m.parameters(),other.parameters()): torch.testing.assert_close(a,b,atol=0,rtol=0)
            for a,b in zip(opt.state.values(),oo.state.values()):
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_semantic_init_not_resume(self):
        from gaussian_jscc.modular_checkpoint import capture_rng,save_training_checkpoint,initialize_channel_run,validate_scene
        from gaussian_jscc.modular_training import configure_phase
        m,_,_,_=setup(); opt=configure_phase(m,'semantic_render',{})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'checkpoint.pt'
            record={'fingerprint':'abc','validation_indices':[0]}
            progress={'phase':'semantic_render','phase_step':2,'step':4,'module_lrs':{},'completed':{'semantic_render':2},'phase_complete':True}
            save_training_checkpoint(p,m,opt,progress,capture_rng(),record)
            fresh=initialize_channel_run(p,'cpu')
            for name in ('semantic_encoder','semantic_decoder'):
                for a,b in zip(getattr(m,name).parameters(),getattr(fresh,name).parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertFalse(torch.equal(m.channel_encoder.input.weight,fresh.channel_encoder.input.weight))
            with self.assertRaises(ValueError): validate_scene(record,dict(record,fingerprint='bad'))
            with self.assertRaises(ValueError): validate_scene(record,dict(record,validation_indices=[1]))
            progress['phase_complete']=False
            save_training_checkpoint(p,m,opt,progress,capture_rng(),record)
            with self.assertRaises(ValueError): initialize_channel_run(p,'cpu')

if __name__=='__main__': unittest.main()
