"""Geometry, information-boundary and optimization checks for the opt-in context."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from gaussian_jscc.data import Geometry, prepare, to_features, to_raw
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec
from test_gaussian_jscc import fixture


class PointContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cfg(self, variant='both'):
        return ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),block_size=16,
                                  decoder_window=4,context_variant=variant,context_centers=(8,2),
                                  context_neighbors=(4,8))

    def test_world_metric_does_not_distort_axes(self):
        from gaussian_jscc.pointnet_context import metric_coordinates, radius_neighbors
        # World offsets (2,0,0) and (0,1,0). Unit-bbox L2 incorrectly prefers X.
        xyz=torch.tensor([[[0.,0,0],[.02,0,0],[0,.1,0]]])
        active=torch.ones(1,3,dtype=torch.bool)
        p=metric_coordinates(xyz,torch.tensor([100.,10.,100.]),active)
        idx,valid,_=radius_neighbors(p,p[:,:1],active,active[:,:1],torch.tensor([[.015]]),3)
        self.assertEqual(set(idx[valid].tolist()),{0,2})

    def test_masking_gradients_and_point_permutation(self):
        from gaussian_jscc.pointnet_context import SpatialPointContext
        torch.manual_seed(13); net=SpatialPointContext(self.cfg())
        net.set_scene_geometry(Geometry([0,0,0],[100,10,80],16))
        xyz=torch.rand(2,17,3); h=torch.randn(2,17,16,requires_grad=True)
        active=torch.ones(2,17,dtype=torch.bool); active[0,12:]=False; active[1]=False
        out=net(h,xyz,active)
        self.assertTrue(torch.isfinite(out).all()); self.assertEqual(int(out[~active].count_nonzero()),0)
        dirty_h=h.detach().clone(); dirty_xyz=xyz.clone()
        dirty_h[~active]=float('nan'); dirty_xyz[~active]=float('nan')
        torch.testing.assert_close(out,net(dirty_h,dirty_xyz,active))
        perm=torch.randperm(17)
        torch.testing.assert_close(net(h[:,perm],xyz[:,perm],active[:,perm]),out[:,perm],atol=1e-5,rtol=1e-5)
        out.square().mean().backward()
        self.assertGreater(float(h.grad[active].abs().sum()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None),0)

    def test_common_weights_rng_and_legacy_identity(self):
        from gaussian_jscc.transport import model_id
        base=self.cfg('baseline'); record=base.to_dict()
        self.assertNotIn('context_variant',record)
        ids=[]; models=[]
        for variant in ('baseline','encoder','decoder','both'):
            torch.manual_seed(42); models.append(ModularGaussianCodec(self.cfg(variant)))
            ids.append(torch.rand(4))
        for net,random_after in zip(models[1:],ids[1:]):
            torch.testing.assert_close(random_after,ids[0],rtol=0,atol=0)
            for n in ('channel_encoder','channel_decoder'):
                for a,b in zip(getattr(net,n).parameters(),getattr(models[0],n).parameters()):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.manual_seed(42); rebuilt=ModularGaussianCodec(ModularCodecConfig.from_dict(record))
        self.assertEqual(model_id(models[0]),model_id(rebuilt))

    def test_coincident_single_and_empty_blocks_remain_finite(self):
        from gaussian_jscc.pointnet_context import SpatialPointContext
        net=SpatialPointContext(self.cfg()); geom=Geometry([0,0,0],[3,4,5],16)
        xyz=torch.ones(3,16,3)*.5; h=torch.randn(3,16,16,requires_grad=True)
        active=torch.zeros(3,16,dtype=torch.bool); active[0,0]=True; active[1,:]=True
        with self.assertRaisesRegex(ValueError,'metadata'): net(h,xyz,active)
        net.set_scene_geometry(geom); result=net(h,xyz,active)
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(result).all()); self.assertTrue(torch.isfinite(h.grad).all())
        self.assertEqual(int(result[2].count_nonzero()),0)
        self.assertEqual(net(h[:,:0],xyz[:,:0],active[:,:0]).shape,(3,0,16))

    def test_geometry_diagnostics_report_neighbor_counts(self):
        from gaussian_jscc.pointnet_context import geometry_diagnostics
        xyz=torch.arange(12).float()[None,:,None].expand(1,12,3)/12
        active=torch.ones(1,12,dtype=torch.bool)
        rows=geometry_diagnostics(xyz,active,Geometry([0,0,0],[10,10,10],16),self.cfg())
        self.assertEqual(len(rows),4)
        for r in rows:
            self.assertGreaterEqual(r['mean_neighbors'],1.)
            self.assertLessEqual(r['mean_neighbors'],r['neighbor_cap'])
            self.assertGreater(r['mean_radius_world'],0.)

    def test_receiver_uses_only_packet_geometry_and_preserves_prefix_budget(self):
        from gaussian_jscc.transport import transmit,receive,save_checkpoint,load_checkpoint
        raw,_=fixture(35,1); raw[:,0]*=20
        q=torch.arange(35)%4
        net=ModularGaussianCodec(self.cfg()); ordered,g,oq=prepare(raw,16,q)
        net.set_scene_geometry(g); expected=[]
        for start in range(0,35,16):
            f,x=to_features(ordered[start:start+16],g,net); qb=oq[start:start+16]
            pred=net(f,x,qb,10,'none'); expected.append(to_raw(pred[qb>0],g,net))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); save_checkpoint(p/'codec.pt',net,0)
            stats=transmit(net,raw,q,10,'none',42,p/'packet')
            receiver=load_checkpoint(p/'codec.pt','cpu')
            receiver.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('source leaked'))
            out=receive(receiver,p/'packet')
            torch.testing.assert_close(out,torch.cat(expected),atol=1e-5,rtol=1e-5)
            self.assertEqual(stats['payload_complex_symbols'],sum((0,4,12,24)[int(v)] for v in q))

    def test_replay_and_checkpoint_have_same_gradients(self):
        from gaussian_jscc.modular_training import render_step,configure_phase
        raw,_=fixture(20,1); raw,g,_=prepare(raw,16)
        net=ModularGaussianCodec(self.cfg()); f,_=to_features(raw,g,net)
        batches=[(f[:16][None],torch.full((1,16),2)),(f[16:][None],torch.full((1,4),3))]
        other=copy.deepcopy(net)
        for phase in ('semantic_render','channel_render'):
            for m,mode in ((net,'checkpoint'),(other,'replay')):
                configure_phase(m,phase,{})
                torch.manual_seed(9)
                render_step(m,batches,g,10,'awgn',lambda x:x[:,3:].square().mean(),phase,mode=mode)
            for (n,a),(n2,b) in zip(net.named_parameters(),other.named_parameters()):
                self.assertEqual(n,n2)
                if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5,msg=n)


if __name__=='__main__': unittest.main()
