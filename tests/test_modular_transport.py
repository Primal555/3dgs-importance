import tempfile
import unittest
from pathlib import Path
import torch
from test_gaussian_jscc import fixture
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, model_id, transmit, receive
from gaussian_jscc.data import prepare, to_features, to_raw
from gaussian_jscc.position_delivery import delivered_positions


class ModularTransportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_legacy_identity_and_modular_roundtrip(self):
        from gaussian_jscc.model_factory import config_from_dict, build_codec
        from gaussian_jscc.modular_config import ModularCodecConfig
        old=CodecConfig(hidden=16,grid_dim=4,levels=(2,))
        self.assertEqual(config_from_dict(old.to_dict()).to_dict(),old.to_dict())
        legacy=GaussianCodec(old); rebuilt=build_codec(config_from_dict(old.to_dict()))
        rebuilt.load_state_dict(legacy.state_dict()); self.assertEqual(model_id(legacy),model_id(rebuilt))
        with self.assertRaises(ValueError): config_from_dict({'architecture':'unknown'})
        with self.assertRaises(ValueError): CodecConfig.from_dict(ModularCodecConfig().to_dict())
        model=build_codec(ModularCodecConfig(hidden=16,grid_dim=4,levels=(2,)))
        with tempfile.TemporaryDirectory() as t:
            save_checkpoint(Path(t)/'codec.pt',model,0)
            copy=load_checkpoint(Path(t)/'codec.pt','cpu')
            self.assertEqual(model_id(copy),model_id(model))

    def test_receiver_only_without_source(self):
        from gaussian_jscc.model_factory import build_codec
        from gaussian_jscc.modular_config import ModularCodecConfig
        raw,_=fixture(19,1)
        model=build_codec(ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),block_size=8))
        q=torch.arange(19)%4; q[8:16]=0
        ordered,geom,oq=prepare(raw,16,q)
        expected=[]
        for start in range(0,len(raw),8):
            f,x=to_features(ordered[start:start+8],geom,model)
            out=model(f,x,oq[start:start+8],10,'none')
            expected.append(to_raw(out[oq[start:start+8]>0],geom,model))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t); save_checkpoint(path/'codec.pt',model,0)
            stats=transmit(model,raw,q,10,'none',7,path/'packet')
            del raw,ordered,model
            receiver=load_checkpoint(path/'codec.pt','cpu')
            receiver.semantic_encoder.forward=lambda *a: (_ for _ in ()).throw(AssertionError('receiver accessed encoder'))
            decoded=receive(receiver,path/'packet')
            torch.testing.assert_close(decoded,torch.cat(expected),atol=1e-6,rtol=1e-5)
            self.assertEqual(stats['payload_complex_symbols'],sum((0,4,12,24)[int(x)] for x in oq))
            self.assertEqual(stats['position_stream_bytes'],(path/'packet/xyz.bin').stat().st_size)

if __name__=='__main__': unittest.main()
