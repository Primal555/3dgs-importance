import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_gaussian_jscc import fixture
import test_xyz_payload_codec as codec_fixture


class PayloadTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_packet_has_no_xyz_and_receiver_needs_no_source(self):
        from gaussian_jscc.xyz_payload_transport import export_packet,receive_packet
        helper=codec_fixture.XYZPayloadCodecTests(); model=helper.model()
        raw,_=fixture(17,1); q=torch.tensor([1,2,3,0,1,2,3,1,2,3,1,2,3,1,2,3,1])
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'packet'
            torch.manual_seed(19); rng=torch.get_rng_state().clone()
            cost=export_packet(model,raw,q,10.,'none',42,root)
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            self.assertEqual(cost['payload_complex_symbols'],int(torch.tensor(model.cfg.rates)[q].sum()))
            self.assertEqual(cost['xyz_bytes'],0); self.assertFalse((root/'xyz.bin').exists())
            recovered,header=receive_packet(model,root)
            self.assertEqual(tuple(recovered.shape),(16,23)); self.assertTrue(torch.isfinite(recovered).all())
            self.assertNotIn('source',header)
            other=helper.model(); other.attr_mean.add_(1)
            with self.assertRaisesRegex(ValueError,'model'): receive_packet(other,root)
            np.save(root/'received.npy',np.zeros((1,2),np.float32))
            with self.assertRaisesRegex(ValueError,'symbol|length'): receive_packet(model,root)

    def test_corrupt_metadata_and_unexpected_coordinate_sidecar_are_rejected(self):
        from gaussian_jscc.xyz_payload_transport import export_packet,receive_packet
        model=codec_fixture.XYZPayloadCodecTests().model(); raw,_=fixture(8,1)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'packet'; export_packet(model,raw,torch.ones(8,dtype=torch.long),10.,'awgn',42,root)
            original=(root/'metadata.bin').read_bytes()
            (root/'metadata.bin').write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaisesRegex(ValueError,'CRC'): receive_packet(model,root)
            (root/'metadata.bin').write_bytes(original); (root/'xyz.bin').write_bytes(b'forbidden')
            with self.assertRaisesRegex(ValueError,'side|coordinate'): receive_packet(model,root)


if __name__=='__main__': unittest.main()
