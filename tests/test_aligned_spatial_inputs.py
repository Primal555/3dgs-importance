from types import SimpleNamespace
from unittest.mock import patch
import torch
from gaussian_jscc.aligned_spatial_inputs import AlignedSpatialWindow


def test_packet_probe_and_original_feature_ids_are_distinct():
    window=object.__new__(AlignedSpatialWindow)
    raw=torch.arange(20,dtype=torch.float32).reshape(4,5)
    order=torch.tensor([2,0,3,1])
    window.scene=SimpleNamespace(raw=raw,order=order)
    window.args=SimpleNamespace(neighbors=2,cpu_threads=1,candidate_batch=8,allocation_batch=None)
    mixed=torch.cat((raw+100,torch.zeros(4,1)),1)
    captured=window.capture_inputs(mixed)
    assert torch.equal(captured[:,:-1],raw)
    assert torch.equal(mixed[:,:-1],raw+100)
    static=torch.arange(12,dtype=torch.float32).reshape(4,3)
    window.static=None;tile=SimpleNamespace(height=2,width=2)
    with patch('gaussian_jscc.aligned_spatial_inputs.static_features',return_value=static), \
         patch('gaussian_jscc.aligned_spatial_inputs.visibility_statistics',return_value=torch.ones(4,12)):
        window.prepare_camera(tile,None)
    assert torch.equal(window.static[order],static)
    assert window.candidate_batch_size==8
    window.args.allocation_batch=128
    assert window.candidate_batch_size==128
