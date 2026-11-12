import contextlib
import io
import json
from unittest.mock import patch
import pytest
import torch
from comparison_baselines.side_information_accounting import side_information_cost
from gaussian_jscc.student_digital_cost import DigitalDeploymentCost, digital_side_proxy
from gaussian_jscc.student_policy_features import VisibilityAccumulator, summarize_visibility
from gaussian_jscc.successive_delivery_packet import export_packet
from gaussian_jscc.transport import model_id
from test_successive_delivery_packet import scene_model
from test_structured_allocation_integration import source, options
from test_student_policy_experiment import args_for, capture_view
import test_budgeted_allocation as fixture


@pytest.mark.parametrize('tier', [None, 0, 2, 3])
def test_digital_meter_matches_actual_serialized_packet_and_proxy(tmp_path, tier):
    model, scene = scene_model()
    q = torch.arange(len(scene.raw))%4 if tier is None else torch.full((len(scene.raw),),tier)
    meter = DigitalDeploymentCost(model, scene, model_id(model), 10., 2.)
    cost = meter.details(q)
    packet = export_packet(model,scene,q,10.,42,tmp_path/'packet')
    for k in ('xyz_bytes','metadata_bytes','scale_stream_bytes','payload_complex_symbols'):
        assert cost[k]==packet[k]
    assert cost['total_channel_uses']==side_information_cost(packet)['total_channel_uses']
    proxy = digital_side_proxy(cost,int((q>0).sum()),meter.details(torch.full_like(q,3)),len(q))
    estimate = cost['payload_complex_symbols']+int((q>0).sum())*(proxy['xyz_per_retained']+proxy['scale_per_retained'])+proxy['fixed_uses']
    assert estimate==pytest.approx(cost['total_channel_uses'])
    assert proxy['bits_per_use']==pytest.approx(4064/1536)


def test_streaming_visibility_preserves_64_view_statistics():
    torch.manual_seed(11)
    f=torch.randn(64,101,14); visible=torch.rand(64,101)>.4
    visible[:,0]=False; visible[:,1]=True
    acc=VisibilityAccumulator(101)
    for x,v in zip(f,visible): acc.add(x,v)
    torch.testing.assert_close(acc.result(),summarize_visibility(f,visible),rtol=2e-6,atol=2e-7)
    assert acc.mean.shape==(101,14)


def test_digital_student_reference_selection_resume_and_teacher_guard(source):
    from gaussian_jscc import student_policy_experiment as entry
    from gaussian_jscc import structured_evidence_allocation as teacher
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view',side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teach=root/'teacher';teacher.main(options(parent,teach))
        args=args_for(parent,teach,root/'wrong','--required-teacher-window-views','32')
        with pytest.raises(ValueError,match='teacher window views'):entry.main(args)
        out=root/'digital'; args=args_for(parent,teach,out,'--cost-protocol','digital_side_file_v1')
        assert entry.main(args,max_updates=3)['status']=='interrupted'
        args.resume=True; result=entry.main(args)
        assert result['status']=='complete'
        report=json.loads((out/'review_summary.json').read_text())
        plan=json.loads((out/'plan.json').read_text())
        assert plan['reference']['c_ref']==report['rows']['uniform_q3']['total_channel_uses']
        assert plan['teacher']['historical_rd_reference']['c_ref']!=plan['reference']['c_ref']
        for row in report['rows'].values():
            assert row['total_channel_uses']==row['comparison_accounting']['total_channel_uses']
            assert row['rd_score']==pytest.approx(row['source_distortion']/row['d_ref']+args.beta*row['total_channel_uses']/row['c_ref'])
