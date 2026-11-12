import contextlib
import io
import json
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc.student_policy_state import selected_map
from gaussian_jscc.budgeted_allocation_evaluation import tier_hash
from test_structured_allocation_integration import source, options
from test_student_policy_experiment import args_for, capture_view
import test_budgeted_allocation as fixture


def test_saved_evaluated_map_survives_boundary_flip_and_rejects_corruption():
    q = torch.tensor([0,1,2,3]); inferred=q.clone(); inferred[0]=1
    best=dict(step=1600,rd_score=1.4,tier_sha256=tier_hash(q))
    state=dict(progress=dict(best=best),evaluated_tiers=q.byte())
    got, report=selected_map(state,inferred,best)
    assert torch.equal(got,q) and report['inference_differs_from_deployed_points']==1
    assert report['matches_original_evaluation']
    state['evaluated_tiers'][1]=3
    with pytest.raises(ValueError,match='hash mismatch'): selected_map(state,inferred,best)
    state.pop('evaluated_tiers'); got, report=selected_map(state,inferred,best)
    assert torch.equal(got,inferred) and not report['matches_original_evaluation']
    assert report['source']=='reconstructed_legacy'


def test_completed_legacy_checkpoint_rechecks_calibration_without_more_training(source):
    from gaussian_jscc import student_policy_experiment as entry
    from gaussian_jscc import structured_evidence_allocation as teacher
    parent,root=source
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render',side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view',side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teach=root/'teacher'; teacher.main(options(parent,teach))
        out=root/'student'; args=args_for(parent,teach,out)
        assert entry.main(args,max_updates=4)['status']=='interrupted'
        checkpoint=out/'checkpoints/best/student.pt'
        state=torch.load(checkpoint,weights_only=True); assert 'evaluated_tiers' in state
        state.pop('evaluated_tiers'); torch.save(state,checkpoint)
        latest=out/'checkpoints/latest/student.pt'
        before=torch.load(latest,weights_only=True)
        original_loss=(out/'metrics/loss.jsonl').read_bytes()
        original_resolver=entry.recoverable_selected_map
        def boundary_flip(saved,inferred,expected,*paths):
            altered=inferred.clone(); altered[0]=(altered[0]+1)%4
            return original_resolver(saved,altered,expected,*paths)
        args.resume=True
        with patch.object(entry,'recoverable_selected_map',side_effect=boundary_flip): result=entry.main(args)
        assert result['status']=='complete' and result['rd_updates']==2
        assert (out/'metrics/loss.jsonl').read_bytes()==original_loss
        after=torch.load(latest,weights_only=True)
        for key in before['student']: torch.testing.assert_close(before['student'][key],after['student'][key],rtol=0,atol=0)
        record=json.loads((out/'evaluation/selection_reproducibility.json').read_text())
        assert not record['matches_original_evaluation'] and 'recovered_calibration_rd' in record
        assert result['tier_sha256']==record['deployed_tier_sha256']
        assert (out/'review.tar.gz').is_file()
        recovery=out/'checkpoints/recovered_selection.pt'
        assert recovery.is_file()
        saved_recovery=torch.load(recovery,weights_only=True)
        different=saved_recovery['tiers'].roll(1)
        recovered,info=original_resolver(state,different,before['progress']['best'],checkpoint,recovery)
        assert torch.equal(recovered,saved_recovery['tiers'])
