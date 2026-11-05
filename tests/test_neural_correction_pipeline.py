import copy
import json
from pathlib import Path
import pytest
import torch
from unittest.mock import patch
from gaussian_jscc import neural_correction_allocation as entry
from test_hybrid_tier_pipeline import environment,options as old_options
from test_structured_allocation_integration import source
from test_learned_benefit_allocation import read,assert_nested_equal


def options(parent,out):
    args=entry.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out)])
    vars(args).update(vars(old_options(parent,out,'online')))
    args.replay_capacity=32
    return args


def test_online_exact_teacher_and_resume(source):
    parent,root=source;full=root/'correction_full';split=root/'correction_split'
    provider=entry.NetworkCorrectionProvider;validation_masks=[]
    def construct(*args,**kwargs):
        result=provider(*args,**kwargs);validation_masks.append(result.window.scheduler.validation.clone())
        return result
    with environment(),patch.object(entry,'NetworkCorrectionProvider',side_effect=construct):
        result=entry.main(options(parent,full))
        entry.main(options(parent,split),max_updates=5)
        args=options(parent,split);args.resume=True;entry.main(args)
    a=read(full/'allocation/checkpoints/latest/state.pt');b=read(split/'allocation/checkpoints/latest/state.pt')
    for key in ('table','views','rng','positive_provider'):assert_nested_equal(a[key],b[key])
    aa=copy.deepcopy(a['blocks']);bb=copy.deepcopy(b['blocks'])
    for state in (aa,bb):
        for record in state['calibrated']['positive']['records'].values():record.pop('record_id')
    assert_nested_equal(aa,bb)
    assert result['network_updates']==5 and result['frozen_codec_verified']
    assert result['positive_history_policy']=='current_network_revision'
    assert (a['blocks']['structured']['tensors']['keep_observations'].sum(1)==2).all()
    rows=[json.loads(x) for x in (full/'allocation/metrics/loss.jsonl').read_text().splitlines()]
    assert all(r['tier_max_log_odds']==1. and r['neural_exact_training_groups']>0 for r in rows)
    assert all(r['neural_exact_extra_teacher_renders']==0 for r in rows)
    assert all('neural_full_state_regression_loss' in r and 'neural_observed_gap_regression_loss' in r for r in rows)
    assert all(r['neural_history_invalidated_rows']==r['neural_candidates'] for r in rows)
    for saved,out in ((a,full),(b,split)):
        active={record['id'] for record in saved['replay']['training']}
        assert all(record['record_id'] in active for record in saved['blocks']['calibrated']['positive']['records'].values())
        validation=validation_masks[0]
        gaps=[]
        for record in saved['replay']['training']:
            group=read(out/'allocation/neural_replay'/record['path'])
            if group.get('supervision')=='action_gap':
                gaps.append(group)
                assert not validation[group['source']]
        assert gaps
    assert (full/'allocation/metrics/network_validation.jsonl').is_file()
    report=json.loads((full/'allocation/comparison/structured_evidence.json').read_text())
    if report['selected_step']:
        assert report['checkpoint_network_updates']==report['positive_network_feedback_revision']+1
    else:
        assert report['positive_network_feedback_revision'] is None
    pretrained=read(full/'benefit_network.pt')['learner']['network']
    assert any(not torch.equal(v,a['positive_provider']['learner']['network'][k]) for k,v in pretrained.items())


def test_defaults_and_no_cross_policy_resume(source):
    parent,root=source;out=root/'correction_dry'
    args=entry.parser().parse_args(['--run',str(parent),'--scene','truck','--out',str(out)])
    assert args.tier_max_log_odds==1 and args.network_mode=='online' and args.pretrain_windows==700
    args.calibration_views=1;args.dry_run=True
    with environment():plan=entry.main(args)
    assert plan['settings']['aligned_spatial_inputs'] and not out.exists()
    args.network_mode='frozen'
    with pytest.raises(ValueError,match='online'):entry.main(args)


def test_completed_updates_resume_final_evaluation_without_training(source):
    import pickle
    from gaussian_jscc import budgeted_allocation as allocator

    parent, root = source
    out = root / 'correction_final_read_failure'
    args = options(parent, out)
    original = allocator.load_checkpoint

    def fail_selected_read(path):
        if Path(path).parent.name == 'best_rd':
            raise pickle.UnpicklingError('Unsupported class _codecs.encode')
        return original(path)

    with environment(), patch.object(allocator, 'load_checkpoint', side_effect=fail_selected_read):
        with pytest.raises(pickle.UnpicklingError, match='_codecs.encode'):
            entry.main(args)
    state = original(out / 'allocation/checkpoints/latest/state.pt')
    assert state['progress']['step'] == args.allocation_steps
    assert not (out / 'complete.json').exists()
    updates = state['positive_provider']['network_updates']
    args.resume = True
    with environment(), patch.object(allocator, 'budgeted_mask_step',
                                      side_effect=AssertionError('finished training repeated')):
        result = entry.main(args)
    assert result['status'] == 'complete' and result['network_updates'] == updates
    assert (out / 'review_summary.json').is_file()
    assert (out / 'review.tar.gz').is_file()
