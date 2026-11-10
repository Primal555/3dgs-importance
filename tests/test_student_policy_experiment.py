import contextlib
import io
import json
import tarfile
from unittest.mock import patch
import pytest
import torch
from gaussian_jscc import student_policy_experiment as entry
from gaussian_jscc import structured_evidence_allocation as teacher_entry
from test_structured_allocation_integration import source, options
import test_budgeted_allocation as fixture


def capture_view(model, scene, camera, verify=False):
    features = torch.zeros(len(scene.raw), 14)
    features[scene.order, :3] = scene.raw[:, :3].cpu()
    features[:, 3] = camera.factor
    return dict(features=features, visible=torch.ones(len(features)).bool(), max_renderer_error=0.)


def args_for(parent, teacher, out, *extra):
    return entry.parser().parse_args(['--run', str(parent), '--teacher-run', str(teacher), '--scene', 'truck',
        '--out', str(out), '--beta', '.1', '--distill-epochs', '2', '--rd-steps', '2', '--width', '16',
        '--coarse-width', '24', '--blocks', '1', '--chunk-size', '7', '--finite-samples', '4',
        '--candidate-batch', '8', '--calibration-views', '1', '--window-views', '4', '--preparation-views', '2',
        '--validate-every', '1', '--save-every', '1', '--profile-every', '1', '--test-views', '2',
        '--trials', '1', '--image-views', '1', '--cpu-threads', '1', '--device', 'cpu', '--no-lpips', *extra])


@pytest.mark.parametrize('interrupt_after', [1, 3])
def test_two_phase_exact_resume_zero_update_prediction_and_compact_review(source, interrupt_after):
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view', side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teacher = root/'teacher'; teacher_entry.main(options(parent, teacher))
        out = root/'student'; result = entry.main(args_for(parent, teacher, out))
        assert result['status'] == 'complete' and result['rd_updates'] == 2 and result['distill_epochs'] == 2
        full = torch.load(out/'checkpoints/latest/student.pt', weights_only=True)
        resumed_out = root/'resumed'; args = args_for(parent, teacher, resumed_out)
        assert entry.main(args, max_updates=interrupt_after)['status'] == 'interrupted'
        args.resume = True; resumed_result = entry.main(args)
        resumed = torch.load(resumed_out/'checkpoints/latest/student.pt', weights_only=True)
        for name, value in full['student'].items():
            torch.testing.assert_close(value, resumed['student'][name], rtol=0, atol=0)
        assert torch.equal(full['rng']['torch'], resumed['rng']['torch'])
        assert result['tier_sha256'] == resumed_result['tier_sha256']
        assert resumed_result['frozen_codec_verified'] and resumed_result['frozen_attributes_verified']
        plan = json.loads((out/'plan.json').read_text())
        assert not (set(plan['preparation_view_names']) & set(plan['calibration_view_names']))
        assert not (set(plan['preparation_view_names']) & set(plan['test_view_names']))
        with tarfile.open(out/'review.tar.gz') as archive: names = archive.getnames()
        assert 'review_summary.json' in names and 'deployment/selected_tiers.npy' in names
        assert not any(n.startswith(('preparation/', 'checkpoints/')) for n in names)
        prediction = root/'predicted'
        predict_args = args_for(parent, teacher, prediction, '--phase', 'predict',
                               '--student-checkpoint', str(out/'checkpoints/best/student.pt'))
        predicted = entry.main(predict_args)
        assert predicted['optimizer_updates'] == 0 and predicted['tier_sha256'] == result['tier_sha256']


def test_prediction_new_scene_has_no_teacher_no_optimizer_and_fixed_normalizers(source):
    from gaussian_jscc.data import read_ply, write_ply
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch('gaussian_jscc.student_policy_features.capture_view', side_effect=capture_view), \
         contextlib.redirect_stdout(io.StringIO()):
        teacher = root/'teacher'; teacher_entry.main(options(parent, teacher))
        out = root/'student'; entry.main(args_for(parent, teacher, out))
        metadata = json.loads((parent/'training.json').read_text())
        spec = metadata['selected_scenes'][0]
        raw, degree = read_ply(spec['ply']); raw[:, 11:14] += .03
        new_ply = root/'heldout.ply'; write_ply(new_ply, raw, degree)
        manifest = root/'heldout.json'
        manifest.write_text(json.dumps(dict(scenes=[dict(spec, name='heldout', role='heldout', ply=str(new_ply))])))
        saved = torch.load(parent/'checkpoints/final/codec.pt', weights_only=True)
        shared_meta = dict(scene_records={'truck': metadata['scene']}, settings=dict(net_bits_per_use=2.))
        predict_args = args_for(parent, teacher, root/'new_prediction', '--phase', 'predict', '--manifest', str(manifest),
            '--scene', 'heldout', '--student-checkpoint', str(out/'checkpoints/best/student.pt'))
        predict_args.teacher_run = None
        with patch('gaussian_jscc.successive_transfer.load_source', return_value=(saved, shared_meta,
                   __import__('gaussian_jscc.model_factory', fromlist=['config_from_dict']).config_from_dict(saved['config']),
                   {'codec_source': 'fixture'})):
            predicted = entry.main(predict_args)
        assert predicted['optimizer_updates'] == 0 and predicted['fixed_feature_normalizers']
        assert predicted['reference_preparation_seconds'] > 0
        assert predicted['allocation_seconds'] >= predicted['reference_preparation_seconds']+predicted['student_forward_seconds']


def test_interrupted_preparation_reuses_completed_source_camera(source):
    parent, root = source
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=fixture.cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), contextlib.redirect_stdout(io.StringIO()):
        teacher = root/'teacher'; teacher_entry.main(options(parent, teacher))
        out = root/'preparation_resume'; args = args_for(parent, teacher, out)
        calls = []
        def interrupted(model, scene, camera, verify=False):
            calls.append(camera.image_name)
            if len(calls) == 2: raise RuntimeError('synthetic preparation interruption')
            return capture_view(model, scene, camera, verify)
        with patch('gaussian_jscc.student_policy_features.capture_view', side_effect=interrupted):
            with pytest.raises(RuntimeError, match='synthetic preparation interruption'): entry.main(args)
        assert not (out/'checkpoints/latest/student.pt').exists()
        assert (out/'preparation/visibility/000.pt').exists()
        args.resume = True
        with patch('gaussian_jscc.student_policy_features.capture_view', side_effect=capture_view) as capture:
            result = entry.main(args)
        assert result['status'] == 'complete' and capture.call_count == 1
