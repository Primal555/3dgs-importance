import copy
import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_footprint_controls as controls


def dataset(tmp_path):
    from test_benefit_stage_one import source
    from gaussian_jscc.benefit_stage_one_data import snapshot
    from gaussian_jscc.full_history_spatial_replay import FullHistorySpatialReplay
    root, _ = source(tmp_path)
    saved = snapshot(root)
    data = tmp_path/'data'; data.mkdir()
    replay = FullHistorySpatialReplay(data/'replay', max_groups=6, validation_groups=2)
    for validation, entries in ((False, saved['replay'].training[:6]), (True, saved['replay'].validation)):
        for entry in entries:
            g = copy.deepcopy(saved['replay']._read(entry))
            for view in g['views']:
                view['inputs']['footprints'] = torch.zeros(4, 2, 15, 9, 9)
            replay.add(g, validation)
    replay.freeze_validation()
    metadata = dict(version='candidate_footprint_labels_v1', config=saved['settings'], seed=46006,
                    source_initial_sha256='fixture', train_records=6, validation_records=2)
    (data/'metadata.json').write_text(json.dumps(metadata))
    (data/'data_manifest.json').write_text(json.dumps(dict(replay_root=str(replay.root),
        training=replay.training, validation=replay.validation)))
    torch.save(dict(learner=saved['learner']), data/'initial.pt')
    (data/'complete.json').write_text(json.dumps(dict(status='complete')))
    return data


def test_two_structures_share_labels_initial_outputs_and_uniform_updates(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    import gaussian_jscc.rendering
    monkeypatch.setattr(gaussian_jscc.rendering, 'render', lambda *a, **k: pytest.fail('offline render'))
    out = tmp_path/'train'
    args = controls.parser().parse_args(['--phase', 'train', '--data', str(data), '--out', str(out),
        '--device', 'cpu', '--updates', '2', '--evaluate-every', '1', '--cpu-threads', '1'])
    result = controls.main(args)
    assert result['status'] == 'complete' and result['trained_arms'] == 2
    assert result['training_coverage']['draws'] == 4
    assert result['initial_max_prediction_difference'] < 1e-6
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert (out/'review.tar.gz').is_file()
    assert (out/'original/final.pt').is_file() and (out/'upgraded/final.pt').is_file()
    assert result['after']['original']['validation']['all']['records'] == 2
    assert result['after']['upgraded']['validation']['ordinary']['records'] > 0
    with pytest.raises(FileExistsError):
        controls.main(args)


def test_wrong_role_rejected_and_architecture_identified(tmp_path):
    data = dataset(tmp_path)
    manifest = json.loads((data/'data_manifest.json').read_text())
    # A validation shard put into the training list must not influence training.
    manifest['training'][0] = manifest['validation'][0]
    (data/'data_manifest.json').write_text(json.dumps(manifest))
    args = controls.parser().parse_args(['--phase', 'train', '--data', str(data),
        '--out', str(tmp_path/'train'), '--device', 'cpu', '--cpu-threads', '1'])
    with pytest.raises(ValueError, match='overlap|validation'):
        controls.main(args)


@pytest.mark.parametrize('response_terms',[False,True])
def test_real_cpu_collection_saves_roles_and_resumes_without_new_labels(tmp_path, monkeypatch,response_terms):
    from types import SimpleNamespace
    from test_successive_allocation import small_successive
    from gaussian_jscc.model_factory import build_codec
    from gaussian_jscc.transport import model_id
    from gaussian_jscc import benefit_footprint_collect as collect
    raw, geometry, features, old_model = small_successive()
    cfg = old_model.cfg; cfg.rates = (0, 8, 24, 48)
    model = build_codec(cfg); model.set_scene_geometry(geometry)
    model.eval().requires_grad_(False); model.set_training_snrs([10.])
    identity = model_id(model)
    transform = torch.eye(4); transform[3, 2] = 3
    cameras = [SimpleNamespace(image_width=16, image_height=16, FoVx=1.2, FoVy=1.2,
        world_view_transform=transform, full_proj_transform=torch.eye(4), camera_center=torch.zeros(3),
        image_name=f'view{i}') for i in range(3)]
    scene = SimpleNamespace(raw=raw, order=torch.arange(len(raw)), groups=[features],
        group_ids=[torch.arange(16).reshape(2, 8)], geometry=geometry, degree=1, white_background=False,
        cameras=cameras, fingerprint='fixture', reference=SimpleNamespace(get=lambda c, d: torch.zeros(3,16,16,device=d)))
    pipeline = tmp_path/'pipeline'; pipeline.mkdir()
    reference = dict(d_ref=1., lambda_s=.1, c_ref=100., beta=.1, side_proxy=dict(xyz_per_retained=1., bits_per_use=2.))
    settings = dict(aligned_spatial_inputs=True, seed=42, beta=.1, window_views=1, calibration_views=1)
    normalizers = {k: torch.ones(2) for k in ('input_mean','input_scale','context_mean','context_scale',
                                           'state_mean','state_scale','output_scale')}
    torch.save(dict(settings=settings, reference=reference, codec_id=identity,
        learner=dict(network=normalizers), scene_fingerprint='fixture'), pipeline/'benefit_network.pt')
    (pipeline/'plan.json').write_text(json.dumps(dict(settings=settings)))
    monkeypatch.setattr(collect, 'load_trial', lambda *a, **k: dict(initial=dict(settings=dict(seed=46006,
        source_checkpoint=str(pipeline/'allocation/checkpoints/latest/state.pt')), learner=dict(network=normalizers)), config={}, initial_sha256='fixture'))
    monkeypatch.setattr(collect, 'load_inputs', lambda a: (dict(state_dict=model.state_dict()), cfg, dict(ply='fixture'), {}, None, {}))
    monkeypatch.setattr(collect, 'build_codec', lambda c: copy.deepcopy(model))
    monkeypatch.setattr(collect, 'read_ply', lambda p: (raw, 1))
    monkeypatch.setattr(collect, 'make_scene', lambda *a, **k: scene)
    monkeypatch.setattr(collect, 'preflight', lambda *a, **k: dict(cpu_fixture=True))
    out = tmp_path/'collected'
    args = controls.parser().parse_args(['--phase', 'collect', '--comparison', 'fixture', '--codec', 'fixture',
        '--pipeline-root', str(pipeline), '--out', str(out), '--device', 'cpu', '--cpu-threads', '1',
        '--train-records', '4', '--validation-records', '2', '--label-groups', '4'])
    if response_terms:
        from gaussian_jscc.benefit_response_features import ResponseTermsWindow, SCHEMA
        def run(a):return collect.collect(a,window_factory=ResponseTermsWindow,label_version=SCHEMA)
    else:
        run=controls.main
    args.max_collect_windows = 1
    with pytest.raises(RuntimeError, match='window limit'):
        run(args)
    args.max_collect_windows = 150; args.resume = True
    result = run(args)
    assert result['frozen_codec_verified']
    manifest = json.loads((out/'data_manifest.json').read_text())
    assert len(manifest['training']) == 4 and len(manifest['validation']) == 2
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    replay = SpatialReplay(out/'replay', max_groups=1, validation_groups=2)
    train = [replay._read(e) for e in manifest['training']]
    val = [replay._read(e) for e in manifest['validation']]
    assert not {r['source'] for r in train} & {r['source'] for r in val}
    assert all(v['inputs']['footprints'].shape == (4,2,15,9,9) for r in train+val for v in r['views'])
    if response_terms:
        for g in train+val:
            for v in g['views']:
                assert v['label_schema']==SCHEMA and v['response_terms'][:,1].min()>=0
                torch.testing.assert_close(v['values'],torch.stack(
                    (v['response_terms'][:,0]+v['response_terms'][:,1],v['response_terms'][:,2]),-1))
    args.resume = True
    assert run(args) == result
    args.train_records += 1
    with pytest.raises(ValueError, match='completed collection'):
        run(args)


def test_paired_training_resume_replays_the_same_final_weights(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    out = tmp_path/'interrupted'
    args = controls.parser().parse_args(['--phase', 'train', '--data', str(data), '--out', str(out),
        '--device', 'cpu', '--updates', '2', '--evaluate-every', '1', '--cpu-threads', '1'])
    original_step = torch.optim.AdamW.step
    calls = 0
    def interrupt(optimizer, *a, **k):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated interruption')
        return original_step(optimizer, *a, **k)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, 'step', interrupt)
        with pytest.raises(RuntimeError, match='simulated'):
            controls.main(args)
    args.resume = True
    controls.main(args)
    straight = tmp_path/'straight'
    args.out = str(straight); args.resume = False
    # Profile CNN uses random initialization. Both fresh runs use the experiment seed.
    controls.main(args)
    for arm in ('original', 'upgraded'):
        a = torch.load(out/arm/'final.pt', map_location='cpu', weights_only=True)['learner']['network']
        b = torch.load(straight/arm/'final.pt', map_location='cpu', weights_only=True)['learner']['network']
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_resume_recovers_interrupted_final_artifacts_and_archive(tmp_path, monkeypatch):
    from gaussian_jscc import benefit_footprint_collect as collector
    data = dataset(tmp_path); out = tmp_path/'ending'
    args = controls.parser().parse_args(['--phase','train','--data',str(data),'--out',str(out),
        '--device','cpu','--updates','1','--evaluate-every','1','--cpu-threads','1'])
    original_save = collector.atomic_save
    def stop_after_first_final(value, path):
        original_save(value, path)
        if str(path).endswith('original\\final.pt') or str(path).endswith('original/final.pt'):
            raise RuntimeError('interrupted final artifacts')
    with monkeypatch.context() as patch:
        patch.setattr(collector,'atomic_save',stop_after_first_final)
        with pytest.raises(RuntimeError, match='final artifacts'):
            controls.main(args)
    args.resume = True
    result = controls.main(args)
    assert result['status']=='complete' and (out/'review.tar.gz').is_file()
    (out/'review.tar.gz').unlink()
    assert controls.main(args)['status']=='complete'
    assert (out/'review.tar.gz').is_file()


def test_script_resumes_collection_then_starts_new_training(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.exists():
        pytest.skip('Git Bash unavailable')
    out = tmp_path/'experiment'; (out/'data').mkdir(parents=True)
    (out/'data/latest.pt').write_text('collection state')
    shim = tmp_path/'python_shim.sh'
    shim.write_text('''#!/usr/bin/env bash
phase=""; out=""; resume=0
while (( $# )); do
  case "$1" in
    --phase) phase="$2"; shift 2;;
    --out) out="$2"; shift 2;;
    --resume) resume=1; shift;;
    *) shift;;
  esac
done
if [[ "$phase" == train && "$resume" == 1 && ! -f "$out/settings.json" ]]; then exit 71; fi
mkdir -p "$out"
if [[ "$phase" == train ]]; then touch "$out/review.tar.gz"; fi
printf '%s:%s\\n' "$phase" "$resume"
''', encoding='utf8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', RESUME='1', PYTHON_BIN=shim.as_posix())
    run = subprocess.run([str(bash), 'scripts/test_benefit_footprint_controls.sh',
        'comparison', 'codec', out.as_posix()], env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout+run.stderr
    assert 'collect:1' in run.stdout and 'train:0' in run.stdout
    assert (out/'review.tar.gz').is_file()
