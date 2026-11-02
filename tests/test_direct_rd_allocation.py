import contextlib
import importlib.util
import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_structured_allocation_integration import source
from test_calibrated_spatial_benefit_integration import options as benefit_options
from test_spatial_benefit_integration import Window
from test_learned_benefit_allocation import cameras, read, assert_nested_equal
import test_budgeted_allocation as fixture


def module():
    assert importlib.util.find_spec('gaussian_jscc.direct_rd_allocation'), 'direct RD entry missing'
    from gaussian_jscc import direct_rd_allocation
    return direct_rd_allocation


class FrozenWindow:
    def __init__(self, model, scene, table, scheduler, reference, args):
        self.table, self.scheduler = table, scheduler
        self.args = args
    def collect(self, step, cameras, learner, replay, pretrain):
        assert self.args.label_groups == 0
        ids = torch.tensor([step % len(self.table.q)])
        self.scheduler.record(ids, step)
        return dict(tier_ids=ids, tier_costs=torch.tensor([[2., 3., 4.]]),
                    q=self.table.q.cpu(), metrics=dict(evaluated_groups=1, labeled_groups=0, teacher_seconds=0.))


def options(entry, parent, benefit, out):
    return entry.parser().parse_args(['--run', str(parent), '--benefit-run', str(benefit), '--scene', 'truck',
        '--out', str(out), '--device', 'cpu', '--calibration-views', '1', '--window-views', '4',
        '--allocation-steps', '3', '--audit-every', '2', '--save-every', '1', '--trials', '1',
        '--accept-trials', '1', '--test-views', '2', '--image-views', '1', '--no-lpips'])


def test_direct_entry_frozen_resume_and_compact_output(source):
    entry = module()
    from gaussian_jscc import calibrated_spatial_benefit_allocation as benefit_entry
    parent, root = source
    benefit = root/'learned'
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(benefit_entry.spatial, 'SpatialWindow', Window), \
         patch.object(benefit_entry.spatial, 'preflight', return_value={}), \
         patch.object(entry, 'BatchedSpatialWindow', FrozenWindow), \
         patch.object(entry, 'batched_preflight', return_value={}), contextlib.redirect_stdout(io.StringIO()):
        bargs = benefit_options(parent, benefit); benefit_entry.main(bargs)
        benefit_state = read(benefit/'checkpoints/latest/state.pt')
        dry = options(entry, parent, benefit, root/'direct_dry'); dry.dry_run = True
        planned = entry.main(dry)
        assert planned['frozen_benefit']['step'] == 3 and not (root/'direct_dry').exists()
        dry.benefit_checkpoint = str(benefit/'checkpoints/initial/state.pt')
        with pytest.raises(ValueError, match='unfitted'):
            entry.main(dry)
        full = options(entry, parent, benefit, root/'direct_full')
        entry.main(full)
        partial = options(entry, parent, benefit, root/'direct_resume')
        entry.main(partial, max_updates=1)
        assert not (root/'direct_resume/complete.json').exists()
        partial.resume = True; entry.main(partial)
        a = read(root/'direct_full/checkpoints/latest/state.pt')
        b = read(root/'direct_resume/checkpoints/latest/state.pt')
        for key in ('q', 'bank', 'scheduler', 'views', 'frozen', 'progress', 'calibration'):
            assert_nested_equal(a[key], b[key])
        assert a['frozen']['inference']['updates'] == benefit_state['learner']['updates']
        assert_nested_equal(a['frozen']['inference']['network'], benefit_state['learner']['network'])
        assert a['progress']['step'] == 3 and a['progress']['final_done']
        assert not (root/'direct_full/replay').exists()
        assert not (root/'direct_full/checkpoints/final').exists()
        assert 'optimizer' not in a['frozen']['inference'] and 'state_dict' not in a
        complete = json.loads((root/'direct_full/complete.json').read_text())
        assert complete['frozen_codec_verified'] and complete['frozen_benefit_verified']
        audits = [json.loads(x) for x in (root/'direct_full/metrics/audits.jsonl').read_text().splitlines()]
        assert [x['step'] for x in audits] == [2, 3]
        assert all(x['after_rd'] <= x['before_rd'] for x in audits)
        summary = json.loads((root/'direct_full/review/summary.json').read_text())
        assert summary['endpoints']['initial']['tier_counts'] == [0, 0, 16, 0]
        assert summary['endpoints']['initial']['total_channel_uses'] == summary['endpoints']['uniform_q2']['total_channel_uses']
        with tarfile.open(root/'direct_full/review.tar.gz') as tar:
            names = tar.getnames()
        assert any(n.endswith('summary.json') for n in names)
        assert not any('checkpoint' in n or 'replay' in n or n.endswith('.pt') for n in names)
        # The new checkpoint embeds inference weights and needs no old replay/optimizer file.
        (benefit/'checkpoints/latest/state.pt').rename(benefit/'old_state.pt')
        assert entry.main(partial)['status'] == 'complete'
        partial.beta = .2
        with pytest.raises(ValueError, match='mismatch'):
            entry.main(partial)


def test_actual_frozen_window_uses_hard_map_without_any_replay(tmp_path):
    entry = module()
    from gaussian_jscc.direct_rd_decisions import HardTierMap
    from gaussian_jscc.batched_spatial_window import BatchedSpatialWindow
    from test_successive_allocation import small_successive
    torch.set_num_threads(1); torch.manual_seed(11)
    raw, geometry, features, codec = small_successive(); codec.eval().requires_grad_(False)
    transform = torch.eye(4); transform[3, 2] = 3
    camera = SimpleNamespace(image_width=16, image_height=16, FoVx=1.2, FoVy=1.2,
        world_view_transform=transform, full_proj_transform=torch.eye(4), camera_center=torch.zeros(3))
    scene = SimpleNamespace(raw=raw, geometry=geometry, groups=[features], group_ids=[torch.arange(16).reshape(2, 8)],
        degree=1, white_background=False, reference=SimpleNamespace(get=lambda c,d: torch.zeros(3,16,16,device=d)))
    q = torch.full((16,), 2); q[0] = 0
    table = HardTierMap(q)
    scheduler = entry.FrozenCandidateScheduler(scene.group_ids, seed=42)
    network = entry.SpatialBenefitNetwork(hidden=16)
    inference = dict(network=network.state_dict(), scales={'value':torch.ones(2), 'rd':torch.tensor(2.)},
                     updates=7, hidden=16, dropout=.05)
    learner = entry.FrozenBenefit(inference, 'cpu')
    args = SimpleNamespace(allocation_batch=4, label_groups=0, snr=10., beta=.1, neighbors=4, cpu_threads=1)
    reference = dict(lambda_s=.01, d_ref=.01, c_ref=1000., side_proxy=dict(xyz_per_retained=24., bits_per_use=2.))
    window = BatchedSpatialWindow(codec, scene, table, scheduler, reference, args)
    result = window.collect(1, [camera], learner, entry.NoReplay(), False)
    assert torch.equal(result['q'], q)
    assert result['metrics']['labeled_groups'] == 0 and result['metrics']['teacher_seconds'] == 0
    assert torch.isfinite(result['tier_costs']).all() and len(result['tier_ids']) == 4
    assert all(not p.requires_grad for p in learner.network.parameters())


def test_launcher_dry_run_uses_one_frozen_experiment(source):
    import os
    from pathlib import Path
    import subprocess
    import sys
    entry = module()
    from gaussian_jscc import calibrated_spatial_benefit_allocation as benefit_entry
    parent, root = source; benefit = root/'launch_source'
    with patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.rendering.render', side_effect=fixture.render), \
         patch.object(benefit_entry.spatial, 'SpatialWindow', Window), \
         patch.object(benefit_entry.spatial, 'preflight', return_value={}), contextlib.redirect_stdout(io.StringIO()):
        benefit_entry.main(benefit_options(parent, benefit))
    bash = Path('C:/softwares/Git/bin/bash.exe')
    if not bash.is_file(): pytest.skip('Git Bash unavailable')
    out = root/'launcher_dry'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHON_BIN=Path(sys.executable).as_posix(), PYTHONUTF8='1')
    result = subprocess.run([str(bash), 'scripts/test_direct_rd_allocation.sh', parent.as_posix(),
        benefit.as_posix(), 'truck', out.as_posix(), '--calibration-views', '1', '--dry-run'],
        env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    plan = json.loads(result.stdout[result.stdout.index('{'):])
    assert plan['version'] == entry.FAMILY and not out.exists()
    assert plan['settings']['allocation_steps'] == 2000
    assert plan['settings']['audit_every'] == 10 and plan['settings']['accept_trials'] == 2
