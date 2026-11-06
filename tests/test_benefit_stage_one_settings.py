import copy
import json
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import benefit_stage_one as offline
from gaussian_jscc.benefit_stage_one_data import snapshot
from test_benefit_stage_one import source as synthetic_source
from test_structured_allocation_integration import source


def test_online_checkpoint_resolves_evaluator_settings_from_exported_artifact(tmp_path):
    root, path = synthetic_source(tmp_path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    settings = copy.deepcopy(state['plan']['settings'])
    settings['tie_fraction'] = .073
    state['plan']['settings'] = dict(beta=.1, network_mode='online',
        benefit_checkpoint=str(root/'benefit_network.pt'))
    state['positive_provider']['initial_network_sha256'] = 'initial-network'
    torch.save(dict(settings=settings, network_sha256='initial-network'), root/'benefit_network.pt')
    torch.save(state, path)
    saved = snapshot(root)
    assert saved['settings']['tie_fraction'] == .073
    assert saved['settings']['hidden'] == 16
    assert saved['evaluator_settings_source'].endswith('benefit_network.pt')
    args = offline.parser().parse_args(['--run', str(root), '--out', str(tmp_path/'offline'),
        '--updates', '1', '--evaluate-every', '1', '--cpu-threads', '1'])
    assert offline.main(args)['status'] == 'complete'


def test_relocated_pipeline_metadata_fallback_and_missing_config_message(tmp_path):
    root, path = synthetic_source(tmp_path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    full = copy.deepcopy(state['plan']['settings'])
    state['plan']['settings'] = dict(benefit_checkpoint='/old/server/benefit_network.pt')
    torch.save(state, path)
    with pytest.raises(ValueError, match='evaluator settings.*tie_fraction'):
        snapshot(root)
    (root/'plan.json').write_text(json.dumps(dict(settings=full)), encoding='utf8')
    saved = snapshot(root/'allocation')
    assert saved['settings']['tie_fraction'] == full['tie_fraction']
    assert saved['source_root'] == root
    assert saved['evaluator_settings_source'].endswith('plan.json')


@pytest.mark.parametrize('mismatch', ['initializer', 'width'])
def test_wrong_evaluator_metadata_is_rejected(tmp_path, mismatch):
    root, path = synthetic_source(tmp_path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    settings = copy.deepcopy(state['plan']['settings'])
    state['plan']['settings'] = dict(beta=.1)
    state['positive_provider']['initial_network_sha256'] = 'correct'
    identity = 'wrong' if mismatch == 'initializer' else 'correct'
    if mismatch == 'width':
        settings['hidden'] = 256
    torch.save(dict(settings=settings, network_sha256=identity), root/'benefit_network.pt')
    torch.save(state, path)
    with pytest.raises(ValueError, match='identity mismatch' if mismatch == 'initializer' else 'differs from checkpoint width'):
        snapshot(root)


def test_real_wide_pipeline_allocation_snapshot_supports_offline_training(source):
    from gaussian_jscc import neural_capacity_history as wide
    from test_neural_capacity_history import options
    from test_hybrid_tier_pipeline import environment
    from test_spatial_benefit_integration import Window
    from test_physical_benefit import record
    class IsolatedWindow(Window):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            scheduler = args[3]
            self.train_source = int((~scheduler.validation).nonzero()[0])
            self.validation_source = int(scheduler.validation.nonzero()[0])

        def collect(self, step, cameras, learner, replay, pretrain):
            g = record(self.train_source, .001, 158)
            g['views'][0]['inputs'] = self.inputs
            replay.add(g)
            if not replay.validation_frozen:
                replay.add(dict(g, source=self.validation_source), validation=True)
            return dict(keep_costs=torch.zeros(len(self.table.logits), 2),
                tier_ids=torch.tensor([self.train_source]), tier_costs=torch.tensor([[.1, .2, .3]]),
                q=torch.ones(len(self.table.logits), dtype=torch.long),
                metrics=dict(source_mse=.1, source_distortion=.1))
    parent, root = source
    run = root/'real_wide'
    with environment(), patch.object(wide, 'pretrain_window', IsolatedWindow):
        wide.main(options(parent, run))
    saved = snapshot(run)
    assert saved['settings']['hidden'] == 16
    assert saved['settings']['tie_fraction'] == .02
    args = offline.parser().parse_args(['--run', str(run), '--out', str(root/'real_offline'),
        '--updates', '1', '--cpu-threads', '1'])
    assert offline.main(args)['status'] == 'complete'
