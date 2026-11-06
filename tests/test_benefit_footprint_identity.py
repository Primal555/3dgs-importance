import copy

import pytest
import torch

from gaussian_jscc.benefit_footprint_collect import verify_source_artifact


def test_pretraining_online_holdout_and_initial_normalizers_must_agree():
    cfg = dict(seed=42, beta=.1, window_views=4, calibration_views=4, aligned_spatial_inputs=True)
    weights = {k: torch.ones(2) for k in ('input_mean','input_scale','context_mean','context_scale',
                                        'state_mean','state_scale','output_scale')}
    saved = dict(initial=dict(learner=dict(network=weights)))
    artifact = dict(settings=cfg, learner=dict(network=copy.deepcopy(weights)), reference=dict(beta=.1))
    verify_source_artifact(saved, artifact, cfg)
    with pytest.raises(ValueError, match='holdout|settings'):
        verify_source_artifact(saved, artifact, dict(cfg, seed=43))
    bad = copy.deepcopy(artifact); bad['learner']['network']['input_mean'] += 1
    with pytest.raises(ValueError, match='normalizer'):
        verify_source_artifact(saved, bad, cfg)
    with pytest.raises(ValueError, match='reference'):
        verify_source_artifact(saved, dict(artifact, reference=dict(beta=.5)), cfg)
