import pytest
import torch
from gaussian_jscc.first_order_stopping import FirstOrderStopping


def test_plateau_reduces_once_then_stops_with_matching_resume():
    ctrl = FirstOrderStopping(min_steps=200, patience=2)
    q = torch.tensor([1, 2, 3, 0])
    assert ctrl.observe(0, 1., 100., q)['action'] == 'improved'
    assert ctrl.observe(100, .998, 100., q)['action'] == 'continue'
    assert ctrl.observe(200, .997, 100., q)['action'] == 'continue'
    assert ctrl.observe(300, .997, 100., q)['action'] == 'reduce_lr'
    state = ctrl.state_dict()
    resumed = FirstOrderStopping(min_steps=200, patience=2); resumed.load_state_dict(state)
    assert resumed.observe(300, .997, 100., q) == ctrl.last_event
    for step in (400, 500):
        assert resumed.observe(step, .997, 100., q) == ctrl.observe(step, .997, 100., q)
    assert ctrl.stopped and resumed.stopped and ctrl.scale == .5


def test_new_improvement_restarts_confirmation_and_moving_map_does_not_stop():
    ctrl = FirstOrderStopping(min_steps=0, patience=1)
    q = torch.tensor([1, 2, 3, 0])
    ctrl.observe(0, 1., 100., q)
    assert ctrl.observe(100, 1., 100., q)['action'] == 'reduce_lr'
    assert ctrl.observe(200, .98, 100., q)['action'] == 'improved'
    assert ctrl.observe(300, .98, 100., q.roll(1))['action'] == 'continue'
    assert not ctrl.stopped
    assert ctrl.observe(400, .98, 90., q.roll(1))['action'] == 'continue'
    assert ctrl.observe(500, .98, 90., q.roll(1))['action'] == 'stop'


def test_repeated_observation_must_be_identical():
    ctrl = FirstOrderStopping()
    q = torch.tensor([1])
    ctrl.observe(0, 1., 100., q)
    with pytest.raises(ValueError): ctrl.observe(0, 2., 100., q)
