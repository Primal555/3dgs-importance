"""Calibration-only, resumable plateau confirmation for first-order allocation."""
import torch
from .dense_rd_control import DeploymentPlateau


class FirstOrderStopping(DeploymentPlateau):
    def __init__(self, min_steps=1250, patience=2, relative_improvement=.005,
                 map_tolerance=.005, cost_tolerance=.005):
        super().__init__(min_steps, patience, factor=.5, min_scale=.5,
                         relative_improvement=relative_improvement,
                         map_tolerance=map_tolerance, cost_tolerance=cost_tolerance)
        self.last_event = None
        self.stopped = False

    def observe(self, step, rd, cost, q):
        if self.history and step == self.history[-1]['step']:
            last = self.history[-1]
            if rd != last['rd'] or cost != last['cost'] or not torch.equal(q.cpu(), last['q']):
                raise ValueError('repeated calibration observation changed')
            return self.last_event
        event = super().observe(step, rd, cost, q)
        self.stopped = event['action'] == 'stop'
        self.last_event = dict(event, step=step, calibration_rd=rd, total_channel_uses=cost)
        return self.last_event

    def state_dict(self):
        return dict(super().state_dict(), last_event=self.last_event, stopped=self.stopped)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.last_event, self.stopped = state['last_event'], state['stopped']
