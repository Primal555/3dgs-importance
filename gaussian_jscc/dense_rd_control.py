"""Deployment-measured LR scheduling and stable-plateau ending for dense masks."""
import math
import torch


class DeploymentPlateau:
    def __init__(self, min_steps=500, patience=3, factor=.5, min_scale=.125,
                 relative_improvement=.001, map_tolerance=.005, cost_tolerance=.002):
        values = (factor, min_scale, relative_improvement, map_tolerance, cost_tolerance)
        if (min_steps < 0 or patience < 1 or not all(math.isfinite(x) for x in values)
                or not 0 < factor < 1 or not 0 < min_scale <= 1
                or not 0 <= relative_improvement < 1 or not 0 <= map_tolerance <= 1
                or not 0 <= cost_tolerance < 1):
            raise ValueError('invalid deployment plateau settings')
        self.settings = dict(min_steps=min_steps, patience=patience, factor=factor,
            min_scale=min_scale, relative_improvement=relative_improvement,
            map_tolerance=map_tolerance, cost_tolerance=cost_tolerance)
        self.scale, self.stale, self.anchor = 1., 0, None
        self.history = []

    def observe(self, step, rd, cost, q):
        if (step < 0 or not math.isfinite(rd) or rd < 0 or not math.isfinite(cost) or cost <= 0
                or q.ndim != 1 or not len(q) or ((q < 0) | (q > 3)).any()):
            raise ValueError('invalid deployment observation')
        if self.history and (step <= self.history[-1]['step'] or len(q) != len(self.history[-1]['q'])):
            raise ValueError('observation steps must increase with the same point count')
        s = self.settings
        improved = self.anchor is None or rd < self.anchor*(1-s['relative_improvement'])
        self.history.append(dict(step=step, rd=rd, cost=cost, q=q.cpu().to(torch.uint8).clone()))
        self.history = self.history[-(s['patience']+1):]
        if improved:
            self.anchor, self.stale = rd, 0
        elif step >= s['min_steps']:
            self.stale += 1
        changes = [float((h['q'] != self.history[0]['q']).float().mean()) for h in self.history]
        costs = [h['cost'] for h in self.history]
        drift, cost_drift = max(changes), (max(costs)-min(costs))/max(costs)
        stable = (len(self.history) == s['patience']+1 and drift <= s['map_tolerance']
                  and cost_drift <= s['cost_tolerance'])
        action = 'improved' if improved else 'continue'
        if step >= s['min_steps'] and self.stale >= s['patience']:
            if self.scale > s['min_scale']:
                self.scale = max(s['min_scale'], self.scale*s['factor'])
                self.stale, action = 0, 'reduce_lr'
            elif stable:
                action = 'stop'
        return dict(action=action, lr_scale=self.scale, stale=self.stale,
                    map_drift=drift, cost_drift=cost_drift, stable=stable, anchor_rd=self.anchor)

    def state_dict(self):
        return dict(settings=self.settings, scale=self.scale, stale=self.stale,
                    anchor=self.anchor, history=self.history)

    def load_state_dict(self, state):
        if state['settings'] != self.settings: raise ValueError('plateau settings changed')
        self.scale, self.stale, self.anchor = state['scale'], state['stale'], state['anchor']
        self.history = [{**h, 'q': h['q'].cpu().clone()} for h in state['history']]
