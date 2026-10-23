"""A nonnegative dual price driven only by measured deployment cost."""
import math


class BudgetController:
    def __init__(self, budget, initial_price=.01, dual_lr=.05, ema_decay=.9):
        if (not math.isfinite(budget) or budget <= 0 or not math.isfinite(initial_price)
                or initial_price < 0 or not math.isfinite(dual_lr) or dual_lr <= 0
                or not math.isfinite(ema_decay) or not 0 <= ema_decay < 1):
            raise ValueError('invalid budget controller settings')
        self.budget, self.price = int(budget), float(initial_price)
        if self.budget < 1: raise ValueError('budget must contain at least one channel use')
        self.dual_lr, self.ema_decay = float(dual_lr), float(ema_decay)
        self.ema, self.observations = None, 0

    def observe(self, actual_uses):
        if not math.isfinite(actual_uses) or actual_uses < 0:
            raise ValueError('invalid actual deployment cost')
        residual = actual_uses / self.budget - 1
        self.ema = residual if self.ema is None else self.ema_decay*self.ema + (1-self.ema_decay)*residual
        self.price = max(0., self.price + self.dual_lr*self.ema)
        self.observations += 1
        return dict(actual_uses=actual_uses, budget=self.budget, residual=residual,
                    residual_ema=self.ema, price=self.price, observations=self.observations)

    def state_dict(self):
        return dict(budget=self.budget, price=self.price, dual_lr=self.dual_lr,
                    ema_decay=self.ema_decay, ema=self.ema, observations=self.observations)

    def load_state_dict(self, state):
        candidate = BudgetController(state['budget'], state['price'], state['dual_lr'], state['ema_decay'])
        if state['ema'] is not None and not math.isfinite(state['ema']): raise ValueError('invalid EMA')
        if state['observations'] < 0: raise ValueError('invalid controller count')
        candidate.ema, candidate.observations = state['ema'], state['observations']
        self.__dict__.update(candidate.__dict__)
