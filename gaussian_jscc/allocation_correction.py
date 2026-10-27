"""Sparse, action-conditioned finite-minus-tangent residual histories on CPU."""
import torch


class ActionCorrections:
    def __init__(self, stride, half_life, context_tv):
        self.stride, self.half_life, self.context_tv = stride, half_life, context_tv
        self.keys = torch.empty(0, dtype=torch.long)
        self.mean = torch.empty(0, 2, dtype=torch.float64)
        self.second = torch.empty_like(self.mean)
        self.context = torch.empty(0, 4, dtype=torch.float32)
        self.weight = torch.empty(0, dtype=torch.float64)
        self.weight2 = torch.empty_like(self.weight)
        self.step = torch.empty(0, dtype=torch.long)
        self.count = torch.empty_like(self.step)
        self.view_bits = torch.empty(0, 0, dtype=torch.long)
        self.view_count = torch.empty(0,dtype=torch.long)
        self.window_size = None

    def configure_views(self,count,window):
        self.view_bits = torch.empty(0,(count+62)//63,dtype=torch.long)
        self.window_size = window

    def query(self, keys, context, step):
        """Return matched flat query positions, attenuated residuals and errors."""
        flat = keys.detach().cpu().long().flatten()
        empty = torch.empty(0, dtype=torch.long)
        if not len(self.keys) or not len(flat): return empty, self.mean[:0], self.mean[:0]
        index = torch.searchsorted(self.keys, flat).clamp_max(len(self.keys)-1)
        pos = (self.keys[index] == flat).nonzero().flatten(); index = index[pos]
        current = context[flat[pos]//self.stride]
        drift = .5*(current-self.context[index]).abs().sum(-1).double()
        attenuation = torch.exp2(-(step-self.step[index]).double()/self.half_life)
        attenuation *= torch.exp(-drift/self.context_tv)
        mean = self.mean[index]
        ess = self.weight[index].square()/self.weight2[index].clamp_min(1e-30)
        ess = torch.minimum(ess,self.view_count[index].double()/self.window_size)
        var = (self.second[index]-mean.square()).clamp_min(0)
        # A single finite window has unmeasured cross-view uncertainty.
        error = (var/ess.clamp_min(1)[:, None]+mean.square()/(ess+1)[:, None]).sqrt()
        error = attenuation[:, None]*error+(1-attenuation[:, None])*mean.abs()
        return pos, mean*attenuation[:, None], error

    def update(self, keys, values, context, step, views):
        keys, values = keys.cpu().long(), values.detach().double().cpu()
        if not len(keys): return
        if (keys.ndim != 1 or keys.unique().numel() != len(keys) or values.shape != (len(keys), 2)
                or (keys < 0).any() or not torch.isfinite(values).all()):
            raise ValueError('invalid exact residual observations')
        union = torch.unique(torch.cat((self.keys, keys)), sorted=True)
        old_index = torch.searchsorted(union, self.keys)
        for name in ('mean', 'second', 'context', 'weight', 'weight2', 'step', 'count','view_bits','view_count'):
            old = getattr(self, name)
            new = torch.zeros((len(union), *old.shape[1:]), dtype=old.dtype)
            new[old_index] = old; setattr(self, name, new)
        self.keys = union; index = torch.searchsorted(union, keys)
        if (self.step[index] >= step).any(): raise ValueError('replayed exact residual')
        current = context[keys//self.stride]
        drift = .5*(current-self.context[index]).abs().sum(-1).double()
        decay = torch.exp2(-(step-self.step[index]).double()/self.half_life)*torch.exp(-drift/self.context_tv)
        old = self.weight[index]*decay; new = old+1
        self.mean[index] = (self.mean[index]*old[:, None]+values)/new[:, None]
        self.second[index] = (self.second[index]*old[:, None]+values.square())/new[:, None]
        self.weight[index] = new; self.weight2[index] = self.weight2[index]*decay.square()+1
        self.step[index] = step; self.count[index] += 1; self.context[index] = current
        for view in views:
            word,shift = divmod(view,63); mask = 1 << shift
            old = self.view_bits[index,word]
            self.view_count[index] += ((old & mask)==0).long()
            self.view_bits[index,word] = old | mask

    def state_dict(self):
        return {k:getattr(self, k).clone() for k in
                ('keys','mean','second','context','weight','weight2','step','count','view_bits','view_count')}

    def load_state_dict(self, state, count):
        expected = {'keys','mean','second','context','weight','weight2','step','count','view_bits','view_count'}
        if set(state) != expected: raise ValueError('incomplete calibrated correction bank')
        keys = state['keys']; n = len(keys)
        if (keys.ndim != 1 or keys.dtype != torch.long or (keys < 0).any()
                or (keys >= count*self.stride).any() or (keys[1:] <= keys[:-1]).any()):
            raise ValueError('invalid calibrated correction keys')
        for k in expected:
            value = state[k]; old = getattr(self,k)
            if (value.shape != (n,*old.shape[1:]) or value.dtype != old.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('invalid calibrated correction '+k)
        if any((state[k] < 0).any() for k in ('weight','weight2','step','count','view_bits','view_count')):
            raise ValueError('negative calibrated correction statistics')
        for k in expected: setattr(self,k,state[k].cpu().clone())
