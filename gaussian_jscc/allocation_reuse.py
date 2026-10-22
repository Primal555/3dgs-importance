"""Ephemeral allocator storage and asynchronous, non-RNG-consuming timings."""
from contextlib import contextmanager
import time

import torch


class AllocationTimings:
    """CUDA events avoid synchronizing every batch; aggregate once per step."""
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.records = []

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        else:
            start = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == 'cuda':
                end.record(torch.cuda.current_stream(self.device))
                self.records.append((name, start, end))
            else:
                self.records.append((name, start, time.perf_counter()))

    def summary(self):
        if not self.enabled:
            return {}
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        result = {}
        for name, start, end in self.records:
            seconds = start.elapsed_time(end) / 1000 if self.device.type == 'cuda' else end-start
            result[name] = result.get(name, 0.) + seconds
        return result


class AllocationStepCache:
    """Owned by one local_mask_step; replay releases each batch immediately."""
    def __init__(self, batches):
        self.batches = batches
        self.calls = 0
        self.entries = {}
        self.peak_bytes = 0

    @contextmanager
    def batch(self):
        index = self.calls % self.batches
        self.calls += 1
        entry = self.entries.setdefault(index, {})
        replay = torch.is_grad_enabled()
        try:
            yield entry
        finally:
            if replay:
                self.entries.pop(index)
            else:
                self.peak_bytes = max(self.peak_bytes, sum(
                    value.numel()*value.element_size()
                    for item in self.entries.values() for value in item.values()))

    def clear(self):
        self.entries.clear()
