import time
import torch
from transformers import TrainerCallback

class PerfCallback(TrainerCallback):
    def __init__(self, accelerator=None, log_every_steps=1):
        self.accelerator = accelerator
        self.log_every_steps = log_every_steps
        self._t0 = None
        self.perf_metrics = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.perf_counter()
        dt = now - (self._t0 or now)
        self._t0 = now

        if self.accelerator is not None and torch.cuda.is_available():
            mem_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3)
            mem_resv_gib = torch.cuda.max_memory_reserved() / (1024**3)

            a = torch.tensor([mem_alloc_gib], device=self.accelerator.device, dtype=torch.float64)
            r = torch.tensor([mem_resv_gib], device=self.accelerator.device, dtype=torch.float64)
            mem_alloc_gib = float(self.accelerator.gather(a).max().item())
            mem_resv_gib = float(self.accelerator.gather(r).max().item())

            torch.cuda.reset_peak_memory_stats()

        if logs:
            sys_metrics  = {
                "time_per_steps": dt,
                "peak_mem_alloc_gib": mem_alloc_gib if mem_alloc_gib is not None else -1,
                "peak_mem_reserved_gib": mem_resv_gib if mem_resv_gib is not None else -1
            }
            logs.update(**sys_metrics)
