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

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_every_steps != 0:
            return

        now = time.perf_counter()
        dt = now - (self._t0 or now)

        mem_alloc_gib = None
        mem_resv_gib = None
        if torch.cuda.is_available():
            mem_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3)
            mem_resv_gib = torch.cuda.max_memory_reserved() / (1024**3)


        if self.accelerator is not None and torch.cuda.is_available():
            a = torch.tensor([mem_alloc_gib], device=self.accelerator.device, dtype=torch.float64)
            r = torch.tensor([mem_resv_gib], device=self.accelerator.device, dtype=torch.float64)
            mem_alloc_gib = float(self.accelerator.gather(a).max().item())
            mem_resv_gib = float(self.accelerator.gather(r).max().item())

        self.perf_metrics  = {f"time_per_{self.log_every_steps}_steps": dt}

        if mem_alloc_gib is not None:
            self.perf_metrics.update({
                "peak_mem_alloc_gib": mem_alloc_gib,
                "peak_mem_reserved_gib": mem_resv_gib
            }) # type: ignore

        self._t0 = now
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if hasattr(self, 'perf_metrics') and (self.accelerator is None or self.accelerator.is_main_process):
            logs.update(self.perf_metrics)
            self.perf_metrics = {}

            if state.log_history:
                state.log_history[-1].update(self.perf_metrics)

        # if self.accelerator.is_main_process:
        #     import pdb
        #     pdb.set_trace()