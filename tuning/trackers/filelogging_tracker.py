# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from datetime import datetime
import json
import logging
import os
import time

# Third Party
import torch
from transformers import TrainerCallback
from accelerate import PartialState
from accelerate.utils import gather

# Local
from .tracker import Tracker
from tuning.config.tracker_configs import TrackerConfigs


class FileLoggingCallback(TrainerCallback):
    """Exports metrics, e.g., training loss to a file in the checkpoint directory."""

    training_logs_filename = "training_logs.jsonl"

    def __init__(self, logs_filename=None, enable_system_metrics=True):
        self.training_logs_filename = logs_filename
        self.enable_system_metrics = enable_system_metrics
        self.sys_metrics = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enable_system_metrics:
            return

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Checks if this log contains keys of interest, e.g., loss, and if so, creates
        training_logs.jsonl in the model output dir (if it doesn't already exist),
        appends the subdict of the log & dumps the file.
        """
        self._track_sys_metrics(args, state, control, logs)

        # All processes get the logs from this node; only update from process 0.
        if not state.is_world_process_zero:
            return

        log_file_path = os.path.join(args.output_dir, self.training_logs_filename)
        if logs is not None and "loss" in logs and "epoch" in logs:
            self._track_loss("loss", "training_loss", log_file_path, logs, state)
        elif logs is not None and "eval_loss" in logs and "epoch" in logs:
            self._track_loss("eval_loss", "validation_loss", log_file_path, logs, state)

    def _track_loss(self, loss_key, log_name, log_file, logs, state):
        try:
            # Take the subdict of the last log line; if any log_keys aren't part of this log
            # object, assume this line is something else, e.g., train completion, and skip.
            log_obj = {
                "name": log_name,
                "data": {
                    "epoch": round(logs["epoch"], 2),
                    "step": state.global_step,
                    "value": logs[loss_key],
                    "timestamp": datetime.isoformat(datetime.now()),
                },
            }
            log_obj.update(**logs)
        except KeyError:
            return

        # append the current log to the jsonl file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(log_obj, sort_keys=True)}\n")

    def _track_sys_metrics(self, args, state, control, logs):
        if not self.enable_system_metrics or state.global_step == 0:
            return

        acc_state = PartialState()

        now = time.perf_counter()
        dt = now - (self._t0 or now)
        self._t0 = now

        mem_alloc_gib = None
        mem_resv_gib = None
        if torch.cuda.is_available():
            mem_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3)
            mem_resv_gib = torch.cuda.max_memory_reserved() / (1024**3)

            acc = acc_state
            a = torch.tensor([mem_alloc_gib], device=acc.device, dtype=torch.float64)
            r = torch.tensor([mem_resv_gib], device=acc.device, dtype=torch.float64)

            mem_alloc_gib = float(gather(a).max().item())
            mem_resv_gib = float(gather(r).max().item())

            torch.cuda.reset_peak_memory_stats()

        if logs:
            sys_metrics  = {
                "time_per_steps": dt,
                "peak_mem_alloc_gib": mem_alloc_gib if mem_alloc_gib is not None else -1,
                "peak_mem_reserved_gib": mem_resv_gib if mem_resv_gib is not None else -1
            }
            logs.update(**sys_metrics)


class FileLoggingTracker(Tracker):
    def __init__(self, tracker_config: TrackerConfigs):
        """Tracker which encodes callback to record metric, e.g., training loss
        to a file in the checkpoint directory.

        Args:
            tracker_config (FileLoggingTrackerConfig): An instance of file logging tracker
                which contains the location of file where logs are recorded.
        """
        super().__init__(name="file_logger", tracker_config=tracker_config)
        # Get logger with root log level
        self.logger = logging.getLogger()

    def get_hf_callback(self):
        """Returns the FileLoggingCallback object associated with this tracker.

        Returns:
            FileLoggingCallback: The file logging callback which inherits
                transformers.TrainerCallback and records the metrics to a file.
        """
        file = self.config.training_logs_filename
        self.hf_callback = FileLoggingCallback(logs_filename=file)
        return self.hf_callback
