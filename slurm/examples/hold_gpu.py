""" Torch GPU Holding Script
This is a simple PyTorch DDP script, allocating identical tensors and holding the GPUs for `sleep_time`.
This script is assumed to be executed with GPUs in a single machine.
"""

import torch
import time

from utils.dlcm_handler import DLCMSlurmHandler
from utils.checkpoint import DummyDLCMJobCheckpointer

sleep_time = 100

# This script does not require an actual checkpointer, but just a Slurm preempt signal handler
# DummyDLCMJobCheckpointer is adequate for this purpose
slurm_handler = DLCMSlurmHandler(DummyDLCMJobCheckpointer()) # Do not modify

dev = torch.cuda.device_count()

for i in range(dev):
    torch.ones(sleep_time * 1024 * 4).to(f"cuda:{i}")

time.sleep(sleep_time)
print("Wake up")