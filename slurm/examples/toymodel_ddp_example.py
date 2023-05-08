""" ToyModel Training Script
This is a simple ToyModel DDP training script, with assuming Slurm to launch as sbatch job. Such assumption can
be observed from usage of SLURM_xx environment variables. 
This script is a modification from PyTorch DDP example
(https://github.com/pytorch/examples/tree/main/distributed/ddp)
"""

""" Modified for Slurm
- L43 Each process is given local rank 0 since `toymodel_ddp_example.sh` assigns 1 gpu per task. This is different
        from `local_rank` in L78, which corresponds to the actual intra machine local rank
- L73 Each process is given rank = `SLURM_PROCID` since Slurm assigns global rank with this environment variable
"""

import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dlcm_handler import DLCMSlurmHandler
from utils.checkpoint import DummyDLCMJobCheckpointer

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():    
    local_rank = 0

    model = ToyModel().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    epochs = 1000000

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(local_rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        time.sleep(1)


def spmd_main(slurm_handler: DLCMSlurmHandler):
    # Parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE")
    }
    
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  

    # Slurm assigns unique procid to each process
    # Assumption: the number of total processes = world size (https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
    dist.init_process_group(backend="gloo", rank=int(os.environ['SLURM_PROCID']), world_size=int(env_dict["WORLD_SIZE"]))

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, "
        + f"local_rank = {int(os.environ['SLURM_LOCALID'])}\n", end=''
    )

    demo_basic()

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # This script does not require an actual checkpointer, but just a Slurm preempt signal handler
    # DummyDLCMJobCheckpointer is adequate for this purpose
    slurm_handler = DLCMSlurmHandler(DummyDLCMJobCheckpointer()) # Do not modify 

    # The main entry point is called directly without using subprocess
    spmd_main(slurm_handler)