#!/bin/bash

### Request n nodes w/n gpu each

#SBATCH -J toymodelddp

#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --signal=USR2@11

#SBATCH --mem=1000

#SBATCH --oversubscribe

#SBATCH --output=./out/%j.out

MASTER_ADDR="172.30.10.111"
MASTER_PORT=7777
WORLD_SIZE=4 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $DLCM_CONDA_ENV

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE

srun -u python \
    $DLCM_PATH/slurm/examples/toymodel_ddp_example.py