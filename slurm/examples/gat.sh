#!/bin/bash

### Request n nodes w/ m gpus each

#SBATCH -J gat

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

#SBATCH --signal=USR2@11

#SBATCH --mem=1000

#SBATCH --oversubscribe

#SBATCH --output=./out/%j.out

# options
WORLD_SIZE=1 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=gnn
REQUIREMENTS="gnn_requirements.txt"

srun ./setup.sh -c $CONDA_ENV -r $REQUIREMENTS -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# Path to write and read assigned master port
# Assumption: /job_addr path is already made on HDFS
srun mkdir -p $HADOOP_DIR/job_addr
export JOB_MASTER_PORT_LOCAL_PATH=$HADOOP_DIR/job_addr
export JOB_MASTER_PORT_HDFS_PATH=/job_addr

# Path to save local checkpoint (point of resume after preemption)
srun mkdir -p $HADOOP_DIR/local_checkpoint
LOCAL_CHECKPOINT_PATH=$HADOOP_DIR/local_checkpoint/$SLURM_JOBID

# Path to save HDFS checkpoint
# Assumption: /hdfs_checkpoint path is already made on HDFS
HDFS_CHECKPOINT_DIR=/hdfs_checkpoint

# Activate conda env
. $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Profiler
# Assumption: profile iteration >= 3 (for wait & warmup)
TENSORBOARD_PATH=$HADOOP_DIR/log
declare -i PROFILE_ITERATION=3

export WORLD_SIZE=$WORLD_SIZE

srun -u python \
	$DLCM_PATH/slurm/examples/gat.py \
	--hdfs-ckpt-dir=$HDFS_CHECKPOINT_DIR \
    --profile-path=$TENSORBOARD_PATH \
	--profile-gpu \
	--profile-cpu \
	--profile-progress \
	--profile-iteration=$PROFILE_ITERATION \
	--epochs 1 \
	# --debug