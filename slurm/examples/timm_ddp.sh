#!/bin/bash

### Request n nodes w/ m gpus each

#SBATCH -J timmddp

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

#SBATCH --signal=USR2@11

#SBATCH --mem=1000

#SBATCH --oversubscribe

#SBATCH --output=./out/%j.out

# options
WORLD_SIZE=2 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=timm_ddp 
REQUIREMENTS="timm_requirements.txt"

srun ./setup.sh -c $CONDA_ENV -r $REQUIREMENTS

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
TENSORBOARD_PATH=$HADOOP_DIR/log/$SLURM_JOBID
srun mkdir -p $TENSORBOARD_PATH
declare -i PROFILE_ITERATION=3

export WORLD_SIZE=$WORLD_SIZE

srun -u python \
	$DLCM_PATH/slurm/examples/timm_ddp.py \
	/cmsdata/ssd0/cmslab/imagenet-pytorch \
	--model=mobilenetv3_small_075 --batch-size=16 \
	--accumulate-iteration=1 \
	--resume=$LOCAL_CHECKPOINT_PATH \
	--hdfs-ckpt-dir=$HDFS_CHECKPOINT_DIR \
	--profile-path=$TENSORBOARD_PATH \
	--profile-gpu \
	--profile-cpu \
	--profile-progress \
	--profile-iteration=$PROFILE_ITERATION \
	# --debug