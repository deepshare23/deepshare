#!/bin/bash

#SBATCH -J fsdp

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1

#SBATCH --signal=USR2@11

#SBATCH --mem-per-gpu=10000
#SBATCH --cpus-per-task=3

#SBATCH --oversubscribe

#SBATCH --output=/home/gajagajago/dlcm/slurm/examples/out/%j.out

# options
declare -i WORLD_SIZE=4 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=fairscale
REQUIREMENTS="fairscale_requirements.txt"

srun $DLCM_PATH/slurm/examples/setup.sh -c $CONDA_ENV -r $DLCM_PATH/slurm/examples/$REQUIREMENTS

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
declare -i PROFILE_ITERATION=4*$WORLD_SIZE

export WORLD_SIZE=$WORLD_SIZE

# Inter-job worker waiting
declare -i WAIT_WORKERS=2 # 1 more than my workers per node
POLLING_FILE_PATH=$DLCM_PATH/slurm/examples/out/ready
srun truncate -s 0 $POLLING_FILE_PATH

export NCCL_IB_DISABLE=1
srun -u python3 \
    $DLCM_PATH/slurm/examples/fsdp.py \
    --profile-path=$TENSORBOARD_PATH \
	--profile-progress \
	--profile-iteration=$PROFILE_ITERATION \
    --data-path=/cmsdata/ssd0/cmslab/dlcm_data/Fairscale-data \
    --dataset=wikitext-2 \
    --optimizer=sgd \
    --flatten_parameters \
    --full_fp16 \
	--wait-workers=$WAIT_WORKERS \
	--polling-file-path=$POLLING_FILE_PATH \
    # --debug