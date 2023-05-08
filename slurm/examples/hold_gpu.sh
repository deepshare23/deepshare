#!/bin/bash

### Request n nodes w/n gpu each

#SBATCH -J holdgpu

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --signal=USR2@11

#SBATCH --mem=1000

#SBATCH --oversubscribe

#SBATCH --output=./out/%j.out

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $DLCM_CONDA_ENV
pip uninstall -y dlcm && pip install $DLCM_PATH

srun -u python \
	$DLCM_PATH/slurm/examples/hold_gpu.py 