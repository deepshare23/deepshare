# DeepShare

DeepShare: RL-based Network Contention-Aware GPU Cluster Manager for Deep Learning

## How to build

Requires
- Python >= 3.9
- conda
- Slurm 22.05.3
- CUDA 11.3.0
- NCCL 2.9.9
- Hadoop 3.3.4

1. Clone 
```
git clone --recurse-submodules git@github.com:deepshare23/deepshare.git
```

2. Set ENVs 
```
# DLCM
export DEEPSHARE_PATH="$HOME/deepshare"
export DEEPSHARE_CONDA_ENV="deepshare+slurm"

# SLURM
export SLURM_BUILD_PATH="$HOME/slurm_build"
export PATH="$SLURM_BUILD_PATH/bin:$PATH"
export SLURM_CONF_DIR="$DLCM_PATH/slurm/etc"
export SLURM_CONF="$SLURM_CONF_DIR/slurm.conf"

# HDFS
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export HADOOP_HOME=$HOME/hadoop-3.3.4
export HADOOP_BIN=$HADOOP_HOME/bin
export HADOOP_SBIN=$HADOOP_HOME/sbin
export HADOOP_DIR=/cmsdata/hdd0/dlcm_hdfs_dir
export PATH=$JAVA_HOME:$HADOOP_SBIN:$HADOOP_BIN:$HADOOP_DIR:$HADOOP_HOME:$PATH

# CONDA
export CONDA_HOME=$conda_home
```

```
# Add the following .hdfscli.cfg file on $HOME of all nodes that access the HDFS

[global]
default.alias = dev

[dev.alias]
url = http://[HDFS namenode IP]:9870
user = [HDFS user ID]
```

3. Set up base conda env
```
conda create -n deepshare+slurm python=3.9 -y
```

4. Set up packages (editable mode)
```
pip install -e .
cd stable-baseline3 && pip install -e .
```

This will add packages (and some misc dependencies) listed at `setup.py`
