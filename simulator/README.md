# DLCM Simulator 

## Prerequisites
Install the required packages. 
```
pip install -r requirements.txt
```

## Simulation
```
python3 ./runner.py [--nodes NODES] [--gpus-per-node GPUS_PER_NODE] \
        [--job-trace JOB_TRACE] [--scheduler SCHEDULER] \
        [--round-duration ROUND_DURATION] [--contention-aware] \
        [--contention-sensitivity-knob KNOB] [--debug]
```