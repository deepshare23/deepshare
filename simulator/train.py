import argparse
from datetime import datetime
import logging
import glob
import os
from rl_training.policy_factory import PolicyFactory
from cluster_env import ClusterEnv

from datetime import datetime
date_time_ymd = datetime.now().strftime("%Y-%m-%d")

# DLCM: Initialized global `_logger` and `args`.
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
args = None

def _parse_args():
    parser = argparse.ArgumentParser(description='GPU scheduler simulator')

    parser.add_argument('--nodes', type=int, default=4,
                        help='number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=8,
                        help='number of GPUs per node')
    parser.add_argument("--job-trace", default=f"{os.environ['DLCM_PATH']}/simulator/traces/train", type=str, metavar="PATH",
                        help="path to job trace file")
    parser.add_argument('--total-jobsets', type=int, default=20,
                        help='Total number of jobsets to train')
    parser.add_argument('--episodes-per-jobset', type=int, default=20,
                        help='Total number of jobsets to train')
    parser.add_argument('--round-dur', type=int, default=300,
                        help='Length of one scheduling round in seconds')
    parser.add_argument('--rl-algo', type=str, default='PPO',
                        help='RL algorithms to use in training')
    parser.add_argument('--coeff-cont', type=float, default=0.8,
                        help='Weighting factor for performance degradation due to contention term in reward')
    parser.add_argument('--coeff-util', type=float, default=0.2,
                        help='Weighting factor for cluster-wide utilization in reward')
    parser.add_argument("--ckpt-dir", default=f"{os.environ['DLCM_PATH']}/simulator/training-ckpt/{date_time_ymd}", type=str, metavar="PATH",
                        help="path to policy checkpoints")
    parser.add_argument('--load_from_ckpt', action='store_true', default=False,
                        help='Whether to load the policy from the checkpoint file')

    return parser.parse_args()


def cleanup():
    extensions = ["*.log"]
    for e in extensions:
        for f in glob.glob(e):
            os.remove(f)
            _logger.info(f'Removed {f}')


if __name__ == '__main__':
    args = _parse_args()
    cleanup()

    ckpt_path_exists = os.path.exists(args.ckpt_dir)
    if not ckpt_path_exists:
        os.makedirs(args.ckpt_dir)
        _logger.info(f'{args.ckpt_dir} created')

    # Init environment
    env = ClusterEnv(args.nodes, args.gpus_per_node, rl_algo=args.rl_algo, coeff_cont=args.coeff_cont, coeff_util=args.coeff_util, total_jobsets=args.total_jobsets, episodes_per_jobset=args.episodes_per_jobset, round_dur=args.round_dur, trace=args.job_trace, ckpt_dir=args.ckpt_dir)

    # Create RL policy with rl_algo
    policy = PolicyFactory().create_policy(env, args.rl_algo)

    # Setup and start training
    env.start(model=policy)
