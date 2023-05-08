# GCN training with Cora dataset (configurable)
# Source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py

import argparse
import logging
import os
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

from utils.dlcm_handler import DLCMSlurmHandler
from utils.checkpoint import DummyDLCMJobCheckpointer


# DLCM: Initialized global `_logger` and `args`.
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('train')
args = None


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# TODO: correctness check on receiving model, optimizer, data args
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

# TODO: correctness check on receiving model, data args
@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="GCN Training"
    )
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')    
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true', help='Track experiment')

    # Custom argument added for DLCM
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--debug', action='store_true', default=False,
                    help='set log level to loggig.debug')
    parser.add_argument('--hdfs-ckpt-dir', default='', type=str, metavar='PATH',
                    help='HDFS directory to store checkpoint (default: none)')
    parser.add_argument('--accumulate-iteration', type=int, default=1,
                    help='batch interval to accumulate gradient without synchronization')
    parser.add_argument('--profile-gpu', action='store_true', default=False,
                        help='profile with torch profiler')
    parser.add_argument('--profile-cpu', action='store_true', default=False,
                        help='profile CPU util')
    parser.add_argument('--profile-progress', action='store_true', default=False,
                        help='profile progres')
    parser.add_argument('--profile-iteration', type=int, default=3,
                        help='batch indices to profile with torch profiler')
    parser.add_argument('--profile-path', default=f'./out/log/{os.environ["SLURM_JOBID"]}', type=str,
                        help='path to write profiler logs to.')
    parser.add_argument('--data-path', default='/cmsdata/ssd0/cmslab/dlcm_data/graph-data/Reddit', type=str, metavar='PATH',
                    help='path to graph data')

    args = parser.parse_args()

    return args


def main(slurm_handler, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Planetoid for training dataset
    # init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
    #         hidden_channels=args.hidden_channels, device=device)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    # dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    dataset = Reddit(args.data_path)

    data = dataset[0]
    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)

    # Initialize GCN model and optimizer
    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)  # Only perform weight-decay on first convolution.

    ### Install DLCM profiler
    if args.profile_gpu:
        slurm_handler.profiler.start_gpu_profile()
    if args.profile_cpu:
        slurm_handler.profiler.start_cpu_profile()
    if args.profile_progress:
        pass

    # Actual training loop
    for epoch in range(args.epochs):

        batch_start_time = time.time()

        loss = train(model, optimizer, data)

        ### Notify DLCM profiler about iteration training status
        batch_time = time.time() - batch_start_time
        slurm_handler.profiler.step(samples=len(data), bt=batch_time)


if __name__ == '__main__':
    args = _parse_args()

    # DLCM: Setup log level
    if args.debug:
        _logger.info(f'Set log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)

    # Install slurm handler and torch profiler
    slurm_handler = DLCMSlurmHandler(DummyDLCMJobCheckpointer()) # TODO: add GNN checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)
    if args.debug:
        _logger.info(f'Setting log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)

    main(slurm_handler, args)
