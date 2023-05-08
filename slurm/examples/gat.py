# GAT training with Cora dataset (configurable)
# Source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py

import argparse
import logging
import os
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv

from utils.dlcm_handler import DLCMSlurmHandler
from utils.checkpoint import DummyDLCMJobCheckpointer

_logger = logging.getLogger('GAT Training')


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def _parse_args():
    parser = argparse.ArgumentParser(
        description="GAT Training"
    )
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
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
    parser.add_argument('--profile', action='store_true', default=False,
                    help='profile with torch profiler')
    parser.add_argument('--profile-iteration', type=int, default=3,
                    help='batch indices to profile with torch profiler')
    parser.add_argument('--tensorboard-path', default=f'./out/log/{os.environ["SLURM_JOBID"]}', type=str,
                    help='path to write tensorboard logs to.')
    parser.add_argument('--world-size', type=str, default='1')

    return parser.parse_args()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
            hidden_channels=args.hidden_channels, lr=args.lr, device=device)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    ## GAT
    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Load checkpoint if exists
    # if args.resume:
    #     if os.environ['SLURM_PROCID'] == '0':
    #         resume_epoch = slurm_handler.load_dlcm_job(
    #             model,
    #             optimizer=optimizer)

    # slurm_handler.checkpointer.setup(model, optimizer)

    # Compute epoch
    # start_epoch = 0
    # num_epochs = args.epochs
    # if args.start_epoch is not None:
    #     # a specified start_epoch will always override the resume epoch
    #     start_epoch = args.start_epoch
    # elif resume_epoch is not None:
    #     start_epoch = resume_epoch

    try:
        # Train the model
        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            # slurm_handler.update_epoch(epoch)
            # TODO: fix the below train() based on the model.train()
            loss = train(model, optimizer, data)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
        # for epoch in range(start_epoch, num_epochs):
        #     train_one_epoch(epoch, trainer, train_data_loader, args, slurm_handler=slurm_handler)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args = _parse_args()
    # Install slurm handler and torch profiler
    slurm_handler = DLCMSlurmHandler(DummyDLCMJobCheckpointer()) # TODO: add GNN checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)
    if args.debug:
        _logger.info(f'Setting log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)
    main(args)
