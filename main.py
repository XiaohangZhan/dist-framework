import multiprocessing as mp
import argparse
import os
import yaml

from utils import dist_init
from trainer import Trainer


def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)

    # exp path
    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    # dist init
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    dist_init(args.launcher, backend='nccl')

    # train
    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Kinematics')
    parser.add_argument('--config', required=True, type=str,
                        help="Experimental configuration file.")
    parser.add_argument('--launcher', default='pytorch', type=str,
                        help="pytorch or slurm")
    parser.add_argument('--load-path', default=None, type=str,
                        help='Pre-trained weights of the backbone CNN')
    parser.add_argument('--load-iter', default=None, type=int,
                        help='The iteration of checkpoint to load.')
    parser.add_argument('--resume', action='store_true',
                        help="Resume from load-iter or not.")
    parser.add_argument('--validate', action='store_true',
                        help="Offline validation.")
    parser.add_argument('--extract', action='store_true',
                        help="Offline feature extraction.")
    parser.add_argument('--evaluate', action='store_true',
                        help="Offline evaluation.")
    parser.add_argument('--local_rank', type=int, default=0,
                        help="To be specified for different nodes.")
    args = parser.parse_args()

    main(args)
