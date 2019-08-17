import os
import sys
import yaml
import argparse
sys.path.append('.')

import torch

import models
import datasets

def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    model = models.__dict__[args.model['algo']](args.model, dist_model=False)

    model.load_state("{}/checkpoints".format(args.exp_path),
                          args.load_iter, resume=False)

    testset = datasets.__dict__[args.data['eval_dataset']](args.data)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    model.switch_to('eval')

    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            outputs = model.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("acc: {}".format(correct / float(total)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Kinematics')
    parser.add_argument('--config', required=True, type=str,
                        help="Experimental configuration file.")
    parser.add_argument('--load-iter', required=True, type=int,
                        help='The iteration of checkpoint to load.')
    args = parser.parse_args()

    main(args)
