import os
import cv2
import time
import numpy as np

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils
import datasets
import pdb

class Trainer(object):

    def __init__(self, args):

        # get rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('{}/events'.format(args.exp_path)):
                os.makedirs('{}/events'.format(args.exp_path))
            if not os.path.exists('{}/images'.format(args.exp_path)):
                os.makedirs('{}/images'.format(args.exp_path))
            if not os.path.exists('{}/logs'.format(args.exp_path)):
                os.makedirs('{}/logs'.format(args.exp_path))
            if not os.path.exists('{}/checkpoints'.format(args.exp_path)):
                os.makedirs('{}/checkpoints'.format(args.exp_path))

            # logger
            if args.trainer['tensorboard'] and not (args.extract or args.evaluate):
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(
                    args.exp_path))
            else:
                self.tb_logger = None
            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(args.exp_path))
            elif args.extract:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_extract.txt'.format(args.exp_path))
            elif args.evaluate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_evaluate.txt'.format(args.exp_path))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(args.exp_path))

        # create model
        self.model = models.__dict__[args.model['algo']](
            args.model, load_path=args.load_path, dist_model=True)

        # optionally resume from a checkpoint
        assert not (args.load_iter is not None and args.load_path is not None)
        if args.load_iter is not None:
            self.model.load_state("{}/checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0


        self.curr_step = self.start_iter

        # lr scheduler & datasets
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]
        eval_class = (None if args.data['eval_dataset']
                      is None else datasets.__dict__[args.data['eval_dataset']])
        extract_class = (None if args.data['extract_dataset']
                         is None else datasets.__dict__[args.data['extract_dataset']])

        if not (args.validate or args.extract or args.evaluate):  # train
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim,
                args.model['lr_steps'],
                args.model['lr_mults'],
                args.model['lr'],
                args.model['warmup_lr'],
                args.model['warmup_steps'],
                last_iter=self.start_iter - 1)

            train_dataset = trainval_class(args.data, 'train')
            train_sampler = utils.DistributedGivenIterationSampler(
                train_dataset,
                args.model['total_iter'],
                args.data['batch_size'],
                last_iter=self.start_iter - 1)
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=args.data['batch_size'],
                                           shuffle=False,
                                           num_workers=args.data['workers'],
                                           pin_memory=False,
                                           sampler=train_sampler)

        if not (args.extract or args.evaluate):  # train or offline validation
            val_dataset = trainval_class(args.data, 'val')
            val_sampler = utils.DistributedSequentialSampler(val_dataset)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.data['batch_size_val'],
                shuffle=False,
                num_workers=args.data['workers'],
                pin_memory=False,
                sampler=val_sampler)

        if not (args.validate or args.extract) and eval_class is not None: # train or offline evaluation
            eval_dataset = eval_class(args.data)
            assert len(eval_dataset) % (self.world_size * args.data['batch_size_eval']) == 0, \
                "Otherwise the padded samples will be involved twice."
            eval_sampler = utils.DistributedSequentialSampler(
                eval_dataset)
            self.eval_loader = DataLoader(eval_dataset,
                                          batch_size=args.data['batch_size_eval'],
                                          shuffle=False,
                                          num_workers=1,
                                          pin_memory=False,
                                          sampler=eval_sampler)

        if args.extract:  # extract
            assert extract_class is not None, 'Please specify extract_dataset'
            extract_dataset = extract_class(args.data, 'extract')
            extract_sampler = utils.DistributedSequentialSampler(
                extract_dataset)
            self.extract_loader = DataLoader(extract_dataset,
                                            batch_size=args.data['batch_size_extract'],
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=False,
                                             sampler=extract_sampler)

        self.args = args

    def run(self):

        assert self.args.validate + self.args.extract + self.args.evaluate < 2

        # offline validate
        if self.args.validate:
            self.validate('off_val')
            return

        # extract
        if self.args.extract:
            assert self.args.extract_output.endswith('.bin'), "extraction output file must end with .bin"
            self.extract()
            return

        # evaluate
        if self.args.evaluate:
            self.evaluate('off_eval')
            return

        if self.args.trainer['initial_val']:
            self.validate('on_val')

        if self.args.trainer['eval'] and  self.args.trainer['initial_eval']:
            self.evaluate('on_eval')

        # train
        self.train()

    def train(self):

        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('train')

        end = time.time()
        for i, inputs in enumerate(self.train_loader):
            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            loss_dict = self.model.step()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item() / self.world_size)

            btime_rec.update(time.time() - end)
            end = time.time()

            self.curr_step += 1

            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer[
                    'print_freq'] == 0:
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k),
                                                  recorder[k].avg,
                                                  self.curr_step)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step,
                                               len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec) + loss_str +
                    'lr {lr:.2g}'.format(lr=curr_lr))

            # save
            if (self.rank == 0 and
                (self.curr_step % self.args.trainer['save_freq'] == 0 or
                 self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/checkpoints".format(self.args.exp_path),
                    self.curr_step)

            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or
                self.curr_step == self.args.model['total_iter']):
                self.validate('on_val')

            if ((self.curr_step % self.args.trainer['eval_freq'] == 0 or
                self.curr_step == self.args.model['total_iter'])) and self.args.trainer['eval']:
                self.evaluate('on_eval')

            

    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')

        end = time.time()
        all_together = []
        for i, inputs in enumerate(self.val_loader):
            if ('val_iter' in self.args.trainer and
                    self.args.trainer['val_iter'] != -1 and
                    i == self.args.trainer['val_iter']):
                break

            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            tensor_dict, loss_dict = self.model.forward()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item() / self.world_size)
            btime_rec.update(time.time() - end)
            end = time.time()

            # tb visualize
            if self.rank == 0:
                disp_start = max(self.args.trainer['val_disp_start_iter'], 0)
                disp_end = min(self.args.trainer['val_disp_end_iter'], len(self.val_loader))
                if (i >= disp_start and i < disp_end):
                    all_together.append(
                        utils.visualize_tensor(tensor_dict['common_tensors'],
                            self.args.data['data_mean'], self.args.data['data_div']))
                if (i == disp_end - 1 and disp_end > disp_start):
                    all_together = torch.cat(all_together, dim=2)
                    grid = vutils.make_grid(all_together,
                                            nrow=1,
                                            normalize=True,
                                            range=(0, 255),
                                            scale_each=False)
                    if self.tb_logger is not None:
                        self.tb_logger.add_image('Image_' + phase, grid,
                                                 self.curr_step)

                    cv2.imwrite("{}/images/{}_{}.png".format(
                        self.args.exp_path, phase, self.curr_step),
                        grid.permute(1, 2, 0).numpy())

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_val':
                    self.tb_logger.add_scalar('val_{}'.format(k),
                                              recorder[k].avg,
                                              self.curr_step)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                    k, loss=recorder[k])

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + loss_str)

        self.model.switch_to('train')

    def extract(self): # feature extraction
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        self.model.switch_to('eval')
        end = time.time()
        tensors_proc = []
        for i, inputs in enumerate(self.extract_loader):
            dtime_rec.update(time.time() - end)
            self.model.set_input(*inputs)

            tensors_proc.append(self.model.extract().cpu().numpy())

            btime_rec.update(time.time() - end)
            end = time.time()

            # logging
            if self.rank == 0:
                self.logger.info(
                    'Extract Iter: [{0}] ({1}/{2})\t'.format(
                        self.curr_step, i, len(self.extract_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec))

        tensors_proc = np.concatenate(tensors_proc, axis=0)
        tensors_gathered = utils.gather_tensors_batch(tensors_proc, part_size=20)
        if self.rank == 0:
            tensors_output = np.concatenate(
                tensors_gathered, axis=0)[:len(self.extract_loader.dataset)]
            if not os.path.isdir(os.path.dirname(self.args.extract_output)):
                os.makedirs(os.path.dirname(self.args.extract_output))
            tensors_output.tofile(self.args.extract_output)

        self.model.switch_to('train')

    def evaluate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['eval_record']:
            recorder[rec] = utils.AverageMeter()
        self.model.switch_to('eval')
        end = time.time()
        for i, inputs in enumerate(self.eval_loader): # padded samples will be evaluted twice.
            dtime_rec.update(time.time() - end)
            self.model.set_input(*inputs)

            eval_dict = self.model.evaluate()
            for k in eval_dict.keys():
                recorder[k].update(utils.reduce_tensors(eval_dict[k]).item() / self.world_size)

            btime_rec.update(time.time() - end)
            end = time.time()

        # logging
        if self.rank == 0:
            eval_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_eval':
                    self.tb_logger.add_scalar('eval_{}'.format(k),
                                              recorder[k].avg,
                                              self.curr_step)
                eval_str += '{}: {value.avg:.5g}\t'.format(
                    k, value=recorder[k])

            self.logger.info(
                'Evaluation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + eval_str)

        self.model.switch_to('train')
