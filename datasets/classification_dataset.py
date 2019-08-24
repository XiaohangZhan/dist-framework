import numpy as np
import cv2
import os
import io
import sys
sys.path.append('.')
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import utils # some data preprocessing funcs in utils/data_utils.py


# implement your dataset here

class ImageLabelDataset(Dataset):

    def __init__(self, config, phase):
        '''
        config: data config
        phase: train or val
        '''
        assert phase in ['train', 'val', 'eval']
        self.aug_transform = transforms.Compose([
            transforms.__dict__[a](*aa) for a, aa in zip(config['aug'], config['aug_args'])])
        self.nonaug_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['data_mean'], config['data_div']),
        ])
        with open(config['{}_list'.format(phase)], 'r') as f:
            lines = f.readlines()
        self.fns = [os.path.join(config['{}_root'.format(phase)], l.split()[0]) for l in lines]
        self.labels = [int(l.strip().split()[1]) for l in lines]
        self.phase = phase

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        img_fn = self.fns[idx]
        label = self.labels[idx]
        image = Image.open(img_fn).convert('RGB')

        if self.phase == 'train':
            image = self.aug_transform(image)
        else:
            image = self.nonaug_transform(image)
        image = self.img_transform(image)
        return image, label

class ImageDataset(Dataset):

    def __init__(self, config, phase):
        '''
        config: data config
        phase: train or val
        '''
        self.nonaug_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['data_mean'], config['data_div']),
        ])
        with open(config['{}_list'.format(phase)], 'r') as f:
            lines = f.readlines()
        self.fns = [os.path.join(config['{}_root'.format(phase)], l.strip()) for l in lines]

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        img_fn = self.fns[idx]
        image = Image.open(img_fn).convert('RGB')

        # transform
        image = self.nonaug_transform(image)
        image = self.img_transform(image)
        return image


def Cifar10(config, phase):
    if phase == "train":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['data_mean'], config['data_div'])])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['data_mean'], config['data_div'])])
    return torchvision.datasets.CIFAR10(root='./data', train=phase=='train',
                                        download=False, transform=transform)

def Cifar10Test(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['data_mean'], config['data_div'])])
    return torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
