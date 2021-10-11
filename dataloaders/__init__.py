from torchvision import transforms
from dataloaders.brats import UncertainBraTS
from dataloaders.brats import BraTSSet
from dataloaders.lidc import LIDC_IDRI, LIDC_SYN
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import numpy as np
import os
import random
import pickle

def make_data_loader(args, **kwargs):
    output = 'threshold'
    if args.loss_type == 'ce' or args.loss_type == 'level-thres':
        output = 'leveling'
    else:
        output = 'annotator'

    if args.dataset == 'uncertain-brats':
        train_set = UncertainBraTS(mode='train', dataset='brain-tumor', task=args.task_num, output=output)
        val_set = UncertainBraTS(mode='val', dataset='brain-tumor', task=args.task_num, output=output)
        test_set = UncertainBraTS(mode='test', dataset='brain-tumor', task=args.task_num, output=output)

        nclass = train_set.NCLASS
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, nclass, len(train_set)
    if args.dataset == 'uncertain-brain-growth':
        train_set = UncertainBraTS(mode='train', dataset='brain-growth', task=0, output=output)
        val_set = UncertainBraTS(mode='val', dataset='brain-growth', task=0, output=output)
        test_set = UncertainBraTS(mode='test', dataset='brain-growth', task=0, output=output)

        nclass = train_set.NCLASS
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, nclass, len(train_set)

    if args.dataset == 'uncertain-kidney':
        train_set = UncertainBraTS(mode='train', dataset='kidney', task=0, output=output)
        val_set = UncertainBraTS(mode='val', dataset='kidney', task=0, output=output)
        test_set = UncertainBraTS(mode='test', dataset='kidney', task=0, output=output)

        nclass = train_set.NCLASS
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, nclass, len(train_set)

    if args.dataset == 'uncertain-prostate':
        train_set = UncertainBraTS(mode='train', dataset='prostate', task=args.task_num, output=output)
        val_set = UncertainBraTS(mode='val', dataset='prostate', task=args.task_num, output=output)
        test_set = UncertainBraTS(mode='test', dataset='prostate', task=args.task_num, output=output)

        nclass = train_set.NCLASS
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, nclass, len(train_set)

    if args.dataset == 'brats':
        train_set = BraTSSet(args)
        val_set = BraTSSet(train=False)
        test_set = None

        nclass = train_set.NCLASS

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, nclass, len(train_set)
    
    if args.dataset == 'lidc':
        nclass = 4
        dataset = LIDC_IDRI(transform=None, mode='qubiq')

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[2*split:], indices[1*split:2*split], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=test_sampler)
        validation_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=val_sampler)
        print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)))

        return train_loader, test_loader, validation_loader, nclass, len(train_indices)

    if args.dataset == 'lidc-syn':
        nclass = 3
        dataset = LIDC_SYN(transform=None, shuffle=args.shuffle)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[8*split:], indices[1*split:2*split], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=test_sampler)
        validation_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=val_sampler)
        print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)))

        return train_loader, validation_loader, test_loader, nclass, len(train_indices)

    if args.dataset == 'lidc-syn-rand':
        nclass = 1

        dataset = LIDC_SYN(transform=None, mode='rand')

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[8*split:], indices[1*split:2*split], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=test_sampler)
        validation_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=val_sampler)
        print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)))

        return train_loader, validation_loader, test_loader, nclass, len(train_indices)
        
    # randomly samples a label during trainning
    if args.dataset == 'lidc-rand':
        nclass = 1
        location = '/home/wanghao/datasets/'

        dataset = LIDC_IDRI(dataset_location=location, transform=None, mode='ged')

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[2*split:], indices[1*split:2*split], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=test_sampler)
        validation_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=val_sampler)
        print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)))

        return train_loader, validation_loader, test_loader, nclass, len(train_indices)
    
    raise NotImplementedError
