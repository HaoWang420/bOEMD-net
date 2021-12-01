from torchvision import transforms
from dataloaders.brats import UncertainBraTS
from dataloaders.brats import BraTSSet
from dataloaders.lidc import LIDC_IDRI, LIDC_SYN, LIDC_IDRI_patient_id
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import numpy as np
import os
import random
import pickle

def make_data_loader(args, **kwargs):

    if args.dataset.name == 'qubiq':
        output = 'threshold'
        if args.loss.name == 'ce' or args.loss.name == 'level-thres':
            output = 'leveling'
        else:
            output = 'annotator'

        train_set = UncertainBraTS(mode='train', dataset=args.dataset.task, task=args.dataset.task_id, output=output)
        val_set = UncertainBraTS(mode='val', dataset=args.dataset.task, task=args.dataset.task_id, output=output)
        test_set = UncertainBraTS(mode='test', dataset=args.dataset.task, task=args.dataset.task_id, output=output)

        nclass = train_set.NCLASS
        nchannel = train_set.NCHANNEL
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

        return train_loader, val_loader, test_loader, nclass, nchannel, len(train_set)

    if args.dataset.name == 'brats':
        train_set = BraTSSet(args)
        val_set = BraTSSet(train=False)
        test_set = None

        nclass = train_set.NCLASS

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        test_loader = None

        return train_loader, val_loader, test_loader, nclass, len(train_set)
    
    elif args.dataset.name == 'lidc':
        nclass = 4
        nchannel = 1
        dataset = LIDC_IDRI(transform=None, mode='qubiq')

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[2*split:], indices[1*split:2*split], indices[:split]
    
    elif args.dataset.name == "lidc-small":
        nclass = 4
        nchannel = 1
        dataset = LIDC_IDRI(transform= None, mode = "qubiq")
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        indices = indices[:int(len(indices) * args.dataset.small_ratio)]
        split = int(np.floor(0.15 * dataset_size * args.dataset.small_ratio))
        np.random.shuffle(indices)
        
        train_indices, test_indices, val_indices = indices[2*split:], indices[1*split:2*split], indices[:split]
        
        
    elif args.dataset.name == 'lidc-syn':
        nclass = 3
        nchannel = 1
        dataset = LIDC_SYN(transform=None, shuffle=args.shuffle)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[8*split:], indices[1*split:2*split], indices[:split]

    elif args.dataset.name == 'lidc-syn-rand':
        nclass = 1
        nchannel = 1

        dataset = LIDC_SYN(transform=None, mode='rand')

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[8*split:], indices[1*split:2*split], indices[:split]
        
    # randomly samples a label during trainning
    elif args.dataset.name == "processed-lidc":
        nclass = 1
        nchannel = 1
        train_set = LIDC_IDRI_patient_id( transform = None, mode = 'ged', data_mode = "train")
        val_set = LIDC_IDRI_patient_id(transform=None, mode = "ged", data_mode = "val")
        test_set = LIDC_IDRI_patient_id(transform= None, mode = "ged", data_mode = "test")
        
        train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = args.workers, pin_memory = False)
        test_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.workers, pin_memory = False)
        validation_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.workers, pin_memory = False)
        
        return train_loader, validation_loader, test_loader, nclass, nchannel, len(train_set)

    elif args.dataset.name == 'lidc-rand':
        nclass = 1
        nchannel = 1
        location = '/home/wanghao/datasets/'

        dataset = LIDC_IDRI(dataset_location=location, transform=None, mode='ged')

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices, val_indices = indices[2*split:], indices[1*split:2*split], indices[:split]

    else:
        raise NotImplementedError

    print(dataset_size)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    # print(val_indices)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=False)
    test_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=False)
    validation_loader = DataLoader(dataset, batch_size=args.test_batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=False)
    print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)))

    return train_loader, validation_loader, test_loader, nclass, nchannel, len(train_indices)