import os
from os import scandir
from re import sub

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from dataloaders.myTransform import RandomHorizontalFlip
from dataloaders.myTransform import RandomVerticalFlip

from mypath import Path

NANNOTATOR = {
        'brain-tumor': 3,
        'brain-growth': 7,
        'kidney': 3,
        'prostate': 6,
        }
NCLASS = {
        'brain-tumor': 3,
        'brain-growth': 1,
        'kidney': 1,
        'prostate': 2,
        
        }
MEANS = {
        'brain-tumor': 501.699,
        'brain-growth': 569.1935,
        'kidney': -383.4882,
        'prostate': 429.64
        }

STDS = {
        'brain-tumor': 252.760,
        'brain-growth': 189.9989,
        'kidney': 472.0944,
        'prostate': 300.0039
        }
NCHANNEL = {
        'brain-tumor': 4,
        'brain-growth': 1,
        'kidney': 1,
        'prostate': 1
}


class UncertainBraTS(torch.utils.data.Dataset):
    """
    Each y label has 4 channels each corresponds to a type of tumor area,
    each channels contains the soft label with pixels' value in range [0, 1]
    indicating the certainty
    """
    THRESHOLD_BACKGROUND = {
            'brain-tumor': 2e-5,
            'brain-growth': 0.0,
            'kidney': -1025.,
            'prostate': 0.0
            }
    def __init__(self, 
                 root=Path.getPath('QUBIQ'), 
                 mode='train', 
                 dataset='brain-tumor', 
                 task=0, 
                 output='threshold', 
                 label_mode='qubiq'):
        """
        choices of dataset ['brain-tumor', 'brain-growth', 'kidney', 'prostate']
        
        param task: specify the task number
        """
        self.root = root
        # self.args = args
        self.mode = mode
        self.label_mode = label_mode
        self.dataset = dataset
        # set number of class(es)
        self.NCLASS = NANNOTATOR[dataset]
        self.NCHANNEL = NCHANNEL[dataset]
        self.task = task
        self.output = output

        
        img_paths = []
        # Three annotations by three professionals
        labels = []

        if dataset not in ['brain-tumor', 'brain-growth', 'kidney', 'prostate']:
            raise NotImplementedError

        data_dir = None
        if mode == 'train':
            data_dir = os.path.join(root, 'train/training_data_v2', dataset, 'Training')
            label_dir = data_dir
        elif mode == 'val':
            data_dir = os.path.join(root, 'val/validation_data_v2', dataset, 'Validation')
        elif mode == 'test':
            data_dir = os.path.join(root, 'test/test_QUBIQ', dataset, 'Testing')

        with os.scandir(data_dir) as dirs:
            # dirs contains folders named as case##
            for subdir in dirs:
                # case
                if subdir.is_dir() is not True:
                    continue

                # discard cases with empty label
                if mode=='train' and dataset == 'brain-tumor' and task == 2 and subdir.name[-2:] in ['01', '03', '06', '07', '11']:
                    continue
                if mode == 'train' and dataset == 'prostate' and task == 1 and subdir.name[-2:] in ['09']:
                    continue
                with os.scandir(subdir) as ssdir:
                    tasks = []
                    for ii in range(self.NCLASS):
                        tasks.append([])

                    for ff in ssdir:
                        if ff.name == 'image.nii.gz':
                            img_paths.append(ff.path)
                        else:
                            for ii in range(self.NCLASS):
                                # task(ii+1)
                                if ff.name[0:6] == 'task' + '{:0>2d}'.format(ii+1):
                                    tasks[ii].append(ff.path)
                    # sort to align the annotators
                    labels.append([sorted(x) for x in tasks])

        self.imgs = img_paths

        # (N, task, seg)
        self.labels = labels
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.mean = MEANS[dataset]
        self.std = STDS[dataset]
        self.original = False

    def __getitem__(self, index):
        x = nib.load(self.imgs[index]).get_fdata()
        thres = self.THRESHOLD_BACKGROUND[self.dataset]

        # normalize
        # get foreground area
        # normalize
        if not self.original:
            fore = x > thres
            x[fore] -= self.mean 
            x[fore] /= self.std
        if self.mode == 'test':
            if self.dataset == 'prostate':
                x = np.squeeze(x)
            if self.dataset == 'brain-tumor':
                x = torch.from_numpy(np.transpose(x, [2, 0, 1])).float()
            else:
                x = torch.from_numpy(np.array([x])).float()

        y = []
        for lb in self.labels[index][self.task]:
            y.append(nib.load(lb).get_fdata())
        y = np.stack(y)

        x, labels = self.__transform(x, y)

        if self.label_mode == 'ged':
            label = labels[np.random.randint(self.NCLASS)]
            return {'image': x, 'label': label, 'labels': labels}
        else:
            return {'image': x, 'label': labels}

    def __transform(self, x, y):
        # transpose from HWC to CHW
        if self.dataset == 'brain-tumor':
            x = torch.from_numpy(np.transpose(x, [2, 0, 1])).float()
            x = F.pad(x[None, ...], [8, 8, 8, 8]).squeeze()
            y = torch.from_numpy(np.array(y)).float()
            y = F.pad(y[None, ...], [8, 8, 8, 8]).squeeze()
        elif self.dataset == 'brain-growth':
            x = torch.from_numpy(np.array([x])).float()
            y = torch.from_numpy(np.array(y)).float()
        elif self.dataset == 'kidney':
            x = torch.from_numpy(np.array([x])).float()
            x = F.pad(x, [8, 7, 8, 7])
            y = torch.from_numpy(np.array(y)).float()
            y = F.pad(y, [8, 7, 8, 7])
        elif self.dataset == 'prostate':
            x = torch.from_numpy(np.array([x])).float()
            y = torch.from_numpy(np.squeeze(y)).float()
            if x.shape[1] == 960:
                x = x[:, 160:800, :]
                y = y[:, 160:800, :]
            x = x[:, :, :, 0]
        return x, y

    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def getMeanAndStd(dataset):
        means = []
        stds = []
        thres = dataset.THRESHOLD_BACKGROUND[dataset.dataset]
        for d in dataset:
            img = d['image']
            means.append(torch.mean(img[img > thres]))
            stds.append(torch.var(img[img>thres]))
        return torch.mean(torch.tensor(means)), torch.sqrt(torch.mean(torch.tensor(stds)))


class BraTSSet(torch.utils.data.Dataset):
    """
    Brain tumor segmentation dataset for pretrain
    """
    NCLASS = 3
    NMODALITY = 4
    def __init__(self, args, root=Path.getPath('brats'), 
            train=True, val_size=10000 
            ):
        self.root = root 
        self.args = args
        self.train = train 
        self.val_size = val_size
        img_paths, labels = [], []
        for ii in range(4):
            img_paths.append([])
        
        label_dir = os.scandir(os.path.join(root, 'seg'))

        for ii, scan_type in enumerate(['t1', 't1ce', 't2', 'flair']):
            with os.scandir(os.path.join(root, scan_type)) as img_dir:
                for subdir in img_dir:
                    img_paths[ii].append(subdir.path)

            
        for subdir in label_dir:
            labels.append(subdir.path)
    
        self.imgs = []
        for ii in img_paths:
            self.imgs.append(sorted(ii))
        
        self.labels = sorted(labels)
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        # pre-calculated mean & std
        self.mean = 75.177
        self.std = 38.379
        
        # close the handler
        label_dir.close()

    
    def __getitem__(self, index):
        if self.train:
            img_path = []
            for ii in range(self.NMODALITY):
                img_path.append(self.imgs[ii][index]).astype(np.float)
            label_path = self.labels[index]
        else:
            img_path = []
            for ii in range(self.NMODALITY):
                img_path.append(self.imgs[ii][self.val_size + index])
            label_path = self.labels[self.val_size + index]

        x = [np.array(Image.open(a)).astype(np.float) for a in img_path]
        y = Image.open(label_path)

        x = np.stack(x, axis=0)

        # normalization
        x[x != 0] -= self.mean
        x[x != 0] /- self.std

        # get regions of brats
        y = np.array(y)
        # whole tumor
        y0 = y > 0
        # tumor core
        y1 = np.logical_and(y==1, y==3)
        # enhancing tumor
        y2 = y==2

        y = np.stack([y0, y1, y2], axis=0).astype(np.float)
        
        sample = self._transform(x, y)

        return sample

    def _transform(self, x, y):
        composed = transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ]) 
        
        x, y = composed((x, y))

        return {'image': torch.from_numpy(x.copy()).float(),
                'label': torch.from_numpy(y.copy()).float()}

    def __len__(self):

    # take out $val_size of the imgs as validation set
        if self.train:
            return len(self.labels) - self.val_size
        else:
            return self.val_size

    
