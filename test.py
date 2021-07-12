import argparse
import enum
import os
from nibabel.nifti1 import Nifti1Image

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '6'

import numpy as np
from tqdm import tqdm
import torch
import nibabel as nib
import re

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.brats import UncertainBraTS
from modeling import build_model, build_transfer_learning_model
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.summaries import TensorboardSummary
from utils.saver import Saver

NCLASS = {
        'brats': 3,
        'brain-growth': 1,
        'kidney': 1,
        'prostate': 2,
        
        }

def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet Training")
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['brats', 'uncertain-brats', 'uncertain-brain-growth', 'uncertain-kidney', 'uncertain-prostate'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['soft-dice', 'dice'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    parser.add_argument('--mode', type=str, default='ssl', help='ssl represents self-supervised stage')

    parser.add_argument('--nclass', type=int ,default=21, help='number of texture classes for training texture segmentation model')

    parser.add_argument('--model', type=str, default='unet', 
                        help='specify the model, default by unet', choices=['unet','prob-unet'])
    parser.add_argument('--pretrained', type=str, default=None, 
                        help='specify the path to pretrained model parameters')

    parser.add_argument('--nchannels', type=int, default=4, help='set input channel of the model')

    parser.add_argument('--dropout', action='store_true', default=False, help='add drop out to the model')

    parser.add_argument('--drop-p', type=float, default=0.5, help='probability of applying dropout')

    parser.add_argument('--task-num', type=int, default=0, help='task No. for uncertain dataset')
    
    args = parser.parse_args()

    datasets = ['uncertain-brats', 'uncertain-kidney', 'uncertain-prostate', 'uncertain-brain-growth']

    nchannels = [4, 1, 1, 1]
    model_paths = [
        [
            'run/uncertain-brats/uncertain-brats0/experiment_1/checkpoint.pth.tar',
            'run/uncertain-brats/uncertain-brats1/experiment_3/checkpoint.pth.tar',
            'run/uncertain-brats/uncertain-brats2/experiment_9/checkpoint.pth.tar'
        ],
        [
            'run/uncertain-kidney/uncertain-kidney/experiment_8/checkpoint.pth.tar'
        ],
        [
            'run/uncertain-prostate/uncertain-prostate0/experiment_3/checkpoint.pth.tar',
            'run/uncertain-prostate/uncertain-prostate1/experiment_1/checkpoint.pth.tar'
        ],
        [
            'run/uncertain-brain-growth/uncertain-brain-growth/experiment_3/checkpoint.pth.tar'
        ]
    ]

    root = 'run/val'

    datasets_alias = {'uncertain-brats': 'brain-tumor', 'uncertain-kidney': 'kidney', 'uncertain-prostate':'prostate', 'uncertain-brain-growth': 'brain-growth'}

    case_pattern = re.compile('case[0-9]{2}')
    

    for index, (dataset, nchannel) in enumerate(zip(datasets, nchannels)):
        args.dataset = dataset
        args.nchannels = nchannel
        args.batch_size = 1
        args.test_batch_size = 1
        for ii in range(NCLASS[dataset[10:]]):
            args.task_num = ii
            testset = UncertainBraTS(args, mode='val', dataset=datasets_alias[dataset], output='leveling')
            model = build_model(args, args.nchannels, testset.NCLASS)
            checkpoint = torch.load(model_paths[index][ii], map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            # turn off dropout and stop update of batchnorm
            model.eval()

            # sigmoid
            sigmoid = torch.nn.Sigmoid()
            tbar = tqdm(testset, desc='\r')
            for i, sample in enumerate(tbar):
                print("task {} of {}".format(dataset, i))
                x = sample['image'][None, ...]
                case = testset.imgs[i]
                with torch.no_grad():
                    # output = sigmoid(model.forward(x))
                    output = torch.argmax(torch.softmax(model(x), dim=1), dim=1) / 10.0

                # channel-wise mean to produced desired probability map
                # output = torch.mean((output > 0.9).float(), dim=1).data.numpy()
                output = output.data.numpy()
                case = case_pattern.findall(case)[0]
                to_save = Nifti1Image(np.squeeze(output), np.identity(4))

                save_path = os.path.join(root, datasets_alias[dataset], case)
                try:
                    os.makedirs(save_path)
                except FileExistsError:
                    pass
                save_path = os.path.join(save_path, "task{:0>2d}.nii.gz".format(ii+1))
                print(save_path)
                nib.save(to_save, save_path)



if __name__ == '__main__':
    main()
