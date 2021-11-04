import argparse
import os
from pickle import NONE
import numpy as np
from tqdm import tqdm
import torch
import math

from mypath import Path
from dataloaders import make_data_loader
from modeling import build_model, build_transfer_learning_model
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.summaries import TensorboardSummary
from utils.saver import Saver
from utils import metrics
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer(object):
    # Define Saver
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.nchannel, \
            self.train_length = make_data_loader(args, **kwargs)
        print("train_length", len(self.train_loader))

        print('number of classes: ', self.nclass)

        # Define network
        self.model = None
        self.model = build_model(args.model, self.nchannel, self.nclass, args.model.name)


        # Define Optimizer
        self.optimizer = None
        # set up the learning rate
        train_params = None
        train_params = [{'params': self.model.parameters(), 'lr': args.optim.lr}]

        if args.optim.name == 'adam':
            self.optimizer = torch.optim.Adam(train_params, weight_decay=args.optim.weight_decay)
        elif args.optim.name == 'sgd':
            self.optimizer = torch.optim.SGD(train_params, weight_decay=args.optim.weight_decay)

        # Define Criterion
        self.criterion = SegmentationLosses(args, nclass=self.nclass, weight=None, cuda=args.cuda).build_loss(
            mode=args.loss.name)


        # Define Evaluator
        self.evaluator = Evaluator(self.nclass, loss=args.loss.name)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.optim.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0

        if self.args.resume is not None:
            self.resume()

    def resume(self, ):
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))

        checkpoint = torch.load(self.args.resume)
        self.args.start_epoch = checkpoint['epoch']
        
        if self.args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume, checkpoint['epoch']))

    def training(self, epoch):
        self.model.train()

        train_loss = 0.0

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            # self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output = self.model(image)

            loss = self.criterion(output, target)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # Show 10 * 3 inference results each epoch
        global_step = i + num_img_tr * epoch
        self.summary.visualize(self.writer, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss / i, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss))

    def val(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            tbar.set_description(f'Val/Epoch {epoch}')
            
            pred = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            self.evaluator.add_batch(target, pred)

        results = self.evaluator.compute()

        for metric in results:
            self.writer.add_scalar(metric, results[metric], epoch)

        for metric in results:
            print(f"{metric} {results[metric]}")

        is_best = True
        self.best_pred = results['qubiq']
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)
