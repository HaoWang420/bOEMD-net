import argparse
import os
import numpy as np
from tqdm import tqdm
import torchbnn as bnn
import torch

from mypath import Path
from dataloaders import make_data_loader
from modeling import build_model, build_transfer_learning_model
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils import metrics
from utils.metrics import Evaluator
from utils.summaries import TensorboardSummary
from utils.saver import Saver


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass, _\
            = make_data_loader(args, **kwargs)
        print('number of classes: ', self.nclass)

        # Define network
        model = None

        if args.pretrained is None:
            model = build_model(args, args.nchannels, self.nclass, args.model)
        else:
            model = build_transfer_learning_model(args, args.nchannels, self.nclass, args.pretrained)

        # set up the learning rate
        train_params = None
        train_params = [{'params': model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion

        self.criterion = SegmentationLosses(nclass=self.nclass, weight=None, cuda=args.cuda).build_loss(
            mode=args.loss_type)

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass, dice=True, loss=args.loss_type)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
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

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        # modified
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        ncc_list = []
        ged_list = []

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        qubiq_score = self.evaluator.QUBIQ_score()
        ged = self.evaluator.GED()
        sd = self.evaluator.SD()
        sa = self.evaluator.SA()

        self.writer.add_scalar('QUBIQ score', qubiq_score, epoch)
        self.writer.add_scalar("GED score", ged, epoch)
        self.writer.add_scalar("sample diversity", sd, epoch)
        self.writer.add_scalar("sample accuracy", sa, epoch)
        # if self.args.dataset == 'uncertain-brain-tumor':
        #     for ii in range(7):
        #         self.write.add_scalar('thres='+str(ii)+'7', dice_class[ii], epoch)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print("dice: {}".format(dice))
        # print("Shape of dice_class: {}".format(dice_class.shape))
        print("QUBIQ score {}".format(qubiq_score))
        print("GED score {}".format(ged))
        print("sample diversity {}".format(sd))
        print("sample accuracy {}".format(sa))
        print('Loss: %.3f' % (test_loss))

        is_best = True
        self.best_pred = qubiq_score
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet Training")
    parser.add_argument('--dataset', type=str, default='uncertain-brats',
                        choices=['brats', 'uncertain-brats', 'uncertain-brain-growth', 'uncertain-kidney',
                                'uncertain-prostate', 'lidc'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='soft-dice',
                        choices=['soft-dice', 'dice', 'fb-dice', 'ce', 'level-thres'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default = 4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0,
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
    parser.add_argument('--checkname', type=str, default=r'/home/qingqiao/bAttenUnet_test/checkpoints',
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=4,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--mode', type=str, default='ssl', help='ssl represents self-supervised stage')

    parser.add_argument('--nclass', type=int, default=3,
                        help='number of texture classes for training texture segmentation model')

    parser.add_argument('--model', type=str, default='decoder-unet',
                        help='specify the model, default by unet',
                        choices=['unet', 'prob-unet', 'multi-unet', 'decoder-unet', 'attn-unet', 'pattn-unet',
                                'pattn-unet-al'])
    parser.add_argument('--pretrained', type=str, default=None,
                        help='specify the path to pretrained model parameters')

    parser.add_argument('--nchannels', type=int, default=4, help='set input channel of the model')

    parser.add_argument('--dropout', action='store_true', default=False, help='add drop out to the model')

    parser.add_argument('--drop-p', type=float, default=0.5, help='probability of applying dropout')

    parser.add_argument('--task-num', type=int, default=0, help='task No. for uncertain dataset')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
