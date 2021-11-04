import argparse
import os
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
def is_float(s):

    return sum([n.isdigit() for n in s.strip().split('.')]) == 2

class Bayeisan_Trainer(object):
    # Define Saver
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # Define beta
        if is_float(args.beta_type):
            self.beta_type = float(args.beta_type)
        else:
            self.beta_type = args.beta_type

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass, \
        self.train_length = make_data_loader(args,
                                             **kwargs)
        print("train_length", len(self.train_loader))

        print('number of classes: ', self.nclass)

        # Define number of epochs
        self.num_epoch = args.epochs

        # Define the parameters for the sample evaluation
        self.num_sample = args.num_sample

        # Define network
        model = None

        if args.pretrained is None:
            model = build_model(args, args.model.nchannels, self.nclass, args.model)
        else:
            model = build_transfer_learning_model(args, args.model.nchannels, self.nclass, args.pretrained)

        # set up the learning rate
        train_params = None
        train_params = [{'params': model.parameters(), 'lr': args.lr}]

        # Define Optimizer
        # optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = torch.optim.Adam(train_params, weight_decay = args.weight_decay)
        # Define Criterion

        self.criterion = SegmentationLosses(nclass=self.nclass, weight=None, cuda=args.cuda).build_loss(
            mode=args.loss.name)

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass, dice=True, loss=args.loss.name)

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
        kl_loss = 0.0

        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            batch_size = image.shape[0]
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            kl = 0
            if self.args.model != "voemd-unet":
                output, kl = self.model(image)
                # print("check for kl", output, kl)
                beta = metrics.get_beta(i, len(self.train_loader), self.beta_type, epoch, self.num_epoch)
                loss = self.criterion(output, target, kl, beta, self.train_length)
            else:
                output, mu_lists, logvar_lists = self.model(image)
                assert len(mu_lists) == len(logvar_lists)
            
                for ii, mu_list in enumerate(mu_lists):
                    for jj, mu in enumerate(mu_list):
                        temp = logvar_lists[ii][jj]
                        # temp = temp.view(temp.shape[0], temp.shape[1], -1)
                        # mu = mu.view(mu.shape[0], mu.shape[1], -1)

                        kl += torch.mean(-0.5 * torch.sum(1 + temp - mu ** 2 - temp.exp(), dim = 1))
                beta = metrics.get_beta(i, len(self.train_loader), self.beta_type, epoch, self.num_epoch)
                # print(output.shape)
                loss = self.criterion(output, target, kl, batch_size / self.train_length, beta)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            kl_loss += kl.item()

            tbar.set_description('Train loss: %.4f' % (train_loss / (i + 1)))
            # tbar.set_description("Train kl loss: %.4f" % (kl_loss / (i + 1)))

            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar("train/total_kl_loss_iter", kl.item(), i + num_img_tr * epoch)

        # Show 10 * 3 inference results each epoch
        global_step = i + num_img_tr * epoch
        self.summary.visualize(self.writer, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss / i, epoch)
        self.writer.add_scalar("train/total_kl_loss_epoch", kl_loss / i, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss))
        print("KL: %.4f" % (kl_loss))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def val(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        kl_loss = 0.0
        ged = 0.0
        ncc_score = 0.0
        qubiq_score = 0.0
        ncc_list = []
        ged_list = []
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, kl = self.model(image)
            beta = metrics.get_beta(i, len(self.val_loader), self.beta_type, epoch, self.num_epoch)
            loss = self.criterion(output, target, kl, beta, self.train_length)

            test_loss += loss.item()
            kl_loss += kl.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            # tbar.set_description("Test KL Loss: %.4f" % (kl_loss / (i + 1)))
            
            pred = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        qubiq_score = self.evaluator.QUBIQ_score()
        ged = self.evaluator.GED()
        sd = self.evaluator.SD()
        sa = self.evaluator.SA()

        self.writer.add_scalar('QUBIQ score', qubiq_score, epoch)
        self.writer.add_scalar("NCC score", ncc_score, epoch)
        self.writer.add_scalar("GED score", ged, epoch)
        self.writer.add_scalar("Sample diversity", sd, epoch)
        self.writer.add_scalar("Sample accuracy", sa, epoch)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print("dice: {}".format(dice))
        # print("Shape of dice_class: {}".format(dice_class.shape))
        print("QUBIQ score {}".format(qubiq_score))
        print("NCC score {}".format(ncc_score))
        print("GED score {}".format(ged))
        print("Sample diversity {}".format(sd))
        print("Sample accuracy {}".format(sa))
        print('Loss: %.3f' % (test_loss))

        is_best = True
        self.best_pred = qubiq_score
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)

    def val_sample(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        num_iter = len(self.val_loader)

        for i, sample in enumerate(tbar):
            if self.args.dataset == 'lidc-syn-rand':
                image, target = sample['image'], sample['labels']
            else:
                image, target = sample['image'], sample['label']
            n, c, w, h = target.shape
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            kl_losses = torch.zeros((self.num_sample, n))

            assert image.shape[0] == 1
            if self.args.dataset == 'lidc-syn-rand':
                image = image.repeat(self.num_sample * 3, 1, 1, 1)
            else:
                image = image.repeat(self.num_sample, 1, 1, 1)

            with torch.no_grad():

                if self.args.model != "voemd-unet":
                    predictions, kl = self.model(image)
                   
                else:
                    predictions, mu_lists, logvar_lists = self.model(image)

            if self.args.dataset == 'lidc-syn-rand':
                predictions = predictions.reshape((self.num_sample, 3, predictions.shape[2], predictions.shape[3]))

            mean_out = torch.mean(predictions, dim=0, keepdim=True)
            mean_kl_loss = torch.mean(kl_losses)

            pred = mean_out.data.cpu().numpy()
            target = target.data.cpu().numpy()
        
            self.evaluator.add_batch(target, pred)

        qubiq_score = self.evaluator.QUBIQ_score()
        ged = self.evaluator.GED()
        sd = self.evaluator.SD()
        sa = self.evaluator.SA()

        # save statistics to experiment dir
        self.evaluator.save(self.saver.experiment_dir)

        self.writer.add_scalar('val_sample/QUBIQ score', qubiq_score, epoch)
        self.writer.add_scalar("val_sample/GED score", ged, epoch)
        self.writer.add_scalar("val_sample/Sample diversity", sd, epoch)
        self.writer.add_scalar("val_sample/Sample accuracy", sa, epoch)
        print('Sampling:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        print("Sampling QUBIQ score {}".format(qubiq_score))
        print("Sampling GED score {}".format(ged))
        print("Sampling SD score {}".format(sd))
        print("Sampling SA score {}".format(sa))
        print('Sampling Loss: %.3f' % (test_loss))

        is_best = True
        self.best_pred = qubiq_score
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)

    def get_weight_SNR(self):
        weight_SNR_vec = []

        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):

                W_mu = module.W_mu.data
                W_p = module.W_rho.data
                sig_W = 1e-6 + F.softplus(W_p, beta=1, threshold=20)

                b_mu = module.bias_mu.data
                b_p = module.bias_rho.data
                sig_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                W_snr = (torch.abs(W_mu) / sig_W)
                b_snr = (torch.abs(b_mu) / sig_b)

                for weight_SNR in W_snr.cpu().view(-1):
                    weight_SNR_vec.append(weight_SNR)

                for weight_SNR in b_snr.cpu().view(-1):
                    weight_SNR_vec.append(weight_SNR)

        return np.array(weight_SNR_vec)

    def sample_weights(self, W_mu, b_mu, W_p, b_p):

        eps_W = W_mu.data.new(W_mu.size()).normal_()
        # sample parameters
        std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
        W = W_mu + 1 * std_w * eps_W

        if b_mu is not None:
            std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
            eps_b = b_mu.data.new(b_mu.size()).normal_()
            b = b_mu + 1 * std_b * eps_b
        else:
            b = None

        return W, b

    def get_weight_KLD(self, Nsamples=20):
        weight_KLD_vec = []

        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):

                W_mu = module.W_mu.data
                W_p = module.W_rho.data

                b_mu = module.bias_mu.data
                b_p = module.bias_rho.data

                std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                KL_W = W_mu.new(W_mu.size()).zero_()
                KL_b = b_mu.new(b_mu.size()).zero_()
                for i in range(Nsamples):
                    W, b = self.sample_weights(W_mu=W_mu, b_mu=b_mu, W_p=W_p, b_p=b_p)
                    # Note that this will currently not work with slab and spike prior
                    KL_W += metrics.isotropic_gauss_loglike(W, W_mu, std_w, do_sum=False) - module.likelihood.loglike(W,
                                                                                                                      do_sum=False)
                    KL_b += metrics.isotropic_gauss_loglike(b, b_mu, std_b, do_sum=False) - module.likelihood.loglike(b,
                                                                                                                      do_sum=False)

                KL_W /= Nsamples
                KL_b /= Nsamples

                for weight_KLD in KL_W.cpu().view(-1):
                    weight_KLD_vec.append(weight_KLD)

                for weight_KLD in KL_b.cpu().view(-1):
                    weight_KLD_vec.append(weight_KLD)

        return np.array(weight_KLD_vec)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Bayesian UNet Training")
    parser.add_argument('--save-path', type=str, default='run')

    parser.add_argument('--dataset', type=str, default='uncertain-brats',
                        choices=['brats', 'uncertain-brats', 'uncertain-brain-growth', 'uncertain-kidney',
                                'uncertain-prostate', 'lidc', 'lidc-rand', 'lidc-syn', 'lidc-syn-rand'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='ELBO',
                        choices=['soft-dice', 'dice', 'fb-dice', 'ce', 'level-thres', "ELBO", "vELBO"],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
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
    parser.add_argument('--checkname', type=str, default='batten_unet',
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--nclass', type=int, default=6,
                        help='number of texture classes for training texture segmentation model')

    parser.add_argument('--model', type=str, default='multi-bunet',
                        help='specify the model, default by unet',
                        choices=['unet', 'prob-unet', 'multi-unet', 'decoder-unet', 'attn-unet', 'pattn-unet',
                                 'pattn-unet-al', "batten-unet", "multi-bunet", "multi-atten-bunet", "bOEOD-unet", "voemd-unet" ])
    parser.add_argument('--pretrained', type=str, default=None,
                        help='specify the path to pretrained model parameters')

    parser.add_argument('--nchannels', type=int, default=4, help='set input channel of the model')

    parser.add_argument('--dropout', action='store_true', default=False, help='add drop out to the model')

    parser.add_argument('--drop-p', type=float, default=0.5, help='probability of applying dropout')

    parser.add_argument('--task-num', type=int, default=1, help='task No. for uncertain dataset')
    parser.add_argument('--num-sample', type=int, default=5, help="Sampling number")
    parser.add_argument("--beta-type", default = '0.001', choices= ['Standard', '1.0' , '0.1', '10.0', '0.0001',  '0.001',
    '0.000001', '0.0000001','Blundell', 'Soenderby', '0.000000001'] )
    # parser.add_argument("--beta-type", action='store_const', default= 'standard', const='standard',
    #                     help="the beta type default valu")

    # lidc synthetic data shuffle
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle of lidc synthetic data')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr

    print(args)
    torch.cuda.cudann_enable = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # TODO
    # build trainer
    trainer = None

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.val_sample(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
