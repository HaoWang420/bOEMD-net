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
from utils.metrics import Evaluator
from utils.summaries import TensorboardSummary
from utils.saver import Saver
from utils import metrics
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.beta_type = args.beta_type

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass, \
                self.train_length = make_data_loader(args,
                                                    **kwargs)

        print('number of classes: ', self.nclass)

        # Define number of epochs
        self.num_epoch = args.epochs

        # Define the parameters for the sample evaluation
        self.num_sample = args.num_sample

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
        self.evaluator = Evaluator(self.nclass, dice=True, loss=args.loss_type, metrics=args.metrics)

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

#             self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            output, kl = self.model(image)
            beta = metrics.get_beta(i, len(self.train_loader), self.beta_type, epoch, self.num_epoch)
            
            loss = self.criterion(output, target, kl, beta, self.train_length)
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

    def val(self, epoch, loader):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(loader, desc='\r')
        test_loss = 0.0
        kl_loss = 0.0

        for i, sample in enumerate(tbar):
            if self.args.dataset == 'lidc-rand':
                image, target = sample['image'], sample['labels']

                # multiple forward pass sampling
                image = torch.cat([image for i in range(target.shape[1])], dim=0)
            else:
                image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output, kl = self.model(image)
            
            if self.args.dataset == 'lidc-rand':
                output = output.permute(1, 0, 2, 3)

            beta = metrics.get_beta(i, len(loader), self.beta_type, epoch, self.num_epoch)
            loss = self.criterion(output, target, kl, beta, self.train_length)

            test_loss += loss.item()
            kl_loss += kl.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            pred = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, np.mean(pred, axis=0, keepdims=True))

        qubiq_score = self.evaluator.QUBIQ_score()
        ged = self.evaluator.GED()
        sd = self.evaluator.SD()

        self.writer.add_scalar('QUBIQ score', qubiq_score, epoch)
        self.writer.add_scalar("GED score", ged, epoch)
        self.writer.add_scalar("Sample diversity", sd, epoch)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        print("QUBIQ score {}".format(qubiq_score))
        print("GED score {}".format(ged))
        print("Sample diversity {}".format(sd))
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
        for i, sample in enumerate(tbar):
            if self.args.dataset == 'lidc-rand':
                image, target = sample['image'], sample['labels']
            else:
                image, target = sample['image'], sample['label']

            n, c, w, h = target.shape
            predictions = target.data.new(self.num_sample, n, c, w, h)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            kl_losses = torch.zeros((self.num_sample, n))
            # if self.args.cuda:
            #     predictions, kl_losses = predictions.cuda(), kl_losses.cuda()
            with torch.no_grad():
                for j in range(self.num_sample):
                    output, kl = self.model(image)
                    # print(output.shape)
                    if self.args.cuda:

                        predictions[j] = output.cpu()
                        kl_losses[j] = kl.cpu()
                    else:
                        predictions[j] = output
                        kl_losses[j] = kl

            mean_out = torch.mean(predictions, dim=0, keepdim=False)
            mean_kl_loss = torch.mean(kl_losses)
            
            pred = mean_out.data.cpu().numpy()
            target = target.data.cpu().numpy()
        
            self.evaluator.add_batch(target, pred)

            tbar.set_description('Sample Dice loss: %.3f' % (test_loss / (i + 1)))
            tbar.set_description("Sample KL Loss: %.4f" % (mean_kl_loss / (i + 1)))

        qubiq_score = self.evaluator.QUBIQ_score()
        ged = self.evaluator.GED()
        sd = self.evaluator.SD()

        self.writer.add_scalar('Sampling QUBIQ score', qubiq_score, epoch)
        self.writer.add_scalar("Sampling GED score", ged, epoch)
        self.writer.add_scalar("Sampling sample diversity", sd, epoch)

        print('Sampling:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        print("Sampling QUBIQ score {}".format(qubiq_score))
        print("Sampling GED score {}".format(ged))
        print("Sampling Sample diversity {}".format(sd))
        print('Sampling Loss: %.3f' % (test_loss))

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
    parser.add_argument('--dataset', type=str, default='uncertain-prostate',
                        choices=['brats', 'uncertain-brats', 'uncertain-brain-growth', 'uncertain-kidney',
                                 'uncertain-prostate', 'lidc', 'lidc-rand'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='ELBO',
                        choices=['soft-dice', 'dice', 'fb-dice', 'ce', 'level-thres', "ELBO"],
                        help='loss func type (default: ce)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
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
    parser.add_argument('--seed', type=int, default=42, metavar='S',
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
    parser.add_argument('--eval-interval', type=int, default=4,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    parser.add_argument('--nclass', type=int, default=6,
                        help='number of texture classes for training texture segmentation model')

    parser.add_argument('--model', type=str, default='batten-unet',
                        help='specify the model, default by unet',
                        choices=['unet', 'prob-unet', 'multi-unet', 'decoder-unet', 'attn-unet', 'pattn-unet',
                                 'pattn-unet-al', 'batten-unet', 'battn-unet-one'])
    parser.add_argument('--pretrained', type=str, default=None,
                        help='specify the path to pretrained model parameters')

    parser.add_argument('--nchannels', type=int, default=1, help='set input channel of the model')

    parser.add_argument('--dropout', action='store_true', default=False, help='add drop out to the model')

    parser.add_argument('--drop-p', type=float, default=0.5, help='probability of applying dropout')

    parser.add_argument('--task-num', type=int, default=0, help='task No. for uncertain dataset')
    parser.add_argument('--num-sample', type=int, default=10, help="Sampling number")
    parser.add_argument("--beta-type", action='store_const', default= 'standard', const='standard',
                        help="the beta type default valu")
    
    parser.add_argument('--metrics', nargs='+', default=['qubiq', 'ged'])

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr

    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    trainer = Bayeisan_Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    temp_epoch = 0
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        # trainer.training(epoch)
        
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.val(epoch, trainer.val_loader)

        if epoch % 100 == (100 - 1):
            trainer.val(epoch, trainer.test_loader)


    trainer.writer.close()
    prefix = f"run/weight_distribution/{args.dataset}/{str(args.task_num + 1)}/{args.model}"

    # prefix = os.path.join(r"/home/qingqiao/bAttenUnet_test/", name)
    name = 'Attention_Unet_Bayes_By_Backprop'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    SNR_vector = trainer.get_weight_SNR()
    np.save(os.path.join(prefix, "snr_vector.npy"), SNR_vector)

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)

    sns.distplot(10 * np.log10(SNR_vector), norm_hist=False, label=name, ax=ax)

    ax.set_ylabel('Density')
    ax.legend()
    plt.title('SNR (dB) density: Total parameters: %d' % (len(SNR_vector)))
    fig.savefig(os.path.join(prefix, 'snr.png'))

    Nsamples = 20
    KLD_vector = trainer.get_weight_KLD(Nsamples)
    np.save(os.path.join(prefix, "kld_vector.npy"), KLD_vector)

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)

    sns.distplot(KLD_vector, norm_hist=False, label=name, ax=ax)
    ax.set_ylabel('Density')
    ax.legend()
    plt.title('KLD density: Total parameters: %d, Nsamples: %d' % (len(KLD_vector), Nsamples))
    fig.savefig(os.path.join(prefix, "kld.png"))
    plt.close()


if __name__ == "__main__":
    main()
