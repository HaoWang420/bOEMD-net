import torch
from torch import flatten, log, sigmoid
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, nclass, weight=None, size_average=True, batch_average=True, cuda=False, ignore_index=True, alpha=1.0):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.alpha = alpha
        self.nclass = nclass

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'triplet':
            raise NotImplementedError
        elif mode == 'dice':
            return self.dice_coef
        elif mode == 'soft-dice':
            return self.SoftDiceLoss
        elif mode == 'level-thres':
            return self.ThresholdLeveling
        elif mode == 'fb-dice':
            return self.ForeBackGroundDice
        elif mode == "ELBO":
            return self.ELBO
        else:
            raise NotImplementedError

    def ELBO(self, input, target, kl, beta, train_size):
        """
        ELBO loss with dice loss

        Parameters
        ----------
        input
        target
        kl
        beta
        train_size

        Returns
        -------

        """
        assert not target.requires_grad
        # maximum likelihood - kl
        sigmoid = nn.Sigmoid()
        bce = nn.BCELoss()

        return self.dice_coef(input, target) + beta * kl / train_size


    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.shape
        weight = torch.zeros(11, device=logit.device)
        total = n * c * h * w
        for ii in range(11):
            weight[ii] = torch.log(total / torch.sum((target==ii).float()))

        criterion = nn.CrossEntropyLoss(weight=weight, size_average=True)

        if self.cuda:
            criterion = criterion.cuda()
        
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss
    
    def ThresholdLeveling(self, logit, target, alpha=0.9):
        n, c, h, w = logit.shape
        leveling_loss = self.CrossEntropyLoss(logit, target)
        logit = torch.nn.functional.softmax(logit)

        threshold_loss = torch.zeros(c, device=logit.device)
        for ii in range(c):
            lp = torch.sum(logit[:, (ii + 1):], dim=1)
            tp = target > ii
            intersection = (lp * tp).sum()
            
            threshold_loss[ii] += 1 - ((2.0 * intersection + 1.) / (lp.sum() + tp.sum() + 1.))
        threshold_loss = torch.mean(threshold_loss)

        return leveling_loss * (1 - alpha) + threshold_loss * alpha

    def SoftDiceLoss(self, logit, target, smooth=1.0, back=False):
        n, c, w, h = logit.shape
        loss = 0.0
        sigmoid = nn.Sigmoid()
        y_pred = sigmoid(logit)
        target = torch.reshape(target, [n, -1, w, h])

        # for compute background dice
        if back:
            y_pred = 1 - y_pred
            target = 1 - target

        
        for ii in range(self.nclass):
            pred = y_pred[:, ii]
            gt = target[:, ii]

            multed = torch.sum(pred * gt, axis=(1, 2))
            summed = torch.sum(pred**2 + gt**2, axis=(1, 2))

            dice = 1. - 2. * multed / (summed + smooth)
            loss += torch.mean(dice)
        loss /= self.nclass
        return loss

    def ForeBackGroundDice(self, logit, target, smooth=1.0):
        return self.SoftDiceLoss(logit, target) + self.SoftDiceLoss(logit, target, back=True)

    
    def dice_coef(self, preds, targets):
        smooth = 1.0
        class_num = self.nclass
        sigmoid = nn.Sigmoid()
        preds = sigmoid(preds)
        loss = torch.zeros(class_num, device=preds.device)
        for i in range(class_num):
            pred = preds[:, i, :, :]
            target = targets[:, i, :, :]
            intersection = (pred * target).sum()

            loss[i] += 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
        
        return torch.mean(loss)
            
