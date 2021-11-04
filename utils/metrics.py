import difflib
import os
import scipy.spatial
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from medpy.metric import jc

labels = {
    1: 'background',
}


def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * torch.log(2 * torch.Tensor([3.14159265359]).to(x.device))
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term)
    return out

def sigmoid(x):  
    # prevent numerical overflow
    x = np.clip(x, -88.72, 88.72)

    return 1/(1+np.exp(-x))

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta

def dice_coef(preds, targets, nclass):
    smooth = 1.0
    class_num = nclass
    sigmoid = nn.Sigmoid()
    preds = sigmoid(preds)
    loss = torch.zeros(class_num, device=preds.device)
    for i in range(class_num):
        pred = preds[:, i, :, :]
        target = targets[:, i, :, :]
        intersection = (pred * target).sum()

        loss[i] += 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    return torch.mean(loss)

class Evaluator(object):
    def __init__(self, num_class, dice=False, model='unet', loss='dice', metrics=['qubiq', 'ged', 'sd', 'sa']):
        self.model = model
        self.dice=dice
        self.loss = loss
        self.metrics = metrics
        self.results = {}

        for metric in metrics:
            self.results[metric] = []
            if metric == 'dice':
                self.results['dice_class'] = []

    def __sigmoid(self, x):
        # prevent numerical overflow
        x = np.clip(x, -88.72, 88.72)

        return 1 / (1 + np.exp(-x))

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc
        
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def getHausdorff(self, target, pred):
        """Compute the 95% Hausdorff distance."""

        # ensure one slice validation/testing
        hd = dict()

        # remove batch aixs
        pred = np.mean(self.__sigmoid(pred[0]) > 0.9, axis=0)
        target = np.mean(target[0], axis=0)

        target_slice = np.floor(target*10)
        logit_slice = np.floor(pred*10)
        target_slice[target_slice==10] = 9
        logit_slice[logit_slice==10] = 9

        tp = [(target_slice==level).astype(int) for level in range(10)]
        lp = [(logit_slice==level).astype(int) for level in range(10)]

        target = sitk.GetImageFromArray(np.array(tp))
        pred = sitk.GetImageFromArray(np.array(lp))
        for k in labels.keys():
            lTestImage = sitk.BinaryThreshold(target, k, k, 1, 0)
            lResultImage = sitk.BinaryThreshold(pred, k, k, 1, 0)

            # Hausdorff distance is only defined when something is detected
            statistics = sitk.StatisticsImageFilter()
            statistics.Execute(lTestImage)
            lTestSum = statistics.GetSum()
            statistics.Execute(lResultImage)
            lResultSum = statistics.GetSum()
            if lTestSum == 0 or lResultSum == 0:
                hd[k] = None
                continue

            # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
            eTestImage = sitk.BinaryErode(lTestImage, (1, 1, 0))
            eResultImage = sitk.BinaryErode(lResultImage, (1, 1, 0))

            hTestImage = sitk.Subtract(lTestImage, eTestImage)
            hResultImage = sitk.Subtract(lResultImage, eResultImage)

            hTestArray = sitk.GetArrayFromImage(hTestImage)
            hResultArray = sitk.GetArrayFromImage(hResultImage)

            # Convert voxel location to world coordinates. Use the coordinate system of the test image
            # np.nonzero   = elements of the boundary in numpy order (zyx)
            # np.flipud    = elements in xyz order
            # np.transpose = create tuples (x,y,z)
            # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
            # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
            print(np.sum(hResultArray))
            testCoordinates = [target.TransformIndexToPhysicalPoint(x.tolist()) for x in
                            np.transpose(np.flipud(np.nonzero(hTestArray)))]
            resultCoordinates = [target.TransformIndexToPhysicalPoint(x.tolist()) for x in
                                np.transpose(np.flipud(np.nonzero(hResultArray)))]

            # Use a kd-tree for fast spatial search
            def getDistancesFromAtoB(a, b):
                kdTree = scipy.spatial.KDTree(a, leafsize=100)
                return kdTree.query(b, k=1, eps=0, p=2)[0]

            # Compute distances from test to result and vice versa.
            dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
            dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
        # here you define the percentile        
        hd[k] = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
        
        return hd

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def _dice_coef(self, gt_image, pre_image, eps=1.0):
        n, c, w, h = pre_image.shape
        gt_image = gt_image.reshape([n, -1, w, h])
        dice_class = np.zeros(c, dtype=np.float)
        for ii in range(c):
            gt = gt_image[:, ii]
            pre = pre_image[:, ii]
            multed = np.sum(gt*pre)
            summed = np.sum(gt+pre)
            dice_class[ii] += 2. * (multed/(summed+eps))

        return np.mean(dice_class), dice_class
    
    def QUBIQ(self, target, logit, eps=1.0):
        n, c, h, w = logit.shape
        if self.loss == 'ce' or self.loss == 'level-thres':
            target = target / 10.0
            logit = np.argmax(logit, axis=1) / 10.0
        else:
            logit = np.mean(self.__sigmoid(logit) > 0.9, axis=1)
            target = np.mean(target, axis=1)


        target_slice = np.floor(target*10)
        logit_slice = np.floor(logit*10)
        target_slice[target_slice==10] = 9
        logit_slice[logit_slice==10] = 9

        tp = [(target_slice==level).astype(int) for level in range(10)]
        lp = [(logit_slice==level).astype(int) for level in range(10)]

        dice = np.zeros(10)
        for ii in range(10):
            intersection = tp[ii] * lp[ii]
            dice[ii] += (2. * intersection.sum() + 1.) / (tp[ii].sum() + lp[ii].sum() + 1.)
        # print(dice.shape)
        return dice


    def add_batch(self, gt_image, pre_image):
        pre_image_sig = self.__sigmoid(pre_image)

        # get QUBIQ uncertainty estimate
        if 'qubiq' in self.metrics:
            self.results['qubiq'].append(self.QUBIQ(gt_image, pre_image))
        if 'dice' in self.metrics:
            mdice, dice_class = self._dice_coef(gt_image, pre_image_sig)
            self.results['dice'].append(mdice)
            self.results['dice_class'].append(dice_class)
        if 'ged' in self.metrics:
            n = pre_image.shape[0]
            for ii in range(n):
                self.results['ged'].append(generalised_energy_distance(pre_image_sig[ii] > 0.9, gt_image[ii]))
        if 'sd' in self.metrics:
            # sample diversity
            n = pre_image.shape[0]
            for ii in range(n):
                self.results['sd'].append(self.sample_diversity(pre_image_sig[ii] > 0.9, gt_image[ii]))
        
        if 'sa' in self.metrics:
            # sample accuracy
            n = pre_image.shape[0]
            for ii in range(n):
                self.results['sa'].append(self.sample_accuracy(pre_image_sig[ii] > 0.9, gt_image[ii]))

    def reset(self):
        for metric in self.results:
            self.results[metric].clear()

    def compute(self, ):
        results = {}
        for metric in self.results:
            results[metric] = np.mean(self.results[metric])
        
        return results
    
    @staticmethod
    def sample_diversity(sample_arr, gt_arr=None):

        N = sample_arr.shape[0]
        sd = []
        for ii in range(N):
            for jj in range(N):
                sd.append(dist_fct(sample_arr[ii, ...], sample_arr[jj, ...]))

        return np.mean(sd)

    @staticmethod
    def sample_accuracy(sample_arr, gt_arr):
        N = sample_arr.shape[0]
        M = gt_arr.shape[0]
        sd = []
        for ii in range(N):
            for jj in range(M):
                sd.append(dist_fct(sample_arr[ii, ...], gt_arr[jj, ...]))

        return np.mean(sd)

def variance_ncc_dist(sample_arr, gt_arr):
    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):

        log_samples = np.log(m_samp + eps)

        return -1.0*np.sum(m_gt*log_samples, axis=0)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """
    sample_arr = sample_arr
    gt_arr = gt_arr

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)


def dist_fct(m1, m2, nlabels=1):
    """energy distance for ged & sample diversity
    """

    per_label_iou = []
    for lbl in [1]:
        # assert not lbl == 0  # tmp check
        m1_bin = (m1 == lbl)*1
        m2_bin = (m2 == lbl)*1

        if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
            per_label_iou.append(1)
        elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
            per_label_iou.append(0)
        else:
            per_label_iou.append(jc(m1_bin, m2_bin))

    # print(1-(sum(per_label_iou) / nlabels))

    return 1-(sum(per_label_iou) / nlabels)


def generalised_energy_distance(sample_arr, gt_arr, nlabels=1, **kwargs):

    def dist_fct(m1, m2, nlabels=1):

        label_range = kwargs.get('label_range', range(nlabels))

        per_label_iou = []
        for lbl in [1]:
            # assert not lbl == 0  # tmp check
            m1_bin = (m1 == lbl)*1
            m2_bin = (m2 == lbl)*1

            if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
                per_label_iou.append(1)
            elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
                per_label_iou.append(0)
            else:
                per_label_iou.append(jc(m1_bin, m2_bin))

        # print(1-(sum(per_label_iou) / nlabels))

        return 1-(sum(per_label_iou) / nlabels)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
            d_sy.append(dist_fct(sample_arr[i,...], gt_arr[j,...]))

    for i in range(N):
        for j in range(N):
            # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
            d_ss.append(dist_fct(sample_arr[i,...], sample_arr[j,...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i,...], gt_arr[j,...]))

    return (2./(N*M))*sum(d_sy) - (1./N**2)*sum(d_ss) - (1./M**2)*sum(d_yy)

def ncc(a, v, zero_norm=True):
    a = a.flatten()
    v = v.flatten()

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a, v)

if __name__ == "__main__":
    a = torch.randn((4, 3, 240, 240))
    v = torch.randn((4, 3, 240, 240))
    print(variance_ncc_dist(a, v))
