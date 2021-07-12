import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataloaders.brats import UncertainBraTS
from modeling.unet import *
from modeling.components import HookBasedFeatureExtractor
import torch.nn.functional as F


def plotNNFilterOverlay(input_im, units, figure_id, interp='bilinear',
                        colormap=cm.jet, colormap_lim=None, title='', alpha=0.8):
    filters = units.shape[1]

    for i in range(filters):
        plt.imshow(input_im[0, 0, :, :], interpolation=interp, cmap='gray')
        plt.imshow(units[0, i, :, :], interpolation=interp, cmap=colormap, alpha=alpha)
        plt.axis('off')
        plt.title(title, fontsize='small')
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

        break

    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()

def draw_attn():
    dataset = UncertainBraTS(None, mode='val', dataset='brain-tumor', task=0, output='annotator')
    n_classes = dataset.NCLASS
    n_channels = 4

    params = torch.load('./run/uncertain-brats/multi-unet-brats0/experiment_8/checkpoint.pth.tar')
    model = DecoderUNet(n_channels, n_classes, attention='prob-al')
    model.load_state_dict(params['state_dict'])
    il = dataset[0]
    x = il['image']
    dataset.original = True
    sigmoid = nn.Sigmoid()
    x_p = dataset[0]['image']
    y = il['label'][None, ...]
    plt.imshow(dataset[0]['label'][0])
    plt.savefig('image.png')

    attn_modules = []
    for name, module in model.named_modules():
        if 'attn' == name[-4:]:
            attn_modules.append((name, module))

    for ii, (name, module) in enumerate(attn_modules):
        plt.subplot(4, 4, ii + 1)
        hbfe = HookBasedFeatureExtractor(model, module, upscale=True)
        _input, _output, result = hbfe.forward(x[None, ...])
        result = sigmoid(result).numpy()
        _output = np.mean(_output.numpy(), axis=1, keepdims=True)

        plotNNFilterOverlay(x_p[None, ...], result[:, (ii % 4): (ii % 4) + 1], ii, interp='bilinear', colormap=cm.jet, title=name, alpha=0.8)
        # plotNNFilterOverlay(x_p[None, ...], _output, ii, interp='bilinear', colormap=cm.jet, title=name, alpha=0.8)
    
    ii = ii + 1
    plt.subplot(4, 4, ii + 1)
    plotNNFilterOverlay(x_p[None, ...], np.zeros_like(x[None, ...]), ii + 1, alpha=0)
    plt.subplot(4, 4, ii + 2)
    plotNNFilterOverlay(y[:, 0:], np.zeros_like(x[None, ...]), ii + 2, alpha=0)
    plt.subplot(4, 4, ii + 3)
    plotNNFilterOverlay(y[:, 1:], np.zeros_like(x[None, ...]), ii + 3, alpha=0)
    plt.subplot(4, 4, ii + 4)
    plotNNFilterOverlay(y[:, 2:], np.zeros_like(x[None, ...]), ii + 4, alpha=0)

    plt.savefig('test.png', dpi=500)

def draw_seg():
    dataset = UncertainBraTS(None, mode='val', dataset='kidney', task=0, output='annotator')
    n_classes = dataset.NCLASS
    n_channels = 1

    params = torch.load('./run/uncertain-kidney/multi-unet-kidney/experiment_5/checkpoint.pth.tar')
    # model = MultiUNet(n_channels, n_classes)
    model = DecoderUNet(n_channels, n_classes, attention='prob-al')
    model.load_state_dict(params['state_dict'])

    il = dataset[1]
    x = il['image']
    y = il['label'][None, ...]
    
    dataset.original = True
    x_p = dataset[1]['image']

    sigmoid = nn.Sigmoid()

    model.eval()
    with torch.no_grad():
        y_p = sigmoid(model(x[None, ...]))

    print(y_p.shape, y.shape)
    # s
    error_map = torch.zeros_like(y[0, 0])
    for ii in range(y_p.shape[1]):
        # y
        for jj in range(y.shape[1]):
            result = nn.functional.binary_cross_entropy(y_p[0, ii], y[0, jj], reduction='none')
            error_map += result
    error_map /= y_p.shape[1] * y.shape[1]

    gamma_map = torch.zeros_like(y[0, 0])
    # s
    for ii in range(y_p.shape[1]):
        # s hat
        for jj in range(y_p.shape[1]):
            if ii == jj:
                continue
            gamma_map += nn.functional.binary_cross_entropy(y_p[0, ii], y_p[0, jj], reduction='none')
    gamma_map /= y_p.shape[1]**2 - y_p.shape[1]

    plt.subplot(1, 6, 1)
    plotNNFilterOverlay(x_p[None, ...], np.zeros_like(x[None, ...]), 1, alpha=0)

    plt.subplot(1, 6, 2)
    plotNNFilterOverlay(np.zeros_like(x_p[None, ...]), error_map[None, None, ...], 2, title='error map', alpha=0.5)

    for ii in range(y_p.shape[1]):
        plt.subplot(1, 6, ii + 3)
        plotNNFilterOverlay(y[:, ii:ii+1], np.zeros_like(y_p), ii + 4, title='sample {}'.format(ii), alpha=0)

    plt.subplot(1, 6, 6)
    plotNNFilterOverlay(np.zeros_like(x_p[None, ...]), gamma_map[None, None, ...], 3, title='gamma map', alpha=0.5)

    plt.savefig('./figs/prediction/PAG-unet.png', dpi=500)
    

if __name__ == "__main__":
    draw_seg()
    
