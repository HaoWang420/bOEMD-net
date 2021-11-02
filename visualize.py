import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataloaders.brats import UncertainBraTS
from dataloaders.lidc import LIDC_IDRI
from modeling.unet import *
from modeling.bAttenUnet import MDecoderUNet
from modeling.components import HookBasedFeatureExtractor
import torch.nn.functional as F
import cv2

# from torch.utils.data.sampler import SubsetRandomSample

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


def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0 * np.divide(image.astype(np.float32), image.max())
    return image.astype(np.uint8)

def resize_image(im, size, interp=cv2.INTER_LINEAR):

    im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    return im_resized

def preproc_image(x, nlabels=None):
    x_b = np.squeeze(x)

    ims = x_b.shape[:2]

    if nlabels:
        x_b = np.uint8((x_b / (nlabels)) * 255)  # not nlabels - 1 because I prefer gray over white
    else:
        x_b = convert_to_uint8(x_b)

    # x_b = cv2.cvtColor(np.squeeze(x_b), cv2.COLOR_GRAY2BGR)
    # x_b = utils.histogram_equalization(x_b)
    x_b = resize_image(x_b, (2 * ims[0], 2 * ims[1]), interp=cv2.INTER_NEAREST)

    # ims_n = x_b.shape[:2]
    # x_b = x_b[ims_n[0]//4:3*ims_n[0]//4, ims_n[1]//4: 3*ims_n[1]//4,...]
    return x_b


def draw_seg():
    torch.cuda.cudann_enable = False
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    n_classes = 4
    task_name = "lidc"

    dataset = LIDC_IDRI( mode='qubiq')

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)

    train_indices, test_indices, val_indices = indices[2*split:], indices[1*split:2*split], indices[:split]

    
    print("Number of training/test/validation patches:", (len(train_indices), len(test_indices), len(val_indices)) )
    n_channels = 1
    index = test_indices[180]

    
    # params = torch.load(r'/data/ssd/qingqiao/BOEMD_run_test/lidc/unet-lidc/experiment_00/checkpoint.pth.tar')
    # params = torch.load(r'')

    params = torch.load(r'/data/ssd/qingqiao/BOEMD_run_test/lidc/battn_unet_e-2_new_loss/experiment_00/checkpoint.pth.tar')
    # model = UNet(n_channels, n_classes)

    model = MDecoderUNet(n_channels, n_classes)
    # model = DecoderUNet(n_channels, n_classes, attention=None)
    # model = DecoderUNet(n_channels, n_classes, attention= 'attn')
    model.load_state_dict(params['state_dict'])
    print("index", index)
    il = dataset[index]

    x = il['image']

    y = il['label'][None, ...]
    
    dataset.original = True
    x_p = dataset[index]['image']

    
    sigmoid = nn.Sigmoid()

    model.eval()
    with torch.no_grad():
        y_p = sigmoid(model(x[None, ...])[0])

    # s
    # print(x, il['label'])
    
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

    plt.figure()
    plotNNFilterOverlay(x_p[None, ...], np.zeros_like(x[None, ...]), 1, alpha=0)
    plt.savefig("./figs/{}_image_{}.png".format(task_name, str(index)))

    plt.figure()
    plotNNFilterOverlay(np.zeros_like(x_p[None, ...]), error_map[None, None, ...], 2, title='error map', alpha=0.5)
    plt.savefig("./figs/{}_error_map_{}.png".format(task_name, str(index)))

    for ii in range(y.shape[1]):
        plt.figure()

        # s_p_d = preproc_image(y[:, ii:ii+1], nlabels=2)
        # plt.imshow(s_p_d, cmap='gray')
        # plt.axis('off')
        # plt.savefig("./figs/{}_ground_truth_{}_{}.png".format(task_name, str(ii),str(index)), bbox_inches= 'tight')
        plotNNFilterOverlay(y[:, ii:ii+1], np.zeros_like(y), ii + 4, title='sample {}'.format(ii), alpha=0.1)
        plt.savefig("./figs/{}_ground_truth_{}_{}.png".format(task_name, str(ii),str(index)))

    
    for ii in range(y_p.shape[1]):    
        plt.figure()
        # s_p_d = preproc_image(y_p[:, ii:ii+1], nlabels=2)
        # plt.imshow(s_p_d, cmap='gray')
        # plt.axis('off')
        # plt.savefig("./figs/{}_sample_{}_{}.png".format(task_name, str(ii),str(index)), bbox_inches= 'tight')
        plotNNFilterOverlay(y_p[:, ii:ii+1], np.zeros_like(y_p), ii + 4, title='sample {}'.format(ii), alpha=0.1)
        plt.savefig("./figs/{}_sample_{}_{}.png".format(task_name, str(ii),str(index)))
    
    plt.figure()
    plotNNFilterOverlay(np.zeros_like(x_p[None, ...]), gamma_map[None, None, ...], 3, title='gamma map', alpha=0.5)
    plt.savefig("./figs/{}_gamma_map_{}.png".format(task_name, str(index)))
    plt.close('all')

if __name__ == "__main__":
    draw_seg()
    
