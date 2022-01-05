import torch
import torch.nn as nn
import numpy as np
from dataloaders import UncertainBraTS
import cv2
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from modeling import UNet, MultiUNet, MDecoderUNet, DecoderUNet, PHISeg

parant_path = "/data/ssd/qingqiao/BOEMD_run_test/qubiq/"

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
    # x_b = resize_image(x_b, (2 * ims[0], 2 * ims[1]), interp=cv2.INTER_NEAREST)

    # ims_n = x_b.shape[:2]
    # x_b = x_b[ims_n[0]//4:3*ims_n[0]//4, ims_n[1]//4: 3*ims_n[1]//4,...]
    return x_b

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
def sigmoid(x):
    sig = nn.Sigmoid()
    return sig(x)
    # x = np.clip(x, -88.72, 88.72)

    # return 1 / (1 + np.exp(-x))
def draw_seg(dataset, model_name, model, path, task_name):
    index = 1
    threshold  = 0.9
    if model_name == "phiseg":
        path = r"/data/ssd/wanghao/bOEMD_results/qubiq/phiseg-brain-tumor-task-1/experiment_03/checkpoint.pth.tar"
    params = torch.load(path)
    fig_path = "./figs/{}".format(model_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    model.load_state_dict(params['state_dict'])

    il = dataset[index]
    x = il['image']
    print("image shape", x.shape)
    y = il['label'][None, ...]
    
    dataset.original = False
    x_p = dataset[index]['image']


    model.eval()
    with torch.no_grad():
        if model_name == "phiseg":
            
            x_temp = x[None, ...].repeat(3, 1,1,1)
            
            pred_list = model.forward(x_temp, None)
            y_p = model.accumulate_output(pred_list)
            y_p = torch.argmax(y_p, dim=1)
            y_p = y_p[None, ...].float()
            
            # pred = pred.reshape([n, nsamples, h, w])
            # y_p = sigmoid(model.accumulate_output(pred_list))
        else:
            y_p = sigmoid(model(x[None, ...]) )
       

    print( model_name,y_p.shape, y.shape)
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

    plt.figure()
    plotNNFilterOverlay(x_p[None, ...], np.zeros_like(x[None, ...]), 1, alpha=0)
    plt.savefig("./figs/{}/{}_image_{}.png".format(model_name ,task_name, str(index)))

    plt.figure()
    plotNNFilterOverlay(np.zeros_like(x_p[None, ...]), error_map[None, None, ...], 2, title='error map', alpha=0.5)
    plt.savefig("./figs/{}/{}_error_map_{}.png".format(model_name,task_name, str(index)))

    for ii in range(y_p.shape[1]):
    
        
        plt.figure()
        # s_p_d = preproc_image(y_p[:, ii:ii+1] , nlabels=2)
        # plt.imshow(s_p_d, cmap='gray')
        # plt.axis('off')
        # plt.savefig("./figs/{}/{}_sample_{}_{}.png".format(model_name,task_name, str(ii),str(index)), bbox_inches= 'tight')
        plotNNFilterOverlay(y_p[:, ii:ii+1] > threshold, np.zeros_like(y_p), ii + 4, title='sample {}'.format(ii), alpha=0.1)
        plt.savefig("./figs/{}/{}_sample_{}_{}.png".format(model_name,task_name, str(ii),str(index)))
    plt.figure()
    g = preproc_image(gamma_map, nlabels=2)
    plt.imshow(g, cmap = "gray")
    plt.axis('off')
    # plotNNFilterOverlay(gamma_map[None, None, ...], np.zeros_like(x_p[None, ...]), 3, title='gamma map', alpha=0.3)
    plt.savefig("./figs/{}/{}_gamma_map_{}.png".format(model_name,task_name, str(index)))
    plt.close('all')

def draw_ground_truth(dataset, task_name):
   
    index = 1
    threshold  = 0.8


    il = dataset[index]
    x = il['image']
    y = il['label'][None, ...]
    
    dataset.original = True
    x_p = dataset[index]['image']

    result_path = "./figs/{}".format("ground_truth")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    gamma_map = torch.zeros_like(y[0, 0])
    # s
    for ii in range(y.shape[1]):
        # s hat
        for jj in range(y.shape[1]):
            if ii == jj:
                continue
            gamma_map += nn.functional.binary_cross_entropy(y[0, ii], y[0, jj], reduction='none')
    gamma_map /= y.shape[1]**2 - y.shape[1]

    for ii in range(y.shape[1]):
        plt.figure()
        s_p_d = preproc_image(y[:, ii:ii+1], nlabels=2)
        plt.imshow(s_p_d, cmap='gray')
        plt.axis('off')
        plt.savefig( os.path.join( result_path, "{}_ground_truth_{}_{}.png".format(task_name, str(ii),str(index)) ), bbox_inches= 'tight')
        # plotNNFilterOverlay(y_p[:, ii:ii+1], np.zeros_like(y_p), ii + 4, title='sample {}'.format(ii), alpha=0.1)
        # plt.savefig("./figs/{}_sample_{}_{}.png".format(task_name, str(ii),str(index)))
    plt.figure()
    plotNNFilterOverlay(np.zeros_like(x_p[None, ...]), gamma_map[None, None, ...], 3, title='gamma map', alpha=0.8)
    plt.savefig(os.path.join(result_path, "{}_gamma_map_{}.png".format(task_name, str(index))))
    plt.close('all') 
    
if __name__ == "__main__":
    
    dataset = UncertainBraTS(mode='val', dataset='brain-tumor', task=0, output='annotator')
    n_classes = dataset.NCLASS
    n_channels = 4
    task_name = "brain_tumor_task1"
    # models = {"boemd": MDecoderUNet(n_channels, n_classes), "decoder_attn_unet": DecoderUNet(n_channels, n_classes, attention= "attn"), 
    #           "decoder_unet": DecoderUNet(n_channels, n_classes), "multi_unet": MultiUNet(n_channels, n_classes), 
    #           "unet": UNet(n_channels, n_classes)}
    models = {"phiseg": PHISeg(n_channels, 2, [32, 64, 128, 192, 192, 192, 192], image_size=[4, 256, 256])}
    draw_ground_truth(dataset, task_name)
    for model in list(models.keys()):
        experiment = "experiment_01"
        # if model != "boemd":
        #     experiment = "experiment_01"
        draw_seg(dataset, model, models[model], os.path.join(parant_path, model, experiment, "checkpoint.pth.tar"), task_name)    