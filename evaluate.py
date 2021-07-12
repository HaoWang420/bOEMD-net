import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataloaders.brats import UncertainBraTS
from modeling.unet import *
from modeling.components import HookBasedFeatureExtractor
from utils.metrics import *
import torch.nn.functional as F

def init_model_dataset(
        path='./run/uncertain-kidney/multi-unet-kidney/experiment_0/checkpoint.pth.tar', 
        dataset='kidney', 
        task=0,
        _model='unet'
    ):
    _dataset = UncertainBraTS(None, mode='val', dataset=dataset, task=task, output='annotator')
    n_classes = _dataset.NCLASS
    n_channels = 4 if dataset == 'brain-tumor' else 1

    params = torch.load(path)
    # model = DecoderUNet(n_channels, n_classes, attention='prob-al')
    if _model == 'unet':
        model = UNet(n_channels, n_classes)
    elif _model == 'multi-unet':
        model = MultiUNet(n_channels, n_classes)
    elif _model == 'decoder-unet':
        model = DecoderUNet(n_channels, n_classes, attention=None)
    elif _model == 'attn-unet':
        model = DecoderUNet(n_channels, n_classes, attention='attn')
    elif _model == 'pattn-unet-al':
        model = DecoderUNet(n_channels, n_classes, attention='prob-al')

    model.load_state_dict(params['state_dict'])
    model.eval()

    return _dataset, model

def compute_ncc_ged(dataset, model):
    ncc_list = []
    ged_list = []
    eva = Evaluator(2, dice=True)
    for ii, sample in enumerate(dataset):
        if ii == 3:
            continue
        x = sample['image']
        y = sample['label']
        with torch.no_grad():
            y_p = model(x[None, ...])
        
        ncc_list.append(variance_ncc_dist(y_p[0]>0.9, y))
        ged_list.append(generalised_energy_distance(y_p[0]>0.9, y))
        # eva.add_batch(y[None, ...].numpy(), y_p.numpy())
    return np.mean(ncc_list), np.mean(ged_list)
    
    print("NCC")
    print(ncc_list)
    print(np.mean(ncc_list))

    print("GED")
    print(ged_list)
    print(np.mean(ged_list))

def main():
    nod = 'brain-tumor'
    name = 'brats'
    task = 'brats1'
    exps = [0, 1, 2, 3, 7]
    models = ['unet', 'multi-unet', 'decoder-unet', 'attn-unet', 'pattn-unet-al']
    paths = [
        './run/uncertain-{}/multi-unet-{}/experiment_{}/checkpoint.pth.tar'.format(name, task, exps[0]), 
        './run/uncertain-{}/multi-unet-{}/experiment_{}/checkpoint.pth.tar'.format(name, task, exps[1]), 
        './run/uncertain-{}/multi-unet-{}/experiment_{}/checkpoint.pth.tar'.format(name, task, exps[2]), 
        './run/uncertain-{}/multi-unet-{}/experiment_{}/checkpoint.pth.tar'.format(name, task, exps[3]), 
        './run/uncertain-{}/multi-unet-{}/experiment_{}/checkpoint.pth.tar'.format(name, task, exps[4]), 
    ]
    for ii, (path, _model) in enumerate(zip(paths, models)):
        # if ii < 4:
        #     continue
        dataset, model = init_model_dataset(
            path=path,
            dataset=nod,
            task=1,
            _model=_model
        )

        _ncc, _ged = compute_ncc_ged(dataset, model)
        print(_model)
        print("NCC: {} GED: {}".format(_ncc, _ged))


if __name__ == '__main__':
    main()