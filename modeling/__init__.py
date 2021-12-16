import os
import torch
from modeling.unet import *
from modeling.bAttenUnet import MDecoderUNet, MMultiBAUNet, MMultiBUNet, ODecoderUNet, ODecoderUNetWrapper
from modeling.variantional_unet import VUNet
from modeling.phiseg.phiseg import PHISeg
from modeling.phiseg.probabilistic_unet import ProbabilisticUnet


def build_model(config, nchannels, nclass, model='unet'):
    if model == 'unet':
        return UNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p
        )
    elif model == "boemd":
        return MDecoderUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            attention=config.attention
        )
    elif model == 'multi-unet':
        return MultiUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p
        )
    elif model == 'decoder-unet':
        return DecoderUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p
        )
    elif model == "multi-bunet":
        return MMultiBUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p
        )
    elif model == "multi-atten-bunet":
        return MMultiBAUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p
        )
    elif model == 'oemd':
        return DecoderUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p,
            attention=config.attention
        )
    elif model == 'bOEOD-unet':
        return ODecoderUNetWrapper(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            attention='attn'
        )
    elif model == 'vae':
        return VUNet(
            n_channels=nchannels,
            n_classes=nclass
        )
    elif model == 'phiseg':
        return PHISeg(
            input_channels=nchannels,
            num_classes=2,
            num_filters=config.nfilters,
            image_size=config.img_size,
        )
    elif model == 'prob-unet':
        return ProbabilisticUnet(
            input_channels=nchannels,
            num_classes=2,
            num_filters=config.nfilters,
            image_size=config.img_size
        )

    else:
        raise NotImplementedError

