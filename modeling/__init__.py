import os
import torch
from modeling.unet import *
from modeling.bAttenUnet import MDecoderUNet, MMultiBAUNet, MMultiBUNet, ODecoderUNet, ODecoderUNetWrapper
from modeling.variantional_unet import VUNet


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
            attention="attn"
        )
    elif model == 'prob-unet':
        return ProbUNet(
            n_channels=nchannels,
            n_classes=nclass,
            bilinear=True,
            dropout=config.dropout,
            dropp=config.drop_p
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
    else:
        raise NotImplementedError


def build_transfer_learning_model(args, nchannels, nclass, pretrained, model='unet'):
    """

    param args:
    param nclass: number of classes
    param pretrained: path to the pretrained model parameters
    """
    # hard coded class number for pretained UNet on BraTS
    pre_model = UNet(
        n_channels=args.model.nchannels,
        n_classes=3,
        bilinear=True,
        dropout=args.dropout,
        dropp=args.drop_p
    )
    if not os.path.isfile(pretrained):
        raise RuntimeError("no checkpoint found at {}".format(pretrained))
    params = torch.load(pretrained)
    pre_model.load_state_dict(params['state_dict'])
    m = UNet(
        n_channels=args.model.nchannels,
        n_classes=nclass,
        bilinear=pre_model.bilinear,
        dropout=args.dropout,
        dropp=args.drop_p
    )

    assert args.model.nchannels == pre_model.n_channels
    m.inc = pre_model.inc
    m.down1 = pre_model.down1
    m.down2 = pre_model.down2
    m.down3 = pre_model.down3
    m.down4 = pre_model.down4
    m.up1 = pre_model.up1
    m.up2 = pre_model.up2
    m.up3 = pre_model.up3
    m.up4 = pre_model.up4

    return m
