# Retrieved from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
from modeling.components import *


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, prior_mu=0, prior_sigma=1):
        super(conv2d, self).__init__()
        self.conv = bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma,
                        in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, padding=padding)
    def forward(self, x):
        return self.conv(x)

# building brick for unet
class DoubleConv(nn.Module):
    """
    (convolution -> BN -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Encoder, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(self, n_classes, bilinear=True, attention=None):
        super(Decoder, self).__init__()
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.up1 = Up(1024, 512 // factor, bilinear, attention)
        self.up2 = Up(512, 256 // factor, bilinear, attention)
        self.up3 = Up(256, 128 // factor, bilinear, attention)
        self.up4 = Up(128, 64, bilinear, attention)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x5, x4, x3, x2, x1 = x

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class Down(nn.Module):
    """
    maxpooling -> double_conv
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling -> double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True, attention=None):
        super(Up, self).__init__()
        self.attention = attention

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        if attention == 'prob':
            self.attn = ProbBlock(in_channels // 2, in_channels // 2, in_channels // 4)
        elif attention == 'prob-al':
            self.attn = ProbBlockAl(in_channels // 2, in_channels // 2, in_channels // 4)
        elif attention == 'attn':
            self.attn = AttentionBlock(in_channels // 2, in_channels // 2, in_channels // 4)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        if self.attention is not None:
            x1 = self.attn(g=x1, x=x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = conv2d(in_channels, out_channels, kernel_size=1, padding= 0)

    def forward(self, x):
        return self.conv(x)
