import torch.nn as nn
from .bAttenUnet_componets import BBBConv2d, ModuleWrapper
import torch
import torch.nn.functional as F

from .unet_component import DoubleConv as UDoubleConv
from .unet_component import Down as UDown
from .unet_component import Up as Uup 



class AttentionBlock(nn.Module):
    """
    Retrieved from: https://www.github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py \n
    The attention gate
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            BBBConv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            BBBConv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            BBBConv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        kl = 0.0

        return x * psi


class ProbBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ProbBlock, self).__init__()
        self.Wm_g = nn.Sequential(
            BBBConv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.Ws_g = nn.Sequential(
            BBBConv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.Wm_x = nn.Sequential(
            BBBConv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.Ws_x = nn.Sequential(
            BBBConv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            BBBConv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if self.training:
            g1_miu = self.Wm_g(g)
            g1_sig = torch.sqrt(torch.square(self.Ws_g(g)))
            g1_eps = torch.normal(mean=torch.zeros_like(g1_miu, device=g.device),
                                  std=torch.ones_like(g1_sig, device=g.device))
            g1 = g1_miu + g1_eps * g1_sig

            x1_miu = self.Wm_x(x)
            x1_sig = torch.sqrt(torch.square(self.Ws_x(x)))
            x1_eps = torch.normal(mean=torch.zeros_like(x1_miu, device=x.device),
                                  std=torch.ones_like(x1_sig, device=x.device))
            x1 = x1_miu + x1_eps * x1_sig

            psi = self.relu(g1 + x1)
            psi = self.psi(psi)

            return x * psi
        else:
            g1 = self.Wm_g(g)

            x1 = self.Wm_x(x)

            psi = self.relu(g1 + x1)
            psi = self.psi(psi)

            return x * psi


class ProbBlockAl(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ProbBlockAl, self).__init__()
        self.W_g = nn.Sequential(
            BBBConv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            BBBConv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi_m = nn.Sequential(
            BBBConv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.psi_sig = nn.Sequential(
            BBBConv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if self.training:
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)

            psi_miu = self.psi_m(psi)
            psi_sig = torch.sqrt(torch.square(self.psi_sig(psi)))
            psi = psi_miu + psi_sig * torch.randn_like(psi_sig)

            psi = self.sigmoid(psi)

            return x * psi
        else:
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)

            psi = self.psi_m(psi)
            psi = self.sigmoid(psi)

            return x * psi


class DoubleConv(nn.Module):
    """
    (convolution -> BN -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            BBBConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Encoder, self).__init__()
        self.bilinear = bilinear
        self.inc = UDoubleConv(n_channels, 64)
        self.down1 = UDown(64, 128)
        self.down2 = UDown(128, 256)
        self.down3 = UDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = UDown(512, 1024 // factor)

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
            self.conv = UDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = UDoubleConv(in_channels, out_channels)

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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DecoderUNet(ModuleWrapper):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5, attention='attn'):
        super(DecoderUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        self.dropp = dropp
        self.encoder = Encoder(n_channels, bilinear)

        for ii in range(n_classes):
            self.add_module('unet' + str(ii), Decoder(1, attention=attention))

    def forward(self, x):
        n, c, h, w = x.shape
        out = []
        x = self.encoder(x)
        for ii in range(self.n_classes):
            out.append(self.__dict__['_modules']['unet' + str(ii)](x))

        return torch.cat(out, dim=1)

class MultiBUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5):
        super(MultiBUNet, self).__init__()

        self.n_classes = n_classes
        for ii in range(n_classes):
            self.add_module('bunet' + str(ii), DecoderUNet(n_channels=n_channels, n_classes=1,
                                                                 bilinear=bilinear, dropout=dropout, dropp=dropp, attention = None))

    def forward(self, x):
        # n, c, h, w = x.shape
        out = []
        for ii in range(self.n_classes):
            out.append(self.__dict__['_modules']['bunet' + str(ii)](x))

        return torch.cat(out, dim=1)

class MultiBAUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5):
        super(MultiBUNet, self).__init__()

        self.n_classes = n_classes
        for ii in range(n_classes):
            self.add_module('baunet' + str(ii), DecoderUNet(n_channels=n_channels, n_classes=1,
                                                                 bilinear=bilinear, dropout=dropout, dropp=dropp))

    def forward(self, x):
        # n, c, h, w = x.shape
        out = []
        for ii in range(self.n_classes):
            out.append(self.__dict__['_modules']['baunet' + str(ii)](x))

        return torch.cat(out, dim=1)
class MMultiBAUNet(ModuleWrapper):
    def __init__(self, n_channels, n_classes, bilinear = True, dropout = False, dropp = 0.5):
        super(MMultiBAUNet, self).__init__()
        self.module_ = MultiBAUNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, dropout=dropout,
                                  dropp=dropp)


class MMultiBUNet(ModuleWrapper):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5):
        super(MMultiBUNet, self).__init__()
        self.module_ = MultiBUNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, dropout=dropout,
                                  dropp=dropp)



class MDecoderUNet(ModuleWrapper):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5, attention='attn'):
        super(MDecoderUNet, self).__init__()
        self.module_ = DecoderUNet(n_channels, n_classes, bilinear=bilinear, dropout=dropout, dropp=dropp,
                                   attention=attention)
        self.n_classes = n_classes

class ODecoderUNetWrapper(ModuleWrapper):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5, attention='attn'):
        super().__init__()
        self.module_ = ODecoderUNet(n_channels, n_classes, bilinear=bilinear, dropout=dropout, dropp=dropp,
                                   attention=attention)
        self.n_classes = n_classes


class ODecoderUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5, attention='attn'):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = Encoder(n_channels, bilinear)
        self.decoder = Decoder(n_classes, bilinear, attention= attention)
    
    def forward(self, x):
        x = self.encoder(x)

        return self.decoder(x)