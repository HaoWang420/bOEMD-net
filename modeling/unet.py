import torch.nn.functional as F

from modeling.unet_component import *

class ProbUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5):
        super(ProbUNet, self).__init__()
        self.unet = UNet(n_channels, n_classes + 1, bilinear, dropout, dropp)
    
    def forward(self, x):
        output = self.unet(x)
        normal_dis = torch.normal(mean=torch.zeros_like(output, device=output.device), 
                                  std=torch.ones_like(output, device=output.device))
        return normal_dis[:, 1:, :, :] * torch.sqrt(output[:, 1:, :, :]) + output[:, 0:1, :, :]


class DecoderUNet(nn.Module): 
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5, attention=None):
        super(DecoderUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        self.dropp = dropp

        if self.dropout:
            self.drop = nn.Dropout2d(dropp)
        self.encoder = Encoder(n_channels, bilinear)

        for ii in range(n_classes):
            self.add_module('unet'+ str(ii), Decoder(1, attention=attention))
    
    def eval(self, mode: bool = True):
        super().eval(mode=mode)
        self.drop.train()
        return self

    def forward(self, x):
        n, c, h, w = x.shape
        out = []
        x = self.encoder(x)

        if self.dropout:
            # dropout at bottleneck
            x[0] = self.drop(x[0])

        for ii in range(self.n_classes):
            out.append(self.__dict__['_modules']['unet' + str(ii)](x))

        return torch.cat(out, dim=1)

class MultiUNet(nn.Module): 
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5):
        super(MultiUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        self.dropp = dropp

        for ii in range(n_classes):
            self.add_module('unet'+ str(ii), UNet(n_channels, 1, dropp=dropp))
    

    def forward(self, x):
        n, c, h, w = x.shape
        out = []
        for ii in range(self.n_classes):
            out.append(self.__dict__['_modules']['unet' + str(ii)](x))

        return torch.cat(out, dim=1)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout=False, dropp=0.5):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        # probability of dropout
        self.dropp = dropp
        for ii in range(1, 10):
            self.add_module('dropout'+str(ii), nn.Dropout2d(p=dropp,))

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        if self.dropout:
            x1 = self.dropout1(x1)
        x2 = self.down1(x1)
        if self.dropout:
            x2 = self.dropout2(x2)
        x3 = self.down2(x2)
        if self.dropout:
            x3 = self.dropout3(x3)
        x4 = self.down3(x3)
        if self.dropout:
            x4 = self.dropout4(x4)
        x5 = self.down4(x4)
        if self.dropout:
            x5 = self.dropout5(x5)
        x = self.up1(x5, x4)
        if self.dropout:
            x = self.dropout6(x)
        x = self.up2(x, x3)
        if self.dropout:
            x = self.dropout7(x)
        x = self.up3(x, x2)
        if self.dropout:
            x = self.dropout8(x)
        x = self.up4(x ,x1)
        if self.dropout:
            x = self.dropout9(x)
        logits = self.outc(x)
        return logits

