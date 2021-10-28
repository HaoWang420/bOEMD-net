import torch.nn.functional as F
from torch import nn 
from modeling.unet_component import *
class VariationalDecoder(nn.Module):
    def __init__(self, n_classes, bilinear=True, attention=None):
        super(VariationalDecoder, self).__init__()
        self.bilinear = bilinear
        print(bilinear)
        factor = 2 if bilinear else 1
        print("factor is", factor)
        self.mu = nn.Conv2d(512, 512 , kernel_size = 1)
        self.logvar = nn.Conv2d(512, 512 , kernel_size = 1)

        self.up1 = Up(1024, 512 // factor, bilinear, attention)
        
        self.mu1 = nn.Conv2d(512, 512, kernel_size = 1)
        self.logvar1 = nn.Conv2d(512 , 512, kernel_size = 1)


        self.up2 = Up(512, 256 // factor, bilinear, attention)
        self.mu2 = nn.Conv2d(256, 256 , kernel_size = 1)
        self.logvar2 = nn.Conv2d(256, 256, kernel_size = 1)

        self.up3 = Up(256, 128 // factor, bilinear, attention)
        self.mu3 = nn.Conv2d(128, 128, kernel_size = 1)
        self.logvar3 = nn.Conv2d(128, 128, kernel_size = 1)

        self.up4 = Up(128, 64, bilinear, attention)
        self.mu4 = nn.Conv2d(64, 64, kernel_size = 1)
        self.logvar4 = nn.Conv2d(64, 64, kernel_size = 1)

        self.outc = OutConv(64, n_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar )
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x5, x4, x3, x2, x1 = x
        mu_list = []
        logvar_list = []

        mu5 = self.mu(x5)
        logvar5 = self.logvar(x5)
        z5 = self.reparameterize(mu5, logvar5)
        mu_list.append(mu5)
        logvar_list.append(logvar5)

        mu4 = self.mu1(x4)
        logvar4 = self.logvar1(x4)
        z4 = self.reparameterize(mu4, logvar4)
        mu_list.append(mu4)
        logvar_list.append(logvar4)


        mu3 = self.mu2(x3)
        logvar3 = self.logvar2(x3)
        z3 = self.reparameterize(mu3, logvar3)
        mu_list.append(mu3)
        logvar_list.append(logvar3)

        mu2 = self.mu3(x2)
        logvar2 = self.logvar3(x2)
        z2 = self.reparameterize(mu2, logvar2)
        mu_list.append(mu2)
        logvar_list.append(logvar2)

        mu1 = self.mu4(x1)
        logvar1 = self.logvar4(x1)
        z1 = self.reparameterize(mu1, logvar1)
        mu_list.append(mu1)
        logvar_list.append(logvar1)

        x = self.up1(z5, z4)
        x = self.up2(x, z3)
        x = self.up3(x, z2)
        x = self.up4(x, z1)
        logits = self.outc(x)

        return logits, mu_list, logvar_list

class VUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True, attention = None):
        super(VUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dim = 1024
        self.encoder = Encoder(n_channels, bilinear)
        
        factor = 2 if bilinear else 1
    
        for ii in range(n_classes):
            self.add_module("unet" + str(ii), VariationalDecoder(1, bilinear= bilinear,attention = attention))
    
    def forward(self, x):
        n, c, h, w = x.shape
        x = self.encoder(x)

        mu_lists = []
        logvar_lists = []
        out = []
        for ii in range(self.n_classes):
            temp = self.__dict__['_modules']['unet' + str(ii)](x)
            out.append(temp[0])
            mu_lists.append(temp[1])
            logvar_lists.append(temp[2])
        
        return torch.cat(out, dim = 1), mu_lists, logvar_lists
    

if __name__ == "__main__":
    model= VUNet(1,4)
    x = torch.randn((32, 1, 128, 128))
    temp = model(x)