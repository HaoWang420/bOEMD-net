import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class AttentionBlock(nn.Module):
    """
    Retrieved from: https://www.github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py \n
    The attention gate
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )

        self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )

        self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
                )

        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class ProbBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ProbBlock, self).__init__()
        self.Wm_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )
        self.Ws_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )

        self.Wm_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )
        self.Ws_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )

        self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )

        self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
                )

        self.psi_m = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                )

        self.psi_sig = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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


class HookBasedFeatureExtractor(nn.Module):
    """
    retrieved from: https://www.github.com/ozan-aktay/Attention-Gated-Networks
    """
    def __init__(self, submodule, layer, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layer = layer
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_io_array(self, m, i, o):
        self.outputs = o.data.clone()

    def forward(self, x):
        target_layer = self.layer

        io_hook = target_layer.register_forward_hook(self.get_io_array)
        result = self.submodule(x)
        io_hook.remove()

        self.rescale_output_array(x.size())

        return self.inputs, self.outputs, result.detach()
    
    def rescale_output_array(self, newsize):
        print(newsize)
        print(self.outputs.size())
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        self.outputs = us(self.outputs).data
