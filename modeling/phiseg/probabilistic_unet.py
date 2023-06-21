# This code is based on: https://github.com/SimonKohl/probabilistic_unet
# Author: Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/
# Modifications: Marc Gantenbein
# This software is licensed under the Apache License 2.0

import torch
import torch.nn as nn
from modeling.phiseg.unet import Unet
from torch.distributions import Normal, Independent, kl
# This code is based on: https://github.com/SimonKohl/probabilistic_unet
# Author: Stefan Knegt https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/
# Modifications: Marc Gantenbein
# This software is licensed under the Apache License 2.0

import torch
import torch.nn as nn
from modeling.phiseg.unet import Unet
from torch.distributions import Normal, Independent, kl
import numpy as np

from modeling.phiseg import utils
from modeling.phiseg.utils import l2_regularisation
from modeling.phiseg.utils import init_weights, init_weights_orthogonal_normal

from modeling.phiseg.torchlayers import Conv2D, Conv2DSequence, ReversibleSequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional
    layers, after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU)
    activation function is applied.
    """

    def __init__(self,
                 input_channels,
                 num_filters,
                 no_convs_per_block,
                 num_classes=2,
                 initializers=None,
                 padding=True,
                 posterior=False,
                 reversible=False):

        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += num_classes

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            # subtract 1 to account for the convolution which is not reversible
            if reversible:
                layers.append(ReversibleSequence(input_dim, output_dim, kernel=3, reversible_depth=no_convs_per_block-1))
            else:
                layers.append(Conv2DSequence(input_dim, output_dim, kernel=3, depth=no_convs_per_block))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels,
                               self.num_filters,
                               self.no_convs_per_block,
                               initializers=initializers,
                               posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, kernel_size=1, stride=1)

        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        if segm is not None:
            with torch.no_grad():
                segm_one_hot = utils.convert_batch_to_onehot(segm, nlabels=2) \
                    .to(torch.device(device))

                segm_one_hot = segm_one_hot.float()
            input = torch.cat([input, torch.add(segm_one_hot, -0.5)], dim=1)

        encoding = self.encoder(input)

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers,
                 use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(Conv2D(self.num_filters[0] + self.latent_dim, self.num_filters[0], kernel_size=1))

            for _ in range(no_convs_fcomb - 2):
                layers.append(Conv2D(self.num_filters[0], self.num_filters[0], kernel_size=1))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1,
                 num_classes=1,
                 num_filters=None,
                 latent_levels=1,
                 latent_dim=2,
                 initializers=None,
                 no_convs_fcomb=4,
                 image_size=(1, 128, 128),
                 beta=10.0,
                 reversible=False):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, initializers=self.initializers,
                         apply_last_layer=False, padding=True, reversible=reversible
                         ).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                             self.latent_dim, initializers=self.initializers).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                  self.latent_dim, initializers=self.initializers, posterior=True
                                                 ).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes,
                           self.no_convs_fcomb, initializers={'w': 'orthogonal', 'b': 'normal'}, use_tile=True
                           ).to(device)

        self.last_conv = Conv2D(32, num_classes, kernel_size=1, activation=torch.nn.Identity, norm=torch.nn.Identity)

    def forward(self, patch, segm=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if self.training: # construct posterior latent space aswell e.g. during validation
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch, False)

        if self.training:
            loss = self.loss(segm)

            return self.reconstruction, loss, self.kl_divergence_loss
        return self.sample()

    def sample(self):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if self.training:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def accumulate_output(self, output_list, use_softmax=False):
        """Adapted to ProbUnet, which does not have an output list"""
        s_accum = output_list
        if use_softmax:
            return torch.nn.functional.softmax(s_accum, dim=1)
        return s_accum

    def KL_two_gauss_with_diag_cov(self, mu0, sigma0, mu1, sigma1):
        sigma0_fs = torch.mul(torch.flatten(sigma0, start_dim=1), torch.flatten(sigma0, start_dim=1))
        sigma1_fs = torch.mul(torch.flatten(sigma1, start_dim=1), torch.flatten(sigma0, start_dim=1))

        logsigma0_fs = torch.log(sigma0_fs + 1e-10)
        logsigma1_fs = torch.log(sigma1_fs + 1e-10)

        mu0_f = torch.flatten(mu0, start_dim=1)
        mu1_f = torch.flatten(mu1, start_dim=1)

        return torch.mean(
            0.5*torch.sum(
                torch.div(
                    sigma0_fs + torch.mul((mu1_f - mu0_f), (mu1_f - mu0_f)),
                    sigma1_fs + 1e-10)
                + logsigma1_fs - logsigma0_fs - 1, dim=1)
        )

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        # if analytic:
        #     # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
        #     kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        # else:
        #     if calculate_posterior:
        #         z_posterior = self.posterior_latent_space.rsample()
        #     log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
        #     log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
        #     kl_div = log_posterior_prob - log_prior_prob
        mu0 = self.posterior_latent_space.mean
        sigma0 = self.posterior_latent_space.stddev
        mu1 = self.prior_latent_space.mean
        sigma1 = self.prior_latent_space.stddev
        kl_div = self.KL_two_gauss_with_diag_cov(mu0, sigma0, mu1, sigma1)
        return kl_div

    def multinoulli_loss(self, reconstruction, target):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        batch_size = reconstruction.shape[0]

        recon_flat = reconstruction.view(batch_size, self.num_classes, -1)
        target_flat = target.view(batch_size, -1).long()
        return torch.mean(
            torch.sum(criterion(target=target_flat, input=recon_flat), dim=1)
        )

    def elbo(self, segm, analytic_kl=False, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = self.multinoulli_loss

        z_posterior = self.posterior_latent_space.rsample()

        self.kl_divergence_loss = torch.mean(self.kl_divergence(
                                                    analytic=analytic_kl, 
                                                    calculate_posterior=False, 
                                                    z_posterior=z_posterior))

        # Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(
                                    use_posterior_mean=reconstruct_posterior_mean, 
                                    calculate_posterior=False,
                                    z_posterior=z_posterior)

        reconstruction_loss = criterion(reconstruction=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + 1.0 * self.kl_divergence_loss)

    def loss(self, mask):
        elbo = self.elbo(mask)
        # already included in optim
        # reg_loss = l2_regularisation(self.posterior) + l2_regularisation(self.prior) + l2_regularisation(
            # self.fcomb.layers)
        loss = -elbo
        return loss