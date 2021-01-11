# Copyright 2020 Intel Corporation
#
# Convolutional Reparameterization Layers with reparameterization estimator to perform
# mean-field Reparameterization inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after convolution operation, which is
# required to compute Evidence Lower Bound (ELBO) loss for Reparameterization inference.
#
# @authors: ranganath.krishnan@intel.com
#
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import math
from ..base_variational_layer import BaseMixtureLayer_
__all__ = [
    'Conv1dMixture', 'Conv2dMixture', 'Conv3dMixture',
    'ConvTranspose1dMixture', 'ConvTranspose2dMixture',
    'ConvTranspose3dMixture'
]


class Conv1dMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean_1=0,
                 prior_variance_1=0.2,
                 posterior_mu_init_1=0,
                 posterior_rho_init_1=-6.0,
                 prior_mean_2=0.3,
                 prior_variance_2=0.2,
                 posterior_mu_init_2=0.3,
                 posterior_rho_init_2=-6,
                 bias=True):
        """
        Implements Conv1d layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseMixtureLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2
        self.bias = bias

        self.mu_kernel_1 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.rho_kernel_1 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        self.mu_kernel_2 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.rho_kernel_2 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        self.eta_weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_1 = Parameter(torch.Tensor(out_channels))
            self.eta_bias = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_channels))

            self.mu_bias_2 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_2 = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_channels))
        else:
            self.register_buffer('eta_bias', None)

            self.register_buffer('prior_bias_mu_1', None)
            self.register_buffer('prior_bias_sigma_1', None)
            self.register_parameter('mu_bias_1', None)
            self.register_parameter('rho_bias_1', None)
            self.register_buffer('eps_bias_1', None)

            self.register_buffer('prior_bias_mu_2', None)
            self.register_buffer('prior_bias_sigma_2', None)
            self.register_parameter('mu_bias_2', None)
            self.register_parameter('rho_bias_2', None)
            self.register_buffer('eps_bias_2', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_kernel_mu_1.fill_(self.prior_mean_1)
        self.prior_kernel_sigma_1.fill_(self.prior_variance_1)
        self.prior_kernel_mu_2.fill_(self.prior_mean_2)
        self.prior_kernel_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(-0.5, 0.5)
        self.mu_kernel_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_kernel_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_kernel_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_kernel_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(-0.5, 0.5)
            self.prior_bias_mu_1.fill_(self.prior_mean_1)
            self.prior_bias_sigma_1.fill_(self.prior_variance_1)
            self.mu_bias_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
            self.rho_bias_1.data.normal_(mean=self.posterior_rho_init_1,
                                         std=0.1)

            self.prior_bias_mu_2.fill_(self.prior_mean_1)
            self.prior_bias_sigma_2.fill_(self.prior_variance_1)
            self.mu_bias_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
            self.rho_bias_2.data.normal_(mean=self.posterior_rho_init_2,
                                         std=0.1)

    def forward(self, input):
        sigma_kernel_1 = torch.log1p(torch.exp(self.rho_kernel_1))
        weight_1 = self.mu_kernel_1 + (sigma_kernel_1 *
                                       self.eps_kernel_1.normal_())

        sigma_kernel_2 = torch.log1p(torch.exp(self.rho_kernel_2))
        weight_2 = self.mu_kernel_2 + (sigma_kernel_2 *
                                       self.eps_kernel_2.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_kernel_1, sigma_kernel_1, self.mu_kernel_2, sigma_kernel_1,
            self.prior_kernel_mu_1, self.prior_kernel_sigma_1,
            self.prior_kernel_mu_2, self.prior_kernel_sigma_2, self.eta_weight,
            weight).sum()

        bias = None

        if self.mu_bias_1 is not None:
            sigma_bias_1 = torch.log1p(torch.exp(self.rho_bias_1))
            bias_1 = self.mu_bias_1 + (sigma_bias_1 *
                                       self.eps_bias_1.data.normal_())

            sigma_bias_2 = torch.log1p(torch.exp(self.rho_bias_2))
            bias_2 = self.mu_bias_2 + (sigma_bias_2 *
                                       self.eps_bias_2.data.normal_())

            bias = (self.eta_bias * bias_1) + ((1 - self.eta_bias) * bias_2)

            kl_bias = self.mixture_kl_div(
                self.mu_bias_1, sigma_bias_1, self.mu_bias_2, sigma_bias_2,
                self.prior_bias_mu_1, self.prior_bias_sigma_1,
                self.prior_bias_mu_2, self.prior_bias_sigma_2, self.eta_bias,
                bias).sum()

        out = F.conv1d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)

        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl


class Conv2dMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean_1=0,
                 prior_variance_1=0.2,
                 posterior_mu_init_1=0,
                 posterior_rho_init_1=-6.0,
                 prior_mean_2=0.3,
                 prior_variance_2=0.2,
                 posterior_mu_init_2=0.3,
                 posterior_rho_init_2=-6,
                 bias=True):
        """
        Implements Conv2d layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseMixtureLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2
        self.bias = bias

        self.mu_kernel_1 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.rho_kernel_1 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))

        self.mu_kernel_2 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.rho_kernel_2 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))

        self.eta_weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))

        self.register_buffer(
            'eps_kernel_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_mu_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))

        self.register_buffer(
            'eps_kernel_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_mu_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_1 = Parameter(torch.Tensor(out_channels))
            self.eta_bias = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_channels))

            self.mu_bias_2 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_2 = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_channels))
        else:
            self.register_buffer('eta_bias', None)

            self.register_buffer('prior_bias_mu_1', None)
            self.register_buffer('prior_bias_sigma_1', None)
            self.register_parameter('mu_bias_1', None)
            self.register_parameter('rho_bias_1', None)
            self.register_buffer('eps_bias_1', None)

            self.register_buffer('prior_bias_mu_2', None)
            self.register_buffer('prior_bias_sigma_2', None)
            self.register_parameter('mu_bias_2', None)
            self.register_parameter('rho_bias_2', None)
            self.register_buffer('eps_bias_2', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_kernel_mu_1.fill_(self.prior_mean_1)
        self.prior_kernel_sigma_1.fill_(self.prior_variance_1)
        self.prior_kernel_mu_2.fill_(self.prior_mean_2)
        self.prior_kernel_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(-0.5, 0.5)
        self.mu_kernel_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_kernel_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_kernel_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_kernel_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(-0.5, 0.5)
            self.prior_bias_mu_1.fill_(self.prior_mean_1)
            self.prior_bias_sigma_1.fill_(self.prior_variance_1)
            self.mu_bias_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
            self.rho_bias_1.data.normal_(mean=self.posterior_rho_init_1,
                                         std=0.1)

            self.prior_bias_mu_2.fill_(self.prior_mean_1)
            self.prior_bias_sigma_2.fill_(self.prior_variance_1)
            self.mu_bias_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
            self.rho_bias_2.data.normal_(mean=self.posterior_rho_init_2,
                                         std=0.1)

    def forward(self, input):
        sigma_kernel_1 = torch.log1p(torch.exp(self.rho_kernel_1))
        weight_1 = self.mu_kernel_1 + (sigma_kernel_1 *
                                       self.eps_kernel_1.normal_())

        sigma_kernel_2 = torch.log1p(torch.exp(self.rho_kernel_2))
        weight_2 = self.mu_kernel_2 + (sigma_kernel_2 *
                                       self.eps_kernel_2.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_kernel_1, sigma_kernel_1, self.mu_kernel_2, sigma_kernel_1,
            self.prior_kernel_mu_1, self.prior_kernel_sigma_1,
            self.prior_kernel_mu_2, self.prior_kernel_sigma_2, self.eta_weight,
            weight).sum()

        bias = None

        if self.mu_bias_1 is not None:
            sigma_bias_1 = torch.log1p(torch.exp(self.rho_bias_1))
            bias_1 = self.mu_bias_1 + (sigma_bias_1 *
                                       self.eps_bias_1.data.normal_())

            sigma_bias_2 = torch.log1p(torch.exp(self.rho_bias_2))
            bias_2 = self.mu_bias_2 + (sigma_bias_2 *
                                       self.eps_bias_2.data.normal_())

            bias = (self.eta_bias * bias_1) + ((1 - self.eta_bias) * bias_2)

            kl_bias = self.mixture_kl_div(
                self.mu_bias_1, sigma_bias_1, self.mu_bias_2, sigma_bias_2,
                self.prior_bias_mu_1, self.prior_bias_sigma_1,
                self.prior_bias_mu_2, self.prior_bias_sigma_2, self.eta_bias,
                bias).sum()

        out = F.conv2d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)

        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl


class Conv3dMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean_1=0,
                 prior_variance_1=0.2,
                 posterior_mu_init_1=0,
                 posterior_rho_init_1=-6.0,
                 prior_mean_2=0.3,
                 prior_variance_2=0.2,
                 posterior_mu_init_2=0.3,
                 posterior_rho_init_2=-6,
                 bias=True):
        """
        Implements Conv3d layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseMixtureLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            posterior_rho_init: float -> init std for the trainable rho parameter, sampled from N(0, posterior_rho_init),
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2
        self.bias = bias

        self.mu_kernel_1 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.rho_kernel_1 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.mu_kernel_2 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.rho_kernel_2 = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.eta_weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.register_buffer(
            'eps_kernel_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_1',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.register_buffer(
            'eps_kernel_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_2',
            torch.Tensor(out_channels, in_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_1 = Parameter(torch.Tensor(out_channels))
            self.eta_bias = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_channels))

            self.mu_bias_2 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_2 = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_channels))
        else:
            self.register_buffer('eta_bias', None)

            self.register_buffer('prior_bias_mu_1', None)
            self.register_buffer('prior_bias_sigma_1', None)
            self.register_parameter('mu_bias_1', None)
            self.register_parameter('rho_bias_1', None)
            self.register_buffer('eps_bias_1', None)

            self.register_buffer('prior_bias_mu_2', None)
            self.register_buffer('prior_bias_sigma_2', None)
            self.register_parameter('mu_bias_2', None)
            self.register_parameter('rho_bias_2', None)
            self.register_buffer('eps_bias_2', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_kernel_mu_1.fill_(self.prior_mean_1)
        self.prior_kernel_sigma_1.fill_(self.prior_variance_1)
        self.prior_kernel_mu_2.fill_(self.prior_mean_2)
        self.prior_kernel_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(-0.5, 0.5)
        self.mu_kernel_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_kernel_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_kernel_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_kernel_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(-0.5, 0.5)
            self.prior_bias_mu_1.fill_(self.prior_mean_1)
            self.prior_bias_sigma_1.fill_(self.prior_variance_1)
            self.mu_bias_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
            self.rho_bias_1.data.normal_(mean=self.posterior_rho_init_1,
                                         std=0.1)

            self.prior_bias_mu_2.fill_(self.prior_mean_1)
            self.prior_bias_sigma_2.fill_(self.prior_variance_1)
            self.mu_bias_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
            self.rho_bias_2.data.normal_(mean=self.posterior_rho_init_2,
                                         std=0.1)

    def forward(self, input):
        sigma_kernel_1 = torch.log1p(torch.exp(self.rho_kernel_1))
        weight_1 = self.mu_kernel_1 + (sigma_kernel_1 *
                                       self.eps_kernel_1.normal_())

        sigma_kernel_2 = torch.log1p(torch.exp(self.rho_kernel_2))
        weight_2 = self.mu_kernel_2 + (sigma_kernel_2 *
                                       self.eps_kernel_2.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_kernel_1, sigma_kernel_1, self.mu_kernel_2, sigma_kernel_1,
            self.prior_kernel_mu_1, self.prior_kernel_sigma_1,
            self.prior_kernel_mu_2, self.prior_kernel_sigma_2, self.eta_weight,
            weight).sum()

        bias = None

        if self.mu_bias_1 is not None:
            sigma_bias_1 = torch.log1p(torch.exp(self.rho_bias_1))
            bias_1 = self.mu_bias_1 + (sigma_bias_1 *
                                       self.eps_bias_1.data.normal_())

            sigma_bias_2 = torch.log1p(torch.exp(self.rho_bias_2))
            bias_2 = self.mu_bias_2 + (sigma_bias_2 *
                                       self.eps_bias_2.data.normal_())

            bias = (self.eta_bias * bias_1) + ((1 - self.eta_bias) * bias_2)

            kl_bias = self.mixture_kl_div(
                self.mu_bias_1, sigma_bias_1, self.mu_bias_2, sigma_bias_2,
                self.prior_bias_mu_1, self.prior_bias_sigma_1,
                self.prior_bias_mu_2, self.prior_bias_sigma_2, self.eta_bias,
                bias).sum()

        out = F.conv3d(input, weight, bias, self.stride, self.padding,
                       self.dilation, self.groups)

        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl


class ConvTranspose1dMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean_1=0,
                 prior_variance_1=0.2,
                 posterior_mu_init_1=0,
                 posterior_rho_init_1=-6.0,
                 prior_mean_2=0.3,
                 prior_variance_2=0.2,
                 posterior_mu_init_2=0.3,
                 posterior_rho_init_2=-6,
                 bias=True):
        """
        Implements Conv1d layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseMixtureLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2
        self.bias = bias

        self.mu_kernel_1 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.rho_kernel_1 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))

        self.mu_kernel_2 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.rho_kernel_2 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))

        self.eta_weight = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_1 = Parameter(torch.Tensor(out_channels))
            self.eta_bias = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_channels))

            self.mu_bias_2 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_2 = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_channels))
        else:
            self.register_buffer('eta_bias', None)

            self.register_buffer('prior_bias_mu_1', None)
            self.register_buffer('prior_bias_sigma_1', None)
            self.register_parameter('mu_bias_1', None)
            self.register_parameter('rho_bias_1', None)
            self.register_buffer('eps_bias_1', None)

            self.register_buffer('prior_bias_mu_2', None)
            self.register_buffer('prior_bias_sigma_2', None)
            self.register_parameter('mu_bias_2', None)
            self.register_parameter('rho_bias_2', None)
            self.register_buffer('eps_bias_2', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_kernel_mu_1.fill_(self.prior_mean_1)
        self.prior_kernel_sigma_1.fill_(self.prior_variance_1)
        self.prior_kernel_mu_2.fill_(self.prior_mean_2)
        self.prior_kernel_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(-0.5, 0.5)
        self.mu_kernel_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_kernel_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_kernel_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_kernel_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(-0.5, 0.5)
            self.prior_bias_mu_1.fill_(self.prior_mean_1)
            self.prior_bias_sigma_1.fill_(self.prior_variance_1)
            self.mu_bias_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
            self.rho_bias_1.data.normal_(mean=self.posterior_rho_init_1,
                                         std=0.1)

            self.prior_bias_mu_2.fill_(self.prior_mean_1)
            self.prior_bias_sigma_2.fill_(self.prior_variance_1)
            self.mu_bias_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
            self.rho_bias_2.data.normal_(mean=self.posterior_rho_init_2,
                                         std=0.1)

    def forward(self, input):
        sigma_kernel_1 = torch.log1p(torch.exp(self.rho_kernel_1))
        weight_1 = self.mu_kernel_1 + (sigma_kernel_1 *
                                       self.eps_kernel_1.normal_())

        sigma_kernel_2 = torch.log1p(torch.exp(self.rho_kernel_2))
        weight_2 = self.mu_kernel_2 + (sigma_kernel_2 *
                                       self.eps_kernel_2.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_kernel_1, sigma_kernel_1, self.mu_kernel_2, sigma_kernel_1,
            self.prior_kernel_mu_1, self.prior_kernel_sigma_1,
            self.prior_kernel_mu_2, self.prior_kernel_sigma_2, self.eta_weight,
            weight).sum()

        bias = None

        if self.mu_bias_1 is not None:
            sigma_bias_1 = torch.log1p(torch.exp(self.rho_bias_1))
            bias_1 = self.mu_bias_1 + (sigma_bias_1 *
                                       self.eps_bias_1.data.normal_())

            sigma_bias_2 = torch.log1p(torch.exp(self.rho_bias_2))
            bias_2 = self.mu_bias_2 + (sigma_bias_2 *
                                       self.eps_bias_2.data.normal_())

            bias = (self.eta_bias * bias_1) + ((1 - self.eta_bias) * bias_2)

            kl_bias = self.mixture_kl_div(
                self.mu_bias_1, sigma_bias_1, self.mu_bias_2, sigma_bias_2,
                self.prior_bias_mu_1, self.prior_bias_sigma_1,
                self.prior_bias_mu_2, self.prior_bias_sigma_2, self.eta_bias,
                bias).sum()

        out = F.conv_transpose1d(input, weight, bias, self.stride,
                                 self.padding, self.dilation, self.groups)

        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl


class ConvTranspose2dMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean_1=0,
                 prior_variance_1=0.2,
                 posterior_mu_init_1=0,
                 posterior_rho_init_1=-6.0,
                 prior_mean_2=0.3,
                 prior_variance_2=0.2,
                 posterior_mu_init_2=0.3,
                 posterior_rho_init_2=-6,
                 bias=True):
        """
        Implements Conv2d layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseMixtureLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2
        self.bias = bias

        self.mu_kernel_1 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))
        self.rho_kernel_1 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))

        self.mu_kernel_2 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))
        self.rho_kernel_2 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))

        self.eta_weight = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))

        self.register_buffer(
            'eps_kernel_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_mu_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))

        self.register_buffer(
            'eps_kernel_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_mu_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_1 = Parameter(torch.Tensor(out_channels))
            self.eta_bias = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_channels))

            self.mu_bias_2 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_2 = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_channels))
        else:
            self.register_buffer('eta_bias', None)

            self.register_buffer('prior_bias_mu_1', None)
            self.register_buffer('prior_bias_sigma_1', None)
            self.register_parameter('mu_bias_1', None)
            self.register_parameter('rho_bias_1', None)
            self.register_buffer('eps_bias_1', None)

            self.register_buffer('prior_bias_mu_2', None)
            self.register_buffer('prior_bias_sigma_2', None)
            self.register_parameter('mu_bias_2', None)
            self.register_parameter('rho_bias_2', None)
            self.register_buffer('eps_bias_2', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_kernel_mu_1.fill_(self.prior_mean_1)
        self.prior_kernel_sigma_1.fill_(self.prior_variance_1)
        self.prior_kernel_mu_2.fill_(self.prior_mean_2)
        self.prior_kernel_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(-0.5, 0.5)
        self.mu_kernel_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_kernel_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_kernel_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_kernel_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(-0.5, 0.5)
            self.prior_bias_mu_1.fill_(self.prior_mean_1)
            self.prior_bias_sigma_1.fill_(self.prior_variance_1)
            self.mu_bias_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
            self.rho_bias_1.data.normal_(mean=self.posterior_rho_init_1,
                                         std=0.1)

            self.prior_bias_mu_2.fill_(self.prior_mean_1)
            self.prior_bias_sigma_2.fill_(self.prior_variance_1)
            self.mu_bias_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
            self.rho_bias_2.data.normal_(mean=self.posterior_rho_init_2,
                                         std=0.1)

    def forward(self, input):
        sigma_kernel_1 = torch.log1p(torch.exp(self.rho_kernel_1))
        weight_1 = self.mu_kernel_1 + (sigma_kernel_1 *
                                       self.eps_kernel_1.normal_())

        sigma_kernel_2 = torch.log1p(torch.exp(self.rho_kernel_2))
        weight_2 = self.mu_kernel_2 + (sigma_kernel_2 *
                                       self.eps_kernel_2.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_kernel_1, sigma_kernel_1, self.mu_kernel_2, sigma_kernel_1,
            self.prior_kernel_mu_1, self.prior_kernel_sigma_1,
            self.prior_kernel_mu_2, self.prior_kernel_sigma_2, self.eta_weight,
            weight).sum()

        bias = None

        if self.mu_bias_1 is not None:
            sigma_bias_1 = torch.log1p(torch.exp(self.rho_bias_1))
            bias_1 = self.mu_bias_1 + (sigma_bias_1 *
                                       self.eps_bias_1.data.normal_())

            sigma_bias_2 = torch.log1p(torch.exp(self.rho_bias_2))
            bias_2 = self.mu_bias_2 + (sigma_bias_2 *
                                       self.eps_bias_2.data.normal_())

            bias = (self.eta_bias * bias_1) + ((1 - self.eta_bias) * bias_2)

            kl_bias = self.mixture_kl_div(
                self.mu_bias_1, sigma_bias_1, self.mu_bias_2, sigma_bias_2,
                self.prior_bias_mu_1, self.prior_bias_sigma_1,
                self.prior_bias_mu_2, self.prior_bias_sigma_2, self.eta_bias,
                bias).sum()

        out = F.conv_transpose2d(input, weight, bias, self.stride,
                                 self.padding, self.dilation, self.groups)

        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl


class ConvTranspose3dMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean_1=0,
                 prior_variance_1=0.2,
                 posterior_mu_init_1=0,
                 posterior_rho_init_1=-6.0,
                 prior_mean_2=0.3,
                 prior_variance_2=0.2,
                 posterior_mu_init_2=0.3,
                 posterior_rho_init_2=-6,
                 bias=True):
        """
        Implements Conv3d layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseMixtureLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """

        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('invalid in_channels size')
        if out_channels % groups != 0:
            raise ValueError('invalid in_channels size')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2
        self.bias = bias

        self.mu_kernel_1 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.rho_kernel_1 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.mu_kernel_2 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.rho_kernel_2 = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.eta_weight = Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.register_buffer(
            'eps_kernel_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_1',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        self.register_buffer(
            'eps_kernel_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_mu_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))
        self.register_buffer(
            'prior_kernel_sigma_2',
            torch.Tensor(in_channels, out_channels // groups, kernel_size,
                         kernel_size, kernel_size))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_1 = Parameter(torch.Tensor(out_channels))
            self.eta_bias = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_channels))

            self.mu_bias_2 = Parameter(torch.Tensor(out_channels))
            self.rho_bias_2 = Parameter(torch.Tensor(out_channels))

            self.register_buffer('eps_bias_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_channels))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_channels))
        else:
            self.register_buffer('eta_bias', None)

            self.register_buffer('prior_bias_mu_1', None)
            self.register_buffer('prior_bias_sigma_1', None)
            self.register_parameter('mu_bias_1', None)
            self.register_parameter('rho_bias_1', None)
            self.register_buffer('eps_bias_1', None)

            self.register_buffer('prior_bias_mu_2', None)
            self.register_buffer('prior_bias_sigma_2', None)
            self.register_parameter('mu_bias_2', None)
            self.register_parameter('rho_bias_2', None)
            self.register_buffer('eps_bias_2', None)

        self.init_parameters()

    def init_parameters(self):
        self.prior_kernel_mu_1.fill_(self.prior_mean_1)
        self.prior_kernel_sigma_1.fill_(self.prior_variance_1)
        self.prior_kernel_mu_2.fill_(self.prior_mean_2)
        self.prior_kernel_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(-0.5, 0.5)
        self.mu_kernel_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_kernel_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_kernel_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_kernel_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(-0.5, 0.5)
            self.prior_bias_mu_1.fill_(self.prior_mean_1)
            self.prior_bias_sigma_1.fill_(self.prior_variance_1)
            self.mu_bias_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
            self.rho_bias_1.data.normal_(mean=self.posterior_rho_init_1,
                                         std=0.1)

            self.prior_bias_mu_2.fill_(self.prior_mean_1)
            self.prior_bias_sigma_2.fill_(self.prior_variance_1)
            self.mu_bias_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
            self.rho_bias_2.data.normal_(mean=self.posterior_rho_init_2,
                                         std=0.1)

    def forward(self, input):
        sigma_kernel_1 = torch.log1p(torch.exp(self.rho_kernel_1))
        weight_1 = self.mu_kernel_1 + (sigma_kernel_1 *
                                       self.eps_kernel_1.normal_())

        sigma_kernel_2 = torch.log1p(torch.exp(self.rho_kernel_2))
        weight_2 = self.mu_kernel_2 + (sigma_kernel_2 *
                                       self.eps_kernel_2.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_kernel_1, sigma_kernel_1, self.mu_kernel_2, sigma_kernel_1,
            self.prior_kernel_mu_1, self.prior_kernel_sigma_1,
            self.prior_kernel_mu_2, self.prior_kernel_sigma_2, self.eta_weight,
            weight).sum()

        bias = None

        if self.mu_bias_1 is not None:
            sigma_bias_1 = torch.log1p(torch.exp(self.rho_bias_1))
            bias_1 = self.mu_bias_1 + (sigma_bias_1 *
                                       self.eps_bias_1.data.normal_())

            sigma_bias_2 = torch.log1p(torch.exp(self.rho_bias_2))
            bias_2 = self.mu_bias_2 + (sigma_bias_2 *
                                       self.eps_bias_2.data.normal_())

            bias = (self.eta_bias * bias_1) + ((1 - self.eta_bias) * bias_2)

            kl_bias = self.mixture_kl_div(
                self.mu_bias_1, sigma_bias_1, self.mu_bias_2, sigma_bias_2,
                self.prior_bias_mu_1, self.prior_bias_sigma_1,
                self.prior_bias_mu_2, self.prior_bias_sigma_2, self.eta_bias,
                bias).sum()

        out = F.conv_transpose3d(input, weight, bias, self.stride,
                                 self.padding, self.dilation, self.groups)

        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl
