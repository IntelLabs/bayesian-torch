# Copyright (C) 2024 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Convolutional layers with flipout Monte Carlo weight estimator to perform
# variational inference in Bayesian neural networks. Variational layers
# enables Monte Carlo approximation of the distribution over the kernel
#
# @authors: Ranganath Krishnan, Piero Esposito
#
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_variational_layer import BaseVariationalLayer_, get_kernel_size
from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver, MinMaxObserver
from torch.quantization.qconfig import QConfig


from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

__all__ = [
    'Conv1dFlipout',
    'Conv2dFlipout',
    'Conv3dFlipout',
    'ConvTranspose1dFlipout',
    'ConvTranspose2dFlipout',
    'ConvTranspose3dFlipout',
]


class Conv1dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv1d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        self.kl = 0

        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv1d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv1d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class Conv2dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv2d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        self.kl = 0
        kernel_size = get_kernel_size(kernel_size, 2)
        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv2d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv2d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return out, kl
        return out



class Conv3dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements Conv3d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.kl = 0

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        kernel_size = get_kernel_size(kernel_size, 3)

        self.mu_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv3d(x,
                           weight=self.mu_kernel,
                           bias=self.mu_bias,
                           stride=self.stride,
                           padding=self.padding,
                           dilation=self.dilation,
                           groups=self.groups)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv3d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class ConvTranspose1dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 output_padding=0,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements ConvTranspose1d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.kl = 0

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.mu_kernel = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(in_channels, out_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(in_channels, out_channels // groups, kernel_size),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv_transpose1d(x,
                                     weight=self.mu_kernel,
                                     bias=self.mu_bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     output_padding=self.output_padding,
                                     groups=self.groups,
                                     dilation=self.dilation)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv_transpose1d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     output_padding=self.output_padding,
                                     groups=self.groups,
                                     dilation=self.dilation)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class ConvTranspose2dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements ConvTranspose2d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.kl = 0

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        kernel_size = get_kernel_size(kernel_size, 2)
        self.mu_kernel = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(out_channels, in_channels // groups, kernel_size[0],
                         kernel_size[1]),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv_transpose2d(x,
                                     bias=self.mu_bias,
                                     weight=self.mu_kernel,
                                     stride=self.stride,
                                     padding=self.padding,
                                     output_padding=self.output_padding,
                                     groups=self.groups,
                                     dilation=self.dilation)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            if return_kl:
                kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv_transpose2d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     output_padding=self.output_padding,
                                     groups=self.groups,
                                     dilation=self.dilation)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs


class ConvTranspose3dFlipout(BaseVariationalLayer_):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 prior_mean=0,
                 prior_variance=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 bias=True):
        """
        Implements ConvTranspose3d layer with Flipout reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_channels: int -> number of channels in the input image,
            out_channels: int -> number of channels produced by the convolution,
            kernel_size: int -> size of the convolving kernel,
            stride: int -> stride of the convolution. Default: 1,
            padding: int -> zero-padding added to both sides of the input. Default: 0,
            dilation: int -> spacing between kernel elements. Default: 1,
            groups: int -> number of blocked connections from input channels to output channels,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bias = bias

        self.kl = 0
        kernel_size = get_kernel_size(kernel_size, 3)
        self.mu_kernel = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))
        self.rho_kernel = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]))

        self.register_buffer(
            'eps_kernel',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_mu',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)
        self.register_buffer(
            'prior_weight_sigma',
            torch.Tensor(in_channels, out_channels // groups, kernel_size[0],
                         kernel_size[1], kernel_size[2]),
            persistent=False)

        if self.bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
            self.rho_bias = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_mu', torch.Tensor(out_channels), persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_channels),
                                 persistent=False)
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(4)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(8)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        # prior values
        self.prior_weight_mu.data.fill_(self.prior_mean)
        self.prior_weight_sigma.data.fill_(self.prior_variance)

        # init our weights for the deterministic and perturbated weights
        self.mu_kernel.data.normal_(mean=self.posterior_mu_init, std=.1)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init, std=.1)

        if self.bias:
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)
            self.prior_bias_mu.data.fill_(self.prior_mean)
            self.prior_bias_sigma.data.fill_(self.prior_variance)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma)
        if self.bias:
           sigma_bias = torch.log1p(torch.exp(self.rho_bias))
           kl += self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma)
        return kl

    def forward(self, x, return_kl=True):

        if self.dnn_to_bnn_flag:
            return_kl = False

        # linear outputs
        outputs = F.conv_transpose3d(x,
                                     weight=self.mu_kernel,
                                     bias=self.mu_bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     output_padding=self.output_padding,
                                     groups=self.groups,
                                     dilation=self.dilation)

        # sampling perturbation signs
        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        # gettin perturbation weights
        sigma_weight = torch.log1p(torch.exp(self.rho_kernel))
        eps_kernel = self.eps_kernel.data.normal_()

        delta_kernel = (sigma_weight * eps_kernel)

        if return_kl:
            kl = self.kl_div(self.mu_kernel, sigma_weight, self.prior_weight_mu,
                             self.prior_weight_sigma)

        bias = None
        if self.bias:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            eps_bias = self.eps_bias.data.normal_()
            bias = (sigma_bias * eps_bias)
            kl = kl + self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                  self.prior_bias_sigma)

        # perturbed feedforward
        x_tmp = x * sign_input
        perturbed_outputs_tmp = F.conv_transpose3d(x * sign_input,
                                     weight=delta_kernel,
                                     bias=bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     output_padding=self.output_padding,
                                     groups=self.groups,
                                     dilation=self.dilation)
        perturbed_outputs = perturbed_outputs_tmp * sign_output
        out = outputs + perturbed_outputs

        if self.quant_prepare:
            # quint8 quantstub
            x = self.quint_quant[0](x) # input
            outputs = self.quint_quant[1](outputs) # output
            sign_input = self.quint_quant[2](sign_input)
            sign_output = self.quint_quant[3](sign_output)
            x_tmp = self.quint_quant[4](x_tmp)
            perturbed_outputs_tmp = self.quint_quant[5](perturbed_outputs_tmp) # output
            perturbed_outputs = self.quint_quant[6](perturbed_outputs) # output
            out = self.quint_quant[7](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_kernel = self.qint_quant[1](self.mu_kernel) # weight
            eps_kernel = self.qint_quant[2](eps_kernel) # random variable
            delta_kernel =self.qint_quant[3](delta_kernel) # multiply activation

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs
