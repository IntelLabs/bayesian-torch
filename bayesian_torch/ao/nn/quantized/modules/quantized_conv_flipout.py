# Copyright (C) 2023 Intel Labs
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
#
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ..base_variational_layer import BaseVariationalLayer_
from .conv_flipout import *

from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

__all__ = [
    'QuantizedConv1dFlipout',
    'QuantizedConv2dFlipout',
    'QuantizedConv3dFlipout',
    'QuantizedConvTranspose1dFlipout',
    'QuantizedConvTranspose2dFlipout',
    'QuantizedConvTranspose3dFlipout',
]


class QuantizedConv1dFlipout(Conv1dFlipout):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        """
        """
        super(QuantizedConv1dFlipout).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        # for conv bn fusion
        self.bn_weight = None
        self.bn_bias = None
        self.bn_running_mean = None
        self.bn_running_var = None
        self.bn_eps = None

        self.is_dequant = False

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        if self.bn_weight is None: # has batchnorm layer, no bn fusion
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))), requires_grad=False).cpu()
        else: # fuse conv and bn
            bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel*(bn_coef.view(-1,1,1,1).expand(self.mu_kernel.shape))), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))*(bn_coef.view(-1,1,1,1).expand(self.rho_kernel.shape))), requires_grad=False).cpu()
        delattr(self, "mu_kernel")
        delattr(self, "rho_kernel")


        ## DO NOT QUANTIZE BIAS!!!! Bias should be in fp32 format.
        ## Variable names may be confusing. We don't quantize them.
        ## TODO: rename variables
        if self.bias: # if has bias
            if self.bn_weight is None: # if no bn fusion
                self.quantized_mu_bias = Parameter(self.mu_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False).cpu()
            else: # if apply bn fusion
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
                self.quantized_mu_bias = Parameter((self.mu_bias-self.bn_running_mean)*bn_coef+self.bn_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias))*bn_coef, requires_grad=False).cpu()
            delattr(self, "mu_bias")
            delattr(self, "rho_bias")
        else:
            if self.bn_weight is not None: # if no bias but apply bn fusion
                self.bias = True
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)*(-self.bn_running_mean)+self.bn_bias
                self.quantized_mu_bias = Parameter(bn_coef, requires_grad=False).cpu()
                self.quantized_sigma_bias = None

        delattr(self, "bn_weight")
        delattr(self, "bn_bias")
        delattr(self, "bn_running_mean")
        delattr(self, "bn_running_var")
        delattr(self, "bn_eps")

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        if x.dtype!=torch.quint8:
            x = torch.quantize_per_tensor(x, default_scale, default_zero_point, torch.quint8)

        bias = None
        if self.bias:
            bias = self.quantized_mu_bias

        outputs = torch.nn.quantized.functional.conv1d(x, self.quantized_mu_weight, bias, self.stride, self.padding,
                        self.dilation, self.groups, scale=default_scale, zero_point=default_zero_point) # input: quint8, weight: qint8, bias: fp32

        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

        # getting perturbation weights
        eps_kernel = torch.quantize_per_tensor(self.eps_kernel.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_kernel.q_scale())
        delta_kernel = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_kernel, new_scale, 0)

        bias = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.quantized_sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        perturbed_outputs = torch.nn.quantized.functional.conv1d(x,
                            weight=delta_kernel, bias=bias, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups, scale=default_scale, zero_point=default_zero_point)
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)

        if return_kl:
            return out, 0
        
        return out


class QuantizedConv2dFlipout(Conv2dFlipout):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False): # be aware of bias
        """

        """
        super(QuantizedConv2dFlipout, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        # for conv bn fusion
        self.bn_weight = None
        self.bn_bias = None
        self.bn_running_mean = None
        self.bn_running_var = None
        self.bn_eps = None

        self.is_dequant = False

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        # 
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        if self.bn_weight is None: # has batchnorm layer, no bn fusion
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))), requires_grad=False).cpu()
        else: # fuse conv and bn
            bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel*(bn_coef.view(-1,1,1,1).expand(self.mu_kernel.shape))), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))*(bn_coef.view(-1,1,1,1).expand(self.rho_kernel.shape))), requires_grad=False).cpu()
        delattr(self, "mu_kernel")
        delattr(self, "rho_kernel")


        ## DO NOT QUANTIZE BIAS!!!! Bias should be in fp32 format.
        ## Variable names may be confusing. We don't quantize them.
        ## TODO: rename variables
        if self.bias: # if has bias
            if self.bn_weight is None: # if no bn fusion
                self.quantized_mu_bias = Parameter(self.mu_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False).cpu()
            else: # if apply bn fusion
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
                self.quantized_mu_bias = Parameter((self.mu_bias-self.bn_running_mean)*bn_coef+self.bn_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias))*bn_coef, requires_grad=False).cpu()
            delattr(self, "mu_bias")
            delattr(self, "rho_bias")
        else:
            if self.bn_weight is not None: # if no bias but apply bn fusion
                self.bias = True
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)*(-self.bn_running_mean)+self.bn_bias
                self.quantized_mu_bias = Parameter(bn_coef, requires_grad=False).cpu()
                self.quantized_sigma_bias = None

        delattr(self, "bn_weight")
        delattr(self, "bn_bias")
        delattr(self, "bn_running_mean")
        delattr(self, "bn_running_var")
        delattr(self, "bn_eps")

    def dequantize(self):
        self.mu_kernel = self.get_dequantized_tensor(self.quantized_mu_weight)
        self.sigma_weight = self.get_dequantized_tensor(self.quantized_sigma_weight)

        if self.bias:
            self.mu_bias = self.get_dequantized_tensor(self.quantized_mu_bias)
            self.sigma_bias = self.get_dequantized_tensor(self.quantized_sigma_bias)
        
        return

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        if x.dtype!=torch.quint8:
            x = torch.quantize_per_tensor(x, default_scale, default_zero_point, torch.quint8)

        bias = None
        if self.bias:
            bias = self.quantized_mu_bias

        outputs = torch.nn.quantized.functional.conv2d(x, self.quantized_mu_weight, bias, self.stride, self.padding,
                        self.dilation, self.groups, scale=default_scale, zero_point=default_zero_point) # input: quint8, weight: qint8, bias: fp32

        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

        # getting perturbation weights
        eps_kernel = torch.quantize_per_tensor(self.eps_kernel.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_kernel.q_scale())
        delta_kernel = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_kernel, new_scale, 0)

        bias = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.quantized_sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        perturbed_outputs = torch.nn.quantized.functional.conv2d(x,
                            weight=delta_kernel, bias=bias, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups, scale=default_scale, zero_point=default_zero_point)
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)

        if return_kl:
            return out, 0
        
        return out


class QuantizedConv3dFlipout(Conv3dFlipout):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        """
        """
        super(QuantizedConv3dFlipout).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)


        # for conv bn fusion
        self.bn_weight = None
        self.bn_bias = None
        self.bn_running_mean = None
        self.bn_running_var = None
        self.bn_eps = None

        self.is_dequant = False

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        # 
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        if self.bn_weight is None: # has batchnorm layer, no bn fusion
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))), requires_grad=False).cpu()
        else: # fuse conv and bn
            bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel*(bn_coef.view(-1,1,1,1).expand(self.mu_kernel.shape))), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))*(bn_coef.view(-1,1,1,1).expand(self.rho_kernel.shape))), requires_grad=False).cpu()
        delattr(self, "mu_kernel")
        delattr(self, "rho_kernel")


        ## DO NOT QUANTIZE BIAS!!!! Bias should be in fp32 format.
        ## Variable names may be confusing. We don't quantize them.
        ## TODO: rename variables
        if self.bias: # if has bias
            if self.bn_weight is None: # if no bn fusion
                self.quantized_mu_bias = Parameter(self.mu_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False).cpu()
            else: # if apply bn fusion
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
                self.quantized_mu_bias = Parameter((self.mu_bias-self.bn_running_mean)*bn_coef+self.bn_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias))*bn_coef, requires_grad=False).cpu()
            delattr(self, "mu_bias")
            delattr(self, "rho_bias")
        else:
            if self.bn_weight is not None: # if no bias but apply bn fusion
                self.bias = True
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)*(-self.bn_running_mean)+self.bn_bias
                self.quantized_mu_bias = Parameter(bn_coef, requires_grad=False).cpu()
                self.quantized_sigma_bias = None

        delattr(self, "bn_weight")
        delattr(self, "bn_bias")
        delattr(self, "bn_running_mean")
        delattr(self, "bn_running_var")
        delattr(self, "bn_eps")

    def dequantize(self):
        self.mu_kernel = self.get_dequantized_tensor(self.quantized_mu_weight)
        self.sigma_weight = self.get_dequantized_tensor(self.quantized_sigma_weight)

        if self.bias:
            self.mu_bias = self.get_dequantized_tensor(self.quantized_mu_bias)
            self.sigma_bias = self.get_dequantized_tensor(self.quantized_sigma_bias)
        
        return

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        if x.dtype!=torch.quint8:
            x = torch.quantize_per_tensor(x, default_scale, default_zero_point, torch.quint8)

        bias = None
        if self.bias:
            bias = self.quantized_mu_bias

        outputs = torch.nn.quantized.functional.conv3d(x, self.quantized_mu_weight, bias, self.stride, self.padding,
                        self.dilation, self.groups, scale=default_scale, zero_point=default_zero_point) # input: quint8, weight: qint8, bias: fp32

        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

        # getting perturbation weights
        eps_kernel = torch.quantize_per_tensor(self.eps_kernel.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_kernel.q_scale())
        delta_kernel = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_kernel, new_scale, 0)

        bias = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.quantized_sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        perturbed_outputs = torch.nn.quantized.functional.conv3d(x,
                            weight=delta_kernel, bias=bias, stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups, scale=default_scale, zero_point=default_zero_point)
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)

        if return_kl:
            return out, 0
        
        return out

class QuantizedConvTranspose1dFlipout(ConvTranspose1dFlipout):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        """
        """
        super(QuantizedConvTranspose1dFlipout).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        # for conv bn fusion
        self.bn_weight = None
        self.bn_bias = None
        self.bn_running_mean = None
        self.bn_running_var = None
        self.bn_eps = None

        self.is_dequant = False

        if not hasattr(self, "output_padding"):
            self.output_padding = 0

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        if self.bn_weight is None: # has batchnorm layer, no bn fusion
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))), requires_grad=False).cpu()
        else: # fuse conv and bn
            bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel*(bn_coef.view(-1,1,1,1).expand(self.mu_kernel.shape))), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))*(bn_coef.view(-1,1,1,1).expand(self.rho_kernel.shape))), requires_grad=False).cpu()
        delattr(self, "mu_kernel")
        delattr(self, "rho_kernel")


        ## DO NOT QUANTIZE BIAS!!!! Bias should be in fp32 format.
        ## Variable names may be confusing. We don't quantize them.
        ## TODO: rename variables
        if self.bias: # if has bias
            if self.bn_weight is None: # if no bn fusion
                self.quantized_mu_bias = Parameter(self.mu_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False).cpu()
            else: # if apply bn fusion
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
                self.quantized_mu_bias = Parameter((self.mu_bias-self.bn_running_mean)*bn_coef+self.bn_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias))*bn_coef, requires_grad=False).cpu()
            delattr(self, "mu_bias")
            delattr(self, "rho_bias")
        else:
            if self.bn_weight is not None: # if no bias but apply bn fusion
                self.bias = True
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)*(-self.bn_running_mean)+self.bn_bias
                self.quantized_mu_bias = Parameter(bn_coef, requires_grad=False).cpu()
                self.quantized_sigma_bias = None

        delattr(self, "bn_weight")
        delattr(self, "bn_bias")
        delattr(self, "bn_running_mean")
        delattr(self, "bn_running_var")
        delattr(self, "bn_eps")

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        if x.dtype!=torch.quint8:
            x = torch.quantize_per_tensor(x, default_scale, default_zero_point, torch.quint8)

        bias = None
        if self.bias:
            bias = self.quantized_mu_bias

        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(self.quantized_mu_weight, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.dilation, self.groups)

        outputs = torch.ops.quantized.conv_transpose1d(x, self._packed_params, scale=default_scale, zero_point=default_zero_point)
        
        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

        # getting perturbation weights
        eps_kernel = torch.quantize_per_tensor(self.eps_kernel.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_kernel.q_scale())
        delta_kernel = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_kernel, new_scale, 0)

        bias = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.quantized_sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(delta_kernel, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.dilation, self.groups)
        perturbed_outputs = torch.ops.quantized.conv_transpose1d(x, self._packed_params, scale=default_scale, zero_point=default_zero_point)
        
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)

        if return_kl:
            return out, 0
        
        return out

class QuantizedConvTranspose2dFlipout(ConvTranspose2dFlipout):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        """
        """
        super(QuantizedConvTranspose2dFlipout).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        # for conv bn fusion
        self.bn_weight = None
        self.bn_bias = None
        self.bn_running_mean = None
        self.bn_running_var = None
        self.bn_eps = None

        self.is_dequant = False

        if not hasattr(self, "output_padding"):
            self.output_padding = 0

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        if self.bn_weight is None: # has batchnorm layer, no bn fusion
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))), requires_grad=False).cpu()
        else: # fuse conv and bn
            bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel*(bn_coef.view(-1,1,1,1).expand(self.mu_kernel.shape))), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))*(bn_coef.view(-1,1,1,1).expand(self.rho_kernel.shape))), requires_grad=False).cpu()
        delattr(self, "mu_kernel")
        delattr(self, "rho_kernel")


        ## DO NOT QUANTIZE BIAS!!!! Bias should be in fp32 format.
        ## Variable names may be confusing. We don't quantize them.
        ## TODO: rename variables
        if self.bias: # if has bias
            if self.bn_weight is None: # if no bn fusion
                self.quantized_mu_bias = Parameter(self.mu_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False).cpu()
            else: # if apply bn fusion
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
                self.quantized_mu_bias = Parameter((self.mu_bias-self.bn_running_mean)*bn_coef+self.bn_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias))*bn_coef, requires_grad=False).cpu()
            delattr(self, "mu_bias")
            delattr(self, "rho_bias")
        else:
            if self.bn_weight is not None: # if no bias but apply bn fusion
                self.bias = True
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)*(-self.bn_running_mean)+self.bn_bias
                self.quantized_mu_bias = Parameter(bn_coef, requires_grad=False).cpu()
                self.quantized_sigma_bias = None

        delattr(self, "bn_weight")
        delattr(self, "bn_bias")
        delattr(self, "bn_running_mean")
        delattr(self, "bn_running_var")
        delattr(self, "bn_eps")

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        if x.dtype!=torch.quint8:
            x = torch.quantize_per_tensor(x, default_scale, default_zero_point, torch.quint8)

        bias = None
        if self.bias:
            bias = self.quantized_mu_bias

        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(self.quantized_mu_weight, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.dilation, self.groups)

        outputs = torch.ops.quantized.conv_transpose2d(x, self._packed_params, scale=default_scale, zero_point=default_zero_point)
        
        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

        # getting perturbation weights
        eps_kernel = torch.quantize_per_tensor(self.eps_kernel.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_kernel.q_scale())
        delta_kernel = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_kernel, new_scale, 0)

        bias = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.quantized_sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(delta_kernel, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.dilation, self.groups)
        perturbed_outputs = torch.ops.quantized.conv_transpose2d(x, self._packed_params, scale=default_scale, zero_point=default_zero_point)
        
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)

        if return_kl:
            return out, 0
        
        return out

class QuantizedConvTranspose3dFlipout(ConvTranspose3dFlipout):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        """
        """
        super(QuantizedConvTranspose3dFlipout).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        # for conv bn fusion
        self.bn_weight = None
        self.bn_bias = None
        self.bn_running_mean = None
        self.bn_running_var = None
        self.bn_eps = None

        self.is_dequant = False

        if not hasattr(self, "output_padding"):
            self.output_padding = 0

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        if self.bn_weight is None: # has batchnorm layer, no bn fusion
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))), requires_grad=False).cpu()
        else: # fuse conv and bn
            bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
            self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_kernel*(bn_coef.view(-1,1,1,1).expand(self.mu_kernel.shape))), requires_grad=False).cpu()
            self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_kernel))*(bn_coef.view(-1,1,1,1).expand(self.rho_kernel.shape))), requires_grad=False).cpu()
        delattr(self, "mu_kernel")
        delattr(self, "rho_kernel")


        ## DO NOT QUANTIZE BIAS!!!! Bias should be in fp32 format.
        ## Variable names may be confusing. We don't quantize them.
        ## TODO: rename variables
        if self.bias: # if has bias
            if self.bn_weight is None: # if no bn fusion
                self.quantized_mu_bias = Parameter(self.mu_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False).cpu()
            else: # if apply bn fusion
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)
                self.quantized_mu_bias = Parameter((self.mu_bias-self.bn_running_mean)*bn_coef+self.bn_bias, requires_grad=False).cpu()
                self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias))*bn_coef, requires_grad=False).cpu()
            delattr(self, "mu_bias")
            delattr(self, "rho_bias")
        else:
            if self.bn_weight is not None: # if no bias but apply bn fusion
                self.bias = True
                bn_coef = self.bn_weight/torch.sqrt(self.bn_running_var+self.bn_eps)*(-self.bn_running_mean)+self.bn_bias
                self.quantized_mu_bias = Parameter(bn_coef, requires_grad=False).cpu()
                self.quantized_sigma_bias = None

        delattr(self, "bn_weight")
        delattr(self, "bn_bias")
        delattr(self, "bn_running_mean")
        delattr(self, "bn_running_var")
        delattr(self, "bn_eps")

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        if x.dtype!=torch.quint8:
            x = torch.quantize_per_tensor(x, default_scale, default_zero_point, torch.quint8)

        bias = None
        if self.bias:
            bias = self.quantized_mu_bias

        self._packed_params = torch.ops.quantized.conv_transpose3d_prepack(self.quantized_mu_weight, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.dilation, self.groups)

        outputs = torch.ops.quantized.conv_transpose3d(x, self._packed_params, scale=default_scale, zero_point=default_zero_point)
        
        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

        # getting perturbation weights
        eps_kernel = torch.quantize_per_tensor(self.eps_kernel.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_kernel.q_scale())
        delta_kernel = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_kernel, new_scale, 0)

        bias = None
        if self.bias:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.quantized_sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        self._packed_params = torch.ops.quantized.conv_transpose3d_prepack(delta_kernel, bias, self.stride,
                                 self.padding, self.output_padding,
                                 self.dilation, self.groups)
        perturbed_outputs = torch.ops.quantized.conv_transpose3d(x, self._packed_params, scale=default_scale, zero_point=default_zero_point)
        
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)

        if return_kl:
            return out, 0
        
        return out
