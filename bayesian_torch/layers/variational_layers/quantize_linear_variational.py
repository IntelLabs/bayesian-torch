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
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from ..base_variational_layer import BaseVariationalLayer_
import math
from .linear_variational import LinearReparameterization



class QuantizedLinearReparameterization(LinearReparameterization):
    def __init__(self,
                 in_features,
                 out_features):
        """

        """
        super(QuantizedLinearReparameterization, self).__init__(
                 in_features,
                 out_features)

        self.is_dequant = False
        self.quant_dict = None

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
        self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_weight), requires_grad=False)
        self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_weight))), requires_grad=False)
        delattr(self, "mu_weight")
        delattr(self, "rho_weight")

        self.quantized_mu_bias = self.mu_bias#Parameter(self.get_quantized_tensor(self.mu_bias), requires_grad=False)
        self.quantized_sigma_bias = Parameter(torch.log1p(torch.exp(self.rho_bias)), requires_grad=False)#Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_bias))), requires_grad=False)
        delattr(self, "mu_bias")
        delattr(self, "rho_bias")

    def dequantize(self): # Deprecated
        self.mu_weight = self.get_dequantized_tensor(self.quantized_mu_weight)
        self.sigma_weight = self.get_dequantized_tensor(self.quantized_sigma_weight)

        self.mu_bias = self.get_dequantized_tensor(self.quantized_mu_bias)
        self.sigma_bias = self.get_dequantized_tensor(self.quantized_sigma_bias)
        return

    def forward(self, input, enable_int8_compute=True, normal_scale=6/255, default_scale=0.2, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        input: tensors
            Input tensor.

        enable_int8_compute: bool, optional
            Whether to enable int8 computation.
        
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

        if self.quant_dict is not None:
            eps_weight = torch.quantize_per_tensor(self.eps_weight.data.normal_(), self.quant_dict[0]['scale'], self.quant_dict[0]['zero_point'], torch.qint8) # Quantize a tensor from normal distribution. 99.7% values will lie within 3 standard deviations, so the original range is set as 6.
            weight = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_weight, self.quant_dict[1]['scale'], self.quant_dict[1]['zero_point'])
            weight = torch.ops.quantized.add(weight, self.quantized_mu_weight, self.quant_dict[2]['scale'], self.quant_dict[2]['zero_point'])
            bias = None

            ## DO NOT QUANTIZE BIAS!!!
            if self.bias:
                if self.quantized_sigma_bias is None: # the case that bias comes from bn fusion
                    bias = self.quantized_mu_bias
                else: # original case
                    bias = self.quantized_mu_bias + (self.quantized_sigma_bias * self.eps_bias.data.normal_())

            if input.dtype!=torch.quint8: # check if input has been quantized
                input = torch.quantize_per_tensor(input, self.quant_dict[3]['scale'], self.quant_dict[3]['zero_point'], torch.quint8) # scale=0.1 by grid search; zero_point=128 for uint8 format

            out = torch.nn.quantized.functional.linear(input, weight, bias, scale=self.quant_dict[4]['scale'], zero_point=self.quant_dict[4]['zero_point']) # input: quint8, weight: qint8, bias: fp32
            out = out.dequantize()

        elif not enable_int8_compute: # Deprecated. Use this method for reducing model size only.
            if not self.is_dequant:
                self.dequantize()
                self.is_dequant = True
            weight = self.mu_weight + (self.sigma_weight * self.eps_weight.data.normal_())
            bias = None
            if self.sigma_bias is not None:
                bias = self.mu_bias + (self.sigma_bias * self.eps_bias.data.normal_())

            out = F.linear(input, weight, bias)

        else:
            eps_weight = torch.quantize_per_tensor(self.eps_weight.data.normal_(), normal_scale, 0, torch.qint8)
            new_scale = (self.quantized_sigma_weight.q_scale())*(eps_weight.q_scale())
            weight = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_weight, new_scale, 0)
            new_scale = max(new_scale, self.quantized_mu_weight.q_scale())
            weight = torch.ops.quantized.add(weight, self.quantized_mu_weight, new_scale, 0)
            bias = None

            if self.quantized_sigma_bias is not None:
                if not self.is_dequant:
                    self.dequantize()
                    self.is_dequant = True
                bias = self.mu_bias + (self.sigma_bias * self.eps_bias.data.normal_())
            if input.dtype!=torch.quint8:
                input = torch.quantize_per_tensor(input, default_scale, default_zero_point, torch.quint8)

            out = torch.nn.quantized.functional.linear(input, weight, bias, scale=default_scale, zero_point=default_zero_point) # input: quint8, weight: qint8, bias: fp32
            out = out.dequantize()
            
        if return_kl:
            return out, 0 # disable kl divergence computing
        
        return out
