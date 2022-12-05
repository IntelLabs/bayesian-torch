# Copyright (C) 2021 Intel Labs
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
# Linear Flipout Layers with flipout weight estimator to perform
# variational inference in Bayesian neural networks. Variational layers
# enables Monte Carlo approximation of the distribution over the weights
#
# @authors: Ranganath Krishnan, Piero Esposito
#
# ======================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from .linear_flipout import LinearFlipout

__all__ = ["QuantizedLinearFlipout"]

class QuantizedLinearFlipout(LinearFlipout):
    def __init__(self,
                 in_features,
                 out_features):

        super(QuantizedLinearFlipout, self).__init__(
                 in_features,
                 out_features)

        self.is_dequant = False

    def get_scale_and_zero_point(self, x):

        # symmetry
        scale = torch.zeros(1).to(x.device)
        zero_point = torch.zeros(1).to(x.device)
        xmax = torch.clamp(x.abs().max(), -100, 100)
        scale = xmax*2/255

        return scale, zero_point

    def get_quantized_tensor(self, x):
        scale, zero_point = self.get_scale_and_zero_point(x)
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):
        dequantized_x = x.dequantize()
        # int8_x = dequantized_x*scale.to(torch.int8)

        return dequantized_x


    def quantize(self):
        self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_weight), requires_grad=False)
        self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_weight))), requires_grad=False)
        delattr(self, "mu_weight")
        delattr(self, "rho_weight")

        self.quantized_mu_bias = Parameter(self.get_quantized_tensor(self.mu_bias), requires_grad=False)
        self.quantized_sigma_bias = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_bias))), requires_grad=False)
        delattr(self, "mu_bias")
        delattr(self, "rho_bias")

    def dequantize(self):
        self.mu_weight = self.get_dequantized_tensor(self.quantized_mu_weight)
        self.sigma_weight = self.get_dequantized_tensor(self.quantized_sigma_weight)

        self.mu_bias = self.get_dequantized_tensor(self.quantized_mu_bias)
        self.sigma_bias = self.get_dequantized_tensor(self.quantized_sigma_bias)
        return

    def forward(self, x):

        bias = None
        if self.quantized_mu_bias is not None:
            if not self.is_dequant:
                    self.dequantize()
                    self.is_dequant = True
            bias = self.mu_bias

        outputs = torch.nn.quantized.functional.linear(x, self.quantized_mu_weight, bias, scale=0.1, zero_point=128) # input: quint8, weight: qint8, bias: fp32

        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, 1, 128, torch.quint8) # scale?
        sign_output = torch.quantize_per_tensor(sign_output, 1, 128, torch.quint8) # scale?

         # getting perturbation weights
        eps_weight = torch.quantize_per_tensor(self.eps_weight.data.normal_(), 6/255, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_weight.q_scale())
        delta_weight = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_weight, new_scale, 0)

        bias = None
        if self.quantized_sigma_bias is not None:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, 0.1, 128)

        perturbed_outputs = torch.nn.quantized.functional.linear(x,
                            weight=delta_weight, bias=bias, scale=0.1, zero_point=128)
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, 0.1, 128)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, 0.1, 128)
        out = out.dequantize()

        return out, 0
