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
# ===============================================================================================
import torch
import torch.nn as nn
import torch.distributions as distributions
from itertools import repeat
import collections

def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()
    
    def jsg_div(self, mu_q, sigma_q, mu_p, sigma_p, alpha=0.5):
        '''
        Calculates skew geometric jenson-shannon divergence between two gaussians (Q||P)

        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P

        returns torch.Tensor of shape 0
        '''
        
        sigma_0_alpha = (sigma_q.pow(2) * sigma_p.pow(2)) \
                            / ((1-alpha)*sigma_q.pow(2) + alpha*sigma_p.pow(2))
        
        mu_0_alpha = sigma_0_alpha * ((alpha*mu_q/sigma_q.pow(2)) \
                                      + ((1-alpha)*mu_p/(sigma_p.pow(2))))
    
        term1 = ((1-alpha)*sigma_q.pow(2) + alpha*sigma_p.pow(2)) / sigma_0_alpha
        
        term2 = torch.log(sigma_0_alpha / (torch.pow(sigma_q, 2-2*alpha) \
                                           * sigma_p.pow(2*alpha)))
        
        term3 = (1-alpha)*(mu_0_alpha - mu_q).pow(2) / sigma_0_alpha
        
        term4 = alpha*(mu_0_alpha - mu_p).pow(2) / sigma_0_alpha
    
        jsg_divergence = 0.5 * (term1 + term2 + term3 + term4 - 1)
        
        return jsg_divergence.mean()
