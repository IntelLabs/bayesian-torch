# Copyright (C) 2020 Intel Corporation
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


class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()

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
        #to avoid problem with cpu torch.log
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.sum()


class BaseMixtureLayer_(BaseVariationalLayer_):
    def __init__(self):
        super().__init__()

    def mixture_kl_div(self, mu_m1_d1, sigma_m1_d1, mu_m1_d2, sigma_m1_d2,
                       mu_m2_d1, sigma_m2_d1, mu_m2_d2, sigma_m2_d2, eta, w):
        """
        Calculates kl divergence between two mixtures of gaussians (Q || P), given a sample w from Q

        Parameters:
             * mu_m1_d1: torch.Tensor -> mu 1 parameter of distribution Q,
             * sigma_m1_d1 : torch.Tensor -> sigma 1 parameter of distribution Q,
             * mu_m1_d2 : torch.Tensor -> mu 2 parameter of distribution Q,
             * sigma_m1_d2 : torch.Tensor -> sigma 1 parameter of distribution Q,
             * mu_m2_d1: torch.Tensor -> mu 1 parameter of distribution P,
             * sigma_m2_d1: torch.Tensor -> sigma 1 parameter of distribution P,
             * mu_m2_d2: torch.Tensor -> mu 2 parameter of distribution P,
             * sigma_m2_d2: torch.Tensor -> sigma 2 parameter of distribution P,
             * eta: torch.Tensor -> mixture proportions of distribution Q,

        returns torch.Tensor of shape 0
        """
        m1_d1 = distributions.Normal(loc=mu_m1_d1, scale=sigma_m1_d1)
        m1_d2 = distributions.Normal(loc=mu_m1_d2, scale=sigma_m1_d2)

        m2_d1 = distributions.Normal(loc=mu_m2_d1, scale=sigma_m2_d1)
        m2_d2 = distributions.Normal(loc=mu_m2_d2, scale=sigma_m2_d2)

        log_prob_m1 = torch.log(
            (eta.abs() * torch.exp(m1_d1.log_prob(w)) +
             (1 - eta).abs() * torch.exp(m1_d2.log_prob(w))) + 0.5)
        log_prob_m2 = torch.log(0.5 * torch.exp(m2_d1.log_prob(w)) +
                                0.5 * torch.exp(m2_d2.log_prob(w)))

        return log_prob_m1 - log_prob_m2
