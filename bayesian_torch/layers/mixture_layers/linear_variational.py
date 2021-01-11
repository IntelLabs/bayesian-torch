# Copyright 2020 Intel Corporation
#
# Linear Reparameterization Layers with reparameterization estimator to perform
# mean-field Reparameterization inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
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


class LinearMixture(BaseMixtureLayer_):
    def __init__(self,
                 in_features,
                 out_features,
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
        Implements Linear layer with mixture reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
            prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
            posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
            prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
            prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
            posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
            posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearMixture, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean_1 = prior_mean_1
        self.prior_variance_1 = prior_variance_1
        self.posterior_mu_init_1 = posterior_mu_init_1  # mean of weight
        self.posterior_rho_init_1 = posterior_rho_init_1  # variance of weight --> sigma = log (1 + exp(rho))

        self.prior_mean_2 = prior_mean_2
        self.prior_variance_2 = prior_variance_2
        self.posterior_mu_init_2 = posterior_mu_init_2  # mean of weight
        self.posterior_rho_init_2 = posterior_rho_init_2

        self.bias = bias

        self.mu_weight_1 = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight_1 = Parameter(torch.Tensor(out_features, in_features))

        self.mu_weight_2 = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight_2 = Parameter(torch.Tensor(out_features, in_features))

        self.eta_weight = Parameter(torch.Tensor(out_features, in_features))

        self.register_buffer('eps_weight_1',
                             torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu_1',
                             torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_sigma_1',
                             torch.Tensor(out_features, in_features))

        self.register_buffer('eps_weight_2',
                             torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_mu_2',
                             torch.Tensor(out_features, in_features))
        self.register_buffer('prior_weight_sigma_2',
                             torch.Tensor(out_features, in_features))

        if bias:
            self.mu_bias_1 = Parameter(torch.Tensor(out_features))
            self.rho_bias_1 = Parameter(torch.Tensor(out_features))
            self.eta_bias = Parameter(torch.Tensor(out_features))

            self.register_buffer('eps_bias_1', torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu_1', torch.Tensor(out_features))
            self.register_buffer('prior_bias_sigma_1',
                                 torch.Tensor(out_features))

            self.mu_bias_2 = Parameter(torch.Tensor(out_features))
            self.rho_bias_2 = Parameter(torch.Tensor(out_features))

            self.register_buffer('eps_bias_2', torch.Tensor(out_features))
            self.register_buffer('prior_bias_mu_2', torch.Tensor(out_features))
            self.register_buffer('prior_bias_sigma_2',
                                 torch.Tensor(out_features))
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
        self.prior_weight_mu_1.fill_(self.prior_mean_1)
        self.prior_weight_sigma_1.fill_(self.prior_variance_1)
        self.prior_weight_mu_2.fill_(self.prior_mean_2)
        self.prior_weight_sigma_2.fill_(self.prior_variance_2)

        self.eta_weight.data.uniform_(0, 0.5)

        self.mu_weight_1.data.normal_(mean=self.posterior_mu_init_1, std=0.1)
        self.rho_weight_1.data.normal_(mean=self.posterior_rho_init_1, std=0.1)

        self.mu_weight_2.data.normal_(mean=self.posterior_mu_init_2, std=0.1)
        self.rho_weight_2.data.normal_(mean=self.posterior_rho_init_2, std=0.1)

        if self.mu_bias_1 is not None:
            self.eta_bias.data.uniform_(0, 0.5)
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
        sigma_weight_1 = torch.log1p(torch.exp(self.rho_weight_1))
        weight_1 = self.mu_weight_1 + (sigma_weight_1 *
                                       self.eps_weight_1.data.normal_())

        sigma_weight_2 = torch.log1p(torch.exp(self.rho_weight_2))
        weight_2 = self.mu_weight_2 + (sigma_weight_2 *
                                       self.eps_weight_2.data.normal_())

        weight = (self.eta_weight * weight_1) + (
            (1 - self.eta_weight) * weight_2)

        kl_weight = self.mixture_kl_div(
            self.mu_weight_1, sigma_weight_1, self.mu_weight_2, sigma_weight_2,
            self.prior_weight_mu_1, self.prior_weight_sigma_1,
            self.prior_weight_mu_2, self.prior_weight_sigma_2, self.eta_weight,
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

        out = F.linear(input, weight, bias)
        if self.mu_bias_1 is not None:
            kl = kl_weight + kl_bias
        else:
            kl = kl_weight

        return out, kl
