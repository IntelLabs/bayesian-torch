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
# Utily functions for variational inference in Bayesian deep neural networks
#
# @authors: Ranganath Krishnan
#
# ===============================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np

def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)


def predictive_entropy(mc_preds):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return entropy(np.mean(mc_preds, axis=0))


def mutual_information(mc_preds):
    """
    Compute the difference between the entropy of the mean of the
    predictive distribution and the mean of the entropy.
    """
    mutual_info = entropy(np.mean(mc_preds, axis=0)) - np.mean(entropy(mc_preds),
                                                               axis=0)
    return mutual_info


def get_rho(sigma, delta):
    """
    sigma is represented by softplus function  'sigma = log(1 + exp(rho))' to make sure it 
    remains always positive and non-transformed 'rho' gets updated during backprop.
    """
    rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
    return rho


def MOPED(model, det_model, det_checkpoint, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
 
    Example implementation for Bayesian model with variational layers.

    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo. Specifying Weight Priors in 
        Bayesian Deep Neural Networks with Empirical Bayes. Proceedings of the AAAI 
        Conference on Artificial Intelligence. AAAI 2020. 
        https://arxiv.org/abs/1906.05323
    """
    det_model.load_state_dict(torch.load(det_checkpoint))
    for (idx, layer), (det_idx,
                       det_layer) in zip(enumerate(model.modules()),
                                         enumerate(det_model.modules())):
        if (str(layer) == 'Conv1dReparameterization()'
                or str(layer) == 'Conv2dReparameterization()'
                or str(layer) == 'Conv3dReparameterization()'
                or str(layer) == 'ConvTranspose1dReparameterization()'
                or str(layer) == 'ConvTranspose2dReparameterization()'
                or str(layer) == 'ConvTranspose3dReparameterization()'
                or str(layer) == 'Conv1dFlipout()'
                or str(layer) == 'Conv2dFlipout()'
                or str(layer) == 'Conv3dFlipout()'
                or str(layer) == 'ConvTranspose1dFlipout()'
                or str(layer) == 'ConvTranspose2dFlipout()'
                or str(layer) == 'ConvTranspose3dFlipout()'):
            #set the priors
            layer.prior_weight_mu = det_layer.weight.data
            if layer.prior_bias_mu is not None:
               layer.prior_bias_mu = det_layer.bias.data

            #initialize surrogate posteriors
            layer.mu_kernel.data = det_layer.weight.data
            layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
            if layer.mu_bias is not None:
               layer.mu_bias.data = det_layer.bias.data
               layer.rho_bias.data = get_rho(det_layer.bias.data, delta)
        elif (str(layer) == 'LinearReparameterization()'
                or str(layer) == 'LinearFlipout()'):
            #set the priors
            layer.prior_weight_mu = det_layer.weight.data
            if layer.prior_bias_mu is not None:
               layer.prior_bias_mu.data = det_layer.bias

            #initialize the surrogate posteriors
            layer.mu_weight.data = det_layer.weight.data
            layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
            if layer.mu_bias is not None:
               layer.mu_bias.data = det_layer.bias.data
               layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

        elif str(layer).startswith('Batch'):
            #initialize parameters
            layer.weight.data = det_layer.weight.data
            if layer.bias is not None:
               layer.bias.data = det_layer.bias
            layer.running_mean.data = det_layer.running_mean.data
            layer.running_var.data = det_layer.running_var.data
            layer.num_batches_tracked.data = det_layer.num_batches_tracked.data

    model.state_dict()
    return model
