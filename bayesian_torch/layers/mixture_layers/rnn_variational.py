from .linear_variational import LinearMixture
from ..base_variational_layer import BaseMixtureLayer_

import torch

# Copyright 2020 Intel Corporation
#
# LSTM Radial Reparameterization Layer with radial reparameterization estimator to perform
# mean-field Reparameterization inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
# required to compute Evidence Lower Bound (ELBO) loss for Reparameterization inference.
#
# @authors: ranganath.krishnan@intel.com, piero.esposito@intel.com
#
# ======================================================================================


class LSTMMixture(BaseMixtureLayer_):
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
        Implements LSTM layer with reparameterization trick.

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
        super().__init__()

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

        self.ih = LinearMixture(prior_mean_1=prior_mean_1,
                                prior_variance_1=prior_variance_1,
                                posterior_mu_init_1=posterior_mu_init_1,
                                posterior_rho_init_1=posterior_rho_init_1,
                                prior_mean_2=prior_mean_2,
                                prior_variance_2=prior_variance_2,
                                posterior_mu_init_2=posterior_mu_init_2,
                                posterior_rho_init_2=posterior_rho_init_2,
                                in_features=in_features,
                                out_features=out_features * 4,
                                bias=bias)

        self.hh = LinearMixture(prior_mean_1=prior_mean_1,
                                prior_variance_1=prior_variance_1,
                                posterior_mu_init_1=posterior_mu_init_1,
                                posterior_rho_init_1=posterior_rho_init_1,
                                prior_mean_2=prior_mean_2,
                                prior_variance_2=prior_variance_2,
                                posterior_mu_init_2=posterior_mu_init_2,
                                posterior_rho_init_2=posterior_rho_init_2,
                                in_features=out_features,
                                out_features=out_features * 4,
                                bias=bias)

    def forward(self, X, hidden_states=None):

        batch_size, seq_size, _ = X.size()

        hidden_seq = []
        c_ts = []

        if hidden_states is None:
            h_t, c_t = (torch.zeros(batch_size,
                                    self.out_features).to(X.device),
                        torch.zeros(batch_size,
                                    self.out_features).to(X.device))
        else:
            h_t, c_t = hidden_states

        HS = self.out_features
        kl = 0
        for t in range(seq_size):
            x_t = X[:, t, :]

            ff_i, kl_i = self.ih(x_t)
            ff_h, kl_h = self.hh(h_t)
            gates = ff_i + ff_h

            kl += kl_i + kl_h

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
            c_ts.append(c_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        c_ts = torch.cat(c_ts, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_ts = c_ts.transpose(0, 1).contiguous()

        return hidden_seq, (hidden_seq, c_ts), kl
