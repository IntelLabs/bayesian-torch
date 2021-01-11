from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import Conv2dFlipout
from bayesian_torch.layers import LinearFlipout

prior_mu = 0
prior_sigma = 0.05
posterior_mu_init = 0
posterior_rho_init = -7.0  #-6.0


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = Conv2dFlipout(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.conv2 = Conv2dFlipout(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = LinearFlipout(
            in_features=9216,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.fc2 = LinearFlipout(in_features=128,
                                 out_features=10,
                                 prior_mean=prior_mu,
                                 prior_variance=prior_sigma,
                                 posterior_mu_init=posterior_mu_init,
                                 posterior_rho_init=posterior_rho_init)

    def forward(self, x):
        kl_sum = 0
        x, kl = self.conv1(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.conv2(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)
        #x = self.dropout2(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        output = F.log_softmax(x, dim=1)
        return output, kl_sum
