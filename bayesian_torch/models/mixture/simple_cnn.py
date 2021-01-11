from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import Conv2dMixture
from bayesian_torch.layers import LinearMixture
"""
#for 95% acc
prior_mean_1=0.5
prior_variance_1=1
posterior_mu_init_1=0.5
posterior_rho_init_1=-3
prior_mean_2=0
prior_variance_2=1
posterior_mu_init_2=0
posterior_rho_init_2=-3
"""

prior_mean_1 = 0
prior_variance_1 = 0.5
posterior_mu_init_1 = 0.5
posterior_rho_init_1 = -6

prior_mean_2 = 0.1
prior_variance_2 = 0.5
posterior_mu_init_2 = 0.1
posterior_rho_init_2 = -6


class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dMixture(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            prior_mean_1=prior_mean_1,
            prior_variance_1=prior_variance_1,
            posterior_mu_init_1=posterior_mu_init_1,
            posterior_rho_init_1=posterior_rho_init_1,
            prior_mean_2=prior_mean_2,
            prior_variance_2=prior_variance_2,
            posterior_mu_init_2=posterior_mu_init_2,
            posterior_rho_init_2=posterior_rho_init_2,
        )
        self.conv2 = Conv2dMixture(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            prior_mean_1=prior_mean_1,
            prior_variance_1=prior_variance_1,
            posterior_mu_init_1=posterior_mu_init_1,
            posterior_rho_init_1=posterior_rho_init_1,
            prior_mean_2=prior_mean_2,
            prior_variance_2=prior_variance_2,
            posterior_mu_init_2=posterior_mu_init_2,
            posterior_rho_init_2=posterior_rho_init_2,
        )
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = LinearMixture(
            in_features=9216,
            out_features=128,
            prior_mean_1=prior_mean_1,
            prior_variance_1=prior_variance_1,
            posterior_mu_init_1=posterior_mu_init_1,
            posterior_rho_init_1=posterior_rho_init_1,
            prior_mean_2=prior_mean_2,
            prior_variance_2=prior_variance_2,
            posterior_mu_init_2=posterior_mu_init_2,
            posterior_rho_init_2=posterior_rho_init_2,
        )
        self.fc2 = LinearMixture(
            in_features=128,
            out_features=10,
            prior_mean_1=prior_mean_1,
            prior_variance_1=prior_variance_1,
            posterior_mu_init_1=posterior_mu_init_1,
            posterior_rho_init_1=posterior_rho_init_1,
            prior_mean_2=prior_mean_2,
            prior_variance_2=prior_variance_2,
            posterior_mu_init_2=posterior_mu_init_2,
            posterior_rho_init_2=posterior_rho_init_2,
        )

    def forward(self, x):
        kl_sum = 0
        x, kl = self.conv1(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.conv2(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)
        x = self.dropout2(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        output = F.log_softmax(x, dim=1)
        return output, kl
