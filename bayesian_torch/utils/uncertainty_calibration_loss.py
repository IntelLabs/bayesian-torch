###########################################################################
# Uncertainty Calibration Loss Functions
#
# This module provides efficient, vectorized implementations of
# uncertainty calibration loss functions to be used as secondary loss
# components during model training. These losses are designed to align
# model predictions with uncertainty estimates, improving reliability
# and interpretability.
#
# Implemented Loss Functions:
# * EaULoss: Error aligned Uncertainty (EaU) calibration loss [regression]
# * EaCLoss: Error aligned Confidence (EaC) calibration loss [regression]
# * AvULoss: Accuracy versus Uncertainty (AvU) calibration loss [classification]
#
# References:
# * https://arxiv.org/abs/2012.07923
# * https://link.springer.com/chapter/10.1007/978-3-031-25072-9_31
#
# @authors: Ranganath Krishnan
###########################################################################

import torch.nn.functional as F
import torch
from torch import nn


class EaULoss(nn.Module):
    """
    Module for computing Error aligned Uncertainty Calibration Loss (EaULoss).

    Computes the EaU loss, which measures the alignment between
    model prediction errors and uncertainty estimates. The loss is calculated
    based on four components:
      - LC: Low error, low uncertainty (certain)
      - LU: Low error, high uncertainty (uncertain)
      - HC: High error, low uncertainty (certain)
      - HU: High error, high uncertainty (uncertain)

    The loss encourages the model to produce uncertainty estimates that align
    with the prediction errors, improving the reliability of uncertainty
    quantification.

    Args:
        error (torch.Tensor): Tensor of prediction errors.
        unc (torch.Tensor): Tensor of uncertainty estimates.
        error_th (float): Threshold for categorizing low and high errors.
        unc_th (float): Threshold for categorizing low and high uncertainty.

    Returns:
        torch.Tensor: The computed EaU loss value.

    Reference:
         https://link.springer.com/chapter/10.1007/978-3-031-25072-9_31
    """

    def __init__(self, beta=1):
        super(EaULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def forward(self, error, unc, error_th, unc_th):
        eau_loss = torch.zeros(1, device=unc.device)

        low_error = torch.le(error, error_th).squeeze()
        certain = torch.le(unc, unc_th).squeeze()
        high_error = torch.gt(error, error_th).squeeze()
        uncertain = torch.gt(unc, unc_th).squeeze()

        lc_index = torch.logical_and(low_error, certain).nonzero().squeeze()
        lu_index = torch.logical_and(low_error, uncertain).nonzero().squeeze()
        hc_index = torch.logical_and(high_error, certain).nonzero().squeeze()
        hu_index = torch.logical_and(high_error, uncertain).nonzero().squeeze()

        """ LC: low error and certain """
        try:
            lc_error = torch.index_select(error, 0, lc_index)
            lc_unc = torch.index_select(unc, 0, lc_index)
            n_lc = torch.dot(1 - torch.tanh(lc_error), 1 - torch.tanh(lc_unc))
        except BaseException:
            n_lc = torch.zeros(1, device=unc.device)

        """ LU: low error and uncertain """
        try:
            lu_error = torch.index_select(error, 0, lu_index)
            lu_unc = torch.index_select(unc, 0, lu_index)
            n_lu = torch.dot(1 - torch.tanh(lu_error), torch.tanh(lu_unc))
        except BaseException:
            n_lu = torch.zeros(1, device=unc.device)

        """ HC: high error and certain """
        try:
            hc_error = torch.index_select(error, 0, hc_index)
            hc_unc = torch.index_select(unc, 0, hc_index)
            n_hc = torch.dot(torch.tanh(hc_error), 1 - torch.tanh(hc_unc))
        except BaseException:
            n_hc = torch.zeros(1, device=unc.device)

        """ HU: high error and uncertain """
        try:
            hu_error = torch.index_select(error, 0, hu_index)
            hu_unc = torch.index_select(unc, 0, hu_index)
            n_hu = torch.dot(torch.tanh(hu_error), torch.tanh(hu_unc))
        except BaseException:
            n_hu = torch.zeros(1, device=unc.device)

        eau = (n_lc + n_hu) / (n_lc + n_lu + n_hc + n_hu + self.eps)
        eau_loss = -1 * self.beta * torch.log(eau + self.eps)
        return eau_loss


class EaCLoss(nn.Module):
    """
    Error aligned Confidence Calibration Loss (EaCLoss).

    This class implements the EaC loss, which measures the alignment between
    model prediction errors and confidence estimates. The loss is calculated
    based on four components:
      - LC: Low error, high confidence (certain)
      - LU: Low error, low confidence (uncertain)
      - HC: High error, high confidence (certain)
      - HU: High error, low confidence (uncertain)

    The EaC loss encourages the model to produce confidence estimates that
    align with the prediction errors, improving the reliability of confidence
    quantification.

    Inputs:
        - error (torch.Tensor): Tensor of prediction errors.
        - conf (torch.Tensor): Tensor of confidence estimates.
        - error_th (float): Threshold for categorizing low and high errors.
        - conf_th (float): Threshold for categorizing low and high confidence.

    Outputs:
        - torch.Tensor: The computed EaC loss value.
    """

    def __init__(self, beta=1):
        super(EaCLoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def forward(self, error, conf, error_th, conf_th):
        eac_loss = torch.zeros(1, device=conf.device)

        low_error = torch.le(error, error_th).squeeze()
        certain = torch.gt(conf, conf_th).squeeze()
        high_error = torch.gt(error, error_th).squeeze()
        uncertain = torch.le(conf, conf_th).squeeze()

        lc_index = torch.logical_and(low_error, certain).nonzero().squeeze()
        lu_index = torch.logical_and(low_error, uncertain).nonzero().squeeze()
        hc_index = torch.logical_and(high_error, certain).nonzero().squeeze()
        hu_index = torch.logical_and(high_error, uncertain).nonzero().squeeze()

        """ LC: low error and certain """
        try:
            lc_error = torch.index_select(error, 0, lc_index)
            lc_conf = torch.index_select(conf, 0, lc_index)
            n_lc = torch.dot(1 - torch.tanh(lc_error), lc_conf)
        except BaseException:
            n_lc = torch.zeros(1, device=conf.device)

        """ LU: low error and uncertain """
        try:
            lu_error = torch.index_select(error, 0, lu_index)
            lu_conf = torch.index_select(conf, 0, lu_index)
            n_lu = torch.dot(1 - torch.tanh(lu_error), 1 - lu_conf)
        except BaseException:
            n_lu = torch.zeros(1, device=conf.device)

        """ HC: high error and certain """
        try:
            hc_error = torch.index_select(error, 0, hc_index)
            hc_conf = torch.index_select(conf, 0, hc_index)
            n_hc = torch.dot(torch.tanh(hc_error), hc_conf)
        except BaseException:
            n_hc = torch.zeros(1, device=conf.device)

        """ HU: high error and uncertain """
        try:
            hu_error = torch.index_select(error, 0, hu_index)
            hu_conf = torch.index_select(conf, 0, hu_index)
            n_hu = torch.dot(torch.tanh(hu_error), 1 - hu_conf)
        except BaseException:
            n_hu = torch.zeros(1, device=conf.device)

        eac = (n_lc + n_hu) / (n_lc + n_lu + n_hc + n_hu + self.eps)
        eac_loss = -1 * self.beta * torch.log(eac + self.eps)
        return eac_loss


class AvULoss(nn.Module):
    """
    Calculates Accuracy vs Uncertainty Loss of a model.
    The input to this loss is torch tensors of logits from the model, true labels,
    and the uncertainty threshold

    Reference:
         https://arxiv.org/abs/2012.07923
    """

    def __init__(self, beta=1):
        super(AvULoss, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def entropy(self, prob):
        return -1 * torch.sum(prob * torch.log(prob + self.eps), dim=-1)

    def forward(self, logits, labels, unc_th):
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, 1)
        unc = self.entropy(probs)

        avu_loss = torch.zeros(1, device=logits.device)

        accurate = torch.eq(predictions, labels).squeeze()
        certain = torch.le(unc, unc_th).squeeze()
        inaccurate = torch.ne(predictions, labels).squeeze()
        uncertain = torch.gt(unc, unc_th).squeeze()

        ac_index = torch.logical_and(accurate, certain).nonzero().squeeze()
        au_index = torch.logical_and(accurate, uncertain).nonzero().squeeze()
        ic_index = torch.logical_and(inaccurate, certain).nonzero().squeeze()
        iu_index = torch.logical_and(inaccurate, uncertain).nonzero().squeeze()

        """ accurate and certain """
        try:
            ac_conf = torch.index_select(confidences, 0, ac_index)
            ac_unc = torch.index_select(unc, 0, ac_index)
            n_ac = torch.dot(ac_conf, 1 - torch.tanh(ac_unc))
        except BaseException:
            n_ac = torch.zeros(1, device=logits.device)

        """ accurate and uncertain """
        try:
            au_conf = torch.index_select(confidences, 0, au_index)
            au_unc = torch.index_select(unc, 0, au_index)
            n_au = torch.dot(au_conf, torch.tanh(au_unc))
        except BaseException:
            n_au = torch.zeros(1, device=logits.device)

        """ inaccurate and certain """
        try:
            ic_conf = torch.index_select(confidences, 0, ic_index)
            ic_unc = torch.index_select(unc, 0, ic_index)
            n_ic = torch.dot(1 - ic_conf, 1 - torch.tanh(ic_unc))
        except BaseException:
            n_ic = torch.zeros(1, device=logits.device)

        """ inaccurate and uncertain """
        try:
            iu_conf = torch.index_select(confidences, 0, iu_index)
            iu_unc = torch.index_select(unc, 0, iu_index)
            n_iu = torch.dot(1 - iu_conf, torch.tanh(iu_unc))
        except BaseException:
            n_iu = torch.zeros(1, device=logits.device)

        avu = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu + self.eps)
        avu_loss = -1 * self.beta * torch.log(avu + self.eps)
        return avu_loss
