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
# Functions related to BNN to QBNN model conversion.
#
# @authors: Jun-Liang Lin
#
# ===============================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import bayesian_torch.layers as bayesian_layers
import torch
import torch.nn as nn
from torch.nn import Identity
from torch.nn.quantized import BatchNorm2d as QBatchNorm2d
from torch.nn import Module, Parameter


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

def qbnn_linear_layer(d):
    layer_type = "Quantized" + d.__class__.__name__
    layer_fn = getattr(bayesian_layers, layer_type)  # Get QBNN layer
    qbnn_layer = layer_fn(
        in_features=d.in_features,
        out_features=d.out_features,
    )
    qbnn_layer.__dict__.update(d.__dict__)

    if d.quant_prepare:
        qbnn_layer.quant_dict = nn.ModuleList()
        for qstub in d.qint_quant:
            qbnn_layer.quant_dict.append(nn.ParameterDict({'scale': torch.nn.Parameter(qstub.scale.float()), 'zero_point': torch.nn.Parameter(qstub.zero_point.float())}))
        qbnn_layer.quant_dict = qbnn_layer.quant_dict[2:]
        for qstub in d.quint_quant:
            qbnn_layer.quant_dict.append(nn.ParameterDict({'scale': torch.nn.Parameter(qstub.scale.float()), 'zero_point': torch.nn.Parameter(qstub.zero_point.float())}))

    qbnn_layer.quantize()
    if d.dnn_to_bnn_flag:
        qbnn_layer.dnn_to_bnn_flag = True
    return qbnn_layer

def qbnn_conv_layer(d):
    layer_type = "Quantized" + d.__class__.__name__
    layer_fn = getattr(bayesian_layers, layer_type)  # Get QBNN layer
    qbnn_layer = layer_fn(
        in_channels=d.in_channels,
        out_channels=d.out_channels,
        kernel_size=d.kernel_size,
        stride=d.stride,
        padding=d.padding,
        dilation=d.dilation,
        groups=d.groups,
    )
    qbnn_layer.__dict__.update(d.__dict__)

    if d.quant_prepare:
        qbnn_layer.quant_dict = nn.ModuleList()
        for qstub in d.qint_quant:
            qbnn_layer.quant_dict.append(nn.ParameterDict({'scale': torch.nn.Parameter(qstub.scale.float()), 'zero_point': torch.nn.Parameter(qstub.zero_point.float())}))
        qbnn_layer.quant_dict = qbnn_layer.quant_dict[2:]
        for qstub in d.quint_quant:
            qbnn_layer.quant_dict.append(nn.ParameterDict({'scale': torch.nn.Parameter(qstub.scale.float()), 'zero_point': torch.nn.Parameter(qstub.zero_point.float())}))

    qbnn_layer.quantize()
    if d.dnn_to_bnn_flag:
        qbnn_layer.dnn_to_bnn_flag = True
    return qbnn_layer

def qbnn_lstm_layer(d):
    layer_type = "Quantized" + d.__class__.__name__
    layer_fn = getattr(bayesian_layers, layer_type)  # Get QBNN layer
    qbnn_layer = layer_fn(
        in_features=d.input_size,
        out_features=d.hidden_size,
    )
    qbnn_layer.__dict__.update(d.__dict__)
    qbnn_layer.quantize()
    if d.dnn_to_bnn_flag:
        qbnn_layer.dnn_to_bnn_flag = True
    return qbnn_layer

def qbnn_batchnorm2d_layer(d):
    layer_fn = QBatchNorm2d  # Get QBNN layer
    qbnn_layer = layer_fn(
        num_features=d.num_features
    )
    qbnn_layer.__dict__.update(d.__dict__)
    # qbnn_layer.weight = Parameter(get_quantized_tensor(d.weight), requires_grad=False)
    # qbnn_layer.bias = Parameter(get_quantized_tensor(d.bias), requires_grad=False)
    # qbnn_layer.running_mean = Parameter(get_quantized_tensor(d.running_mean), requires_grad=False)
    # qbnn_layer.running_var = Parameter(get_quantized_tensor(d.running_var), requires_grad=False)
    # qbnn_layer.scale = Parameter(torch.tensor([0.1]), requires_grad=False)
    # qbnn_layer.zero_point = Parameter(torch.tensor([128]), requires_grad=False)
    return qbnn_layer


# batch norm folding
def batch_norm_folding(conv, bn):
    layer_type = "Quantized" + conv.__class__.__name__
    layer_fn = getattr(bayesian_layers, layer_type)  # Get QBNN layer
    qbnn_layer = layer_fn(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    qbnn_layer.__dict__.update(conv.__dict__)
    qbnn_layer.bn_weight = bn.weight
    qbnn_layer.bn_bias = bn.bias
    qbnn_layer.bn_running_mean = bn.running_mean
    qbnn_layer.bn_running_var = bn.running_var
    qbnn_layer.bn_eps = bn.eps
    qbnn_layer.quantize()
    if conv.dnn_to_bnn_flag:
        qbnn_layer.dnn_to_bnn_flag = True
    return qbnn_layer

# replaces linear and conv layers
def bnn_to_qbnn(m, fuse_conv_bn=False):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            if "Conv" in m._modules[name].__class__.__name__:
                setattr(m, name, qbnn_conv_layer(m._modules[name]))
            elif "Linear" in m._modules[name].__class__.__name__:
                setattr(m, name, qbnn_linear_layer(m._modules[name]))
            else:
                bnn_to_qbnn(m._modules[name], fuse_conv_bn=fuse_conv_bn)
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(m, name, qbnn_linear_layer(m._modules[name]))
        elif "LSTM" in m._modules[name].__class__.__name__:
            setattr(m, name, qbnn_lstm_layer(m._modules[name]))
        else:
            if fuse_conv_bn:
                if 'conv1' in m._modules.keys() and 'bn1' in m._modules.keys():
                    if 'Identity' not in m._modules['bn1'].__class__.__name__:
                        setattr(m, 'conv1', batch_norm_folding(m._modules['conv1'], m._modules['bn1']))
                        setattr(m, 'bn1', Identity())
                if 'conv2' in m._modules.keys() and 'bn2' in m._modules.keys():
                    if 'Identity' not in m._modules['bn2'].__class__.__name__:
                        setattr(m, 'conv2', batch_norm_folding(m._modules['conv2'], m._modules['bn2']))
                        setattr(m, 'bn2', Identity())
                if 'conv3' in m._modules.keys() and 'bn3' in m._modules.keys():
                    if 'Identity' not in m._modules['bn3'].__class__.__name__:
                        setattr(m, 'conv3', batch_norm_folding(m._modules['conv3'], m._modules['bn3']))
                        setattr(m, 'bn3', Identity())
                if 'downsample' in m._modules.keys():
                    if m._modules['downsample'].__class__.__name__=='Sequential' and len(m._modules['downsample'])==2:
                        if 'Identity' not in m._modules['downsample'][1].__class__.__name__:
                            m._modules['downsample'][0]=batch_norm_folding(m._modules['downsample'][0], m._modules['downsample'][1])
                            m._modules['downsample'][1]=Identity()
            else:
                if "Conv" in m._modules[name].__class__.__name__:
                    setattr(m, name, qbnn_conv_layer(m._modules[name]))
                
                elif "Batch" in m._modules[name].__class__.__name__:
                    setattr(m, name, qbnn_batchnorm2d_layer(m._modules[name]))

    return

if __name__ == "__main__":
    class FusionTest(nn.Module):
        def __init__(self):
            super(FusionTest, self).__init__()
            self.conv1 = bayesian_layers.Conv2dReparameterization(1,3,2,bias=False)
            self.bn1 = nn.BatchNorm2d(3)
        def forward(self, x):
            x = self.conv1(x)[0]
            x = self.bn1(x)
            return x
    m = FusionTest()
    m.conv1.rho_kernel = Parameter(torch.zeros(m.conv1.rho_kernel.shape)-100)
    m.eval()
    print(m)
    input = torch.randn(1,1,3,3)
    print(m(input))
    bnn_to_qbnn(m)
    print(m)
    if input.dtype!=torch.quint8:
        input = torch.quantize_per_tensor(input, 0.1, 128, torch.quint8)
    print(m(input))
