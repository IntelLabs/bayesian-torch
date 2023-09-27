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
# Define prepare and convert function
#

import torch
import torch.nn as nn
from bayesian_torch.models.bayesian.resnet_variational_large import (
    BasicBlock,
    Bottleneck,
    ResNet,
)
from typing import Any, List, Optional, Type, Union
from torch import Tensor
from bayesian_torch.models.bnn_to_qbnn import bnn_to_qbnn
from torch.nn import BatchNorm2d
# import copy

__all__ = [
    "prepare",
    "convert",
]

class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out


class QuantizableResNet(ResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)

        x= self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layer1:
            x=layer(x)

        for layer in self.layer2:
            x = layer(x)

        for layer in self.layer3:
            x = layer(x)

        for layer in self.layer4:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)


        # x = self.dequant(x)
        return x



def enable_prepare(m):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            enable_prepare(m._modules[name])
        elif "Reparameterization" in m._modules[name].__class__.__name__ or "Flipout" in m._modules[name].__class__.__name__:
            prepare = getattr(m._modules[name], "prepare", None)
            if callable(prepare):
                m._modules[name].prepare()
                m._modules[name].dnn_to_bnn_flag=True
        elif "BatchNorm2dLayer" in m._modules[name].__class__.__name__: # replace BatchNorm2dLayer with BatchNorm2d in downsample
            layer_fn = BatchNorm2d  # Get QBNN layer
            bn_layer = layer_fn(
                num_features=m._modules[name].num_features
            )
            bn_layer.__dict__.update(m._modules[name].__dict__)
            setattr(m, name, bn_layer)
            


def prepare(model):
    """
    1. construct quantizable model
    2. traverse the model to enable the prepare function in each layer
    3. run torch.quantize.prepare()
    """
    qmodel = QuantizableResNet(QuantizableBottleneck, [3, 4, 6, 3])
    qmodel.load_state_dict(model.module.state_dict())
    qmodel.eval()
    enable_prepare(qmodel)
    qmodel.qconfig = torch.quantization.get_default_qconfig("onednn")
    qmodel = torch.quantization.prepare(qmodel)

    return qmodel

def convert(model):
    qmodel = torch.quantization.convert(model) # torch layers
    bnn_to_qbnn(qmodel) # bayesian layers
    return qmodel
