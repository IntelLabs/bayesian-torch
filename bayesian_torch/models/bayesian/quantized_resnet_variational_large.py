'''
Bayesian ResNet for CIFAR10.

ResNet architecture ref:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from bayesian_torch.layers import QuantizedConv2dReparameterization
from bayesian_torch.layers import QuantizedLinearReparameterization
from torch.nn.quantized import BatchNorm2d as QuantizedBatchNorm2d
from torch.nn import Identity

__all__ = [
    'QResNet', 'qresnet18', 'qresnet34', 'qresnet50', 'qresnet101', 'qresnet152'
]

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = QuantizedConv2dReparameterization(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias)
        self.bn1 = QuantizedBatchNorm2d(planes)
        self.conv2 = QuantizedConv2dReparameterization(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)
        self.bn2 = QuantizedBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    QuantizedConv2dReparameterization(
                        in_channels=in_planes,
                        out_channels=self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=bias), QuantizedBatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        sh = self.shortcut(x.contiguous()).contiguous()
        new_scale = max(out.q_scale(), sh.q_scale())
        out = torch.ops.quantized.add(out, sh, new_scale, 0)
        # out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False):
        super(Bottleneck, self).__init__()
        self.conv1 = QuantizedConv2dReparameterization(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            bias=bias)
        self.bn1 =QuantizedBatchNorm2d(planes)
        self.conv2 = QuantizedConv2dReparameterization(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias)
        self.bn2 = QuantizedBatchNorm2d(planes)
        self.conv3 = QuantizedConv2dReparameterization(
            in_channels=planes,
            out_channels=planes * 4,
            kernel_size=1,
            bias=bias)
        self.bn3 = QuantizedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        new_scale = max(out.q_scale(), residual.q_scale())
        out = torch.ops.quantized.add(out, residual, new_scale, 0)
        out = self.relu(out)

        return out

class QResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, bias=False):
        super(QResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = QuantizedConv2dReparameterization(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=bias)
        self.bn1 = QuantizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bias=bias)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bias=bias)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bias=bias)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bias=bias)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = QuantizedLinearReparameterization(
            in_features=512 * block.expansion,
            out_features=num_classes,
        )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, blocks, stride=1, bias=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuantizedConv2dReparameterization(in_channels=self.inplanes,
                                         out_channels=planes * block.expansion,
                                         kernel_size=1,
                                         stride=stride,
                                         bias=bias),
                QuantizedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bias=bias))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bias=bias))

        return nn.Sequential(*layers)

    def quant_then_dequant(self, m, fuse_conv_bn=False): ## quantize only; need to rename this function
        for name, value in list(m._modules.items()):
            if m._modules[name]._modules:
                self.quant_then_dequant(m._modules[name], fuse_conv_bn=fuse_conv_bn)
                
            if "QuantizedConv" in m._modules[name].__class__.__name__:
                m._modules[name].quantize()
                m._modules[name].quantized_sigma_bias = None ### work around
                m._modules[name].dnn_to_bnn_flag = True ## since we don't compute kl in quantized models, this flag will be removed after refactoring

            if "QuantizedLinear" in m._modules[name].__class__.__name__:
                m._modules[name].quantize()
                m._modules[name].dnn_to_bnn_flag = True ## since we don't compute kl in quantized models, this flag will be removed after refactoring

            if fuse_conv_bn and "BatchNorm2d" in m._modules[name].__class__.__name__: # quite confusing, should be quantizedbatchnorm2d
                setattr(m, name, Identity())

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layer1:
            x = layer(x)

        for layer in self.layer2:
            x = layer(x)

        for layer in self.layer3:
            x = layer(x)

        for layer in self.layer4:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def qresnet18(pretrained=False, **kwargs):
    model = QResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def qresnet34(pretrained=False, **kwargs):
    model = QResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def qresnet50(pretrained=False, **kwargs):
    model = QResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def qresnet101(pretrained=False, **kwargs):
    model = QResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def qresnet152(pretrained=False, **kwargs):
    model = QResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model



def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(lambda p: p.requires_grad and len(p.data.size()) > 1,
                       net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('qresnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
