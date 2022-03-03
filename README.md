<div align="center">

<img src="https://raw.githubusercontent.com/IntelLabs/bayesian-torch/5a4793ba0a54660c891df8af0d5729e840340a88/assets/bayesian-torch.png" width="500px">
<h2 >
A library for Bayesian neural network layers and uncertainty estimation in Deep Learning </a>
</h2>

[![python](https://img.shields.io/badge/python-3.7%2B-blue)](https://github.com/IntelLabs/bayesian-torch)
[![pytorch](https://img.shields.io/badge/pytorch-1.7.0%2B-orange)](https://github.com/IntelLabs/bayesian-torch)
[![version](https://img.shields.io/badge/release-0.2.1-green)](https://github.com/IntelLabs/bayesian-torch/releases)
[![license](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://github.com/IntelLabs/bayesian-torch/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/bayesian-torch?period=total&units=international_system&left_color=grey&right_color=darkblue&left_text=downloads)](https://pepy.tech/project/bayesian-torch)
<h4 align="center">
  <a href="https://github.com/IntelLabs/bayesian-torch#installing-bayesian-torch">Get Started</a> |
  <a href="https://github.com/IntelLabs/bayesian-torch#usage">Example usage</a> |
  <a href="https://github.com/IntelLabs/bayesian-torch/blob/main/doc/bayesian_torch.layers.md">Documentation</a> |
  <a href="https://github.com/IntelLabs/bayesian-torch#citing">Citing</a>   
</h4>

</div>

___

Bayesian-Torch is a library of neural network layers and utilities extending the core of PyTorch to enable Bayesian inference in deep learning models to quantify principled uncertainty estimates in model predictions.

## Overview
Bayesian-Torch is designed to be flexible and enables seamless extension of deterministic deep neural network model to corresponding Bayesian form by simply replacing the deterministic layers with Bayesian layers. It enables user to perform stochastic variational inference in deep neural networks. 

**Bayesian layers:**

* **[Variational layers with reparameterized Monte Carlo estimators](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/variational_layers)** [[Blundell et al. 2015](https://arxiv.org/abs/1505.05424)]

      
      LinearReparameterization 
      Conv1dReparameterization, Conv2dReparameterization, Conv3dReparameterization, ConvTranspose1dReparameterization, ConvTranspose2dReparameterization, ConvTranspose3dReparameterization
      LSTMReparameterization
      
* **[Variational layers with Flipout Monte Carlo estimators](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/flipout_layers)** [[Wen et al. 2018](https://arxiv.org/abs/1803.04386)]
      
      LinearFlipout 
      Conv1dFlipout, Conv2dFlipout, Conv3dFlipout, ConvTranspose1dFlipout, ConvTranspose2dFlipout, ConvTranspose3dFlipout
      LSTMFlipout



**Key features:**
* [dnn_to_bnn()](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/models/dnn_to_bnn.py#L127): An API to convert deterministic deep neural network (dnn) model of any architecture to Bayesian deep neural network (bnn) model, simplifying the model definition i.e. drop-in replacements  of Convolutional, Linear and LSTM layers to corresponding Bayesian layers. This will enable seamless conversion of existing topology of larger models to Bayesian deep neural network models for extending towards uncertainty-aware applications. 
* [MOPED](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/utils/util.py#L72): Specifying weight priors and variational posteriors in Bayesian neural networks with Empirical Bayes [[Krishnan et al. 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5875)]
* [AvUC](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/utils/avuc_loss.py): Accuracy versus Uncertainty Calibration loss [[Krishnan and Tickoo 2020](https://proceedings.neurips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf)]

## Installing Bayesian-Torch

**To install core library using `pip`:**
```
pip install bayesian-torch
```

**To install latest development version from source:**
```sh
git clone https://github.com/IntelLabs/bayesian-torch
cd bayesian-torch
pip install .
```
<!--
This code has been tested on PyTorch v1.8.1 LTS.

Dependencies:

- Create conda environment with python=3.7
- Install PyTorch and torchvision packages within conda environment following instructions from [PyTorch install guide](https://pytorch.org/get-started/locally/)
- conda install -c conda-forge accimage
- pip install tensorboard
- pip install scikit-learn
-->

## Usage
There are two ways to build Bayesian deep neural networks using Bayesian-Torch: 
1. Convert an existing deterministic deep neural network (dnn) model to Bayesian deep neural network (bnn) model with dnn_to_bnn() API
2. Define your custom model using the Bayesian layers ([Reparameterization](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/variational_layers) or [Flipout](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/flipout_layers))

(1) For instance, building Bayesian-ResNet18 from torchvision deterministic ResNet18 model is as simple as:
```
import torch
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}
    
model = torchvision.models.resnet18()
dnn_to_bnn(model, const_bnn_prior_parameters)
```
To use MOPED method i.e. setting the prior and initializing variational parameters from a pretrained deterministic model (helps training convergence of larger models):
```
const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}
    
model = torchvision.models.resnet18(pretrained=True)
dnn_to_bnn(model, const_bnn_prior_parameters)
```
Training snippet:
```
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

output = model(x_train)
kl = get_kl_loss(model)
ce_loss = criterion(output, y_train)
loss = ce_loss + kl / args.batch_size 

loss.backward()
optimizer.step()
```
Testing snippet:
```
model.eval()
with torch.no_grad():
    output_mc = []
    for mc_run in range(args.num_monte_carlo):
        logits = model(x_test)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_mc.append(probs)
    output = torch.stack(output_mc)  
    pred_mean = output.mean(dim=0)
    y_pred = torch.argmax(pred_mean, axis=-1)
    test_acc = (y_pred.data.cpu().numpy() == y_test.data.cpu().numpy()).mean()
```
Uncertainty Quantification:
```
from utils.util import predictive_entropy, mutual_information

predictive_uncertainty = predictive_entropy(output.data.cpu().numpy())
model_uncertainty = mutual_information(output.data.cpu().numpy())
```

(2) For building custom models, we have provided [example model implementations](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/models/bayesian) using the Bayesian layers.

## Example usage (training and evaluation of models)

We have provided [example usages](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/examples) and [scripts](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/scripts) to train/evaluate the models. The instructions for CIFAR10 examples is provided below, similar scripts for ImageNet and MNIST are available.
```
cd bayesian_torch
```
### Training

To train Bayesian ResNet on CIFAR10, run this command:

**Mean-field variational inference (Reparameterized Monte Carlo estimator)**
```train
sh scripts/train_bayesian_cifar.sh
```

**Mean-field variational inference (Flipout Monte Carlo estimator)**
```train
sh scripts/train_bayesian_flipout_cifar.sh
```

To train deterministic ResNet on CIFAR10, run this command:

**Vanilla**
```train
sh scripts/train_deterministic_cifar.sh
```


### Evaluation

To evaluate Bayesian ResNet on CIFAR10, run this command:

**Mean-field variational inference (Reparameterized Monte Carlo estimator)**
```test
sh scripts/test_bayesian_cifar.sh
```

**Mean-field variational inference (Flipout Monte Carlo estimator)**
```test
sh scripts/test_bayesian_flipout_cifar.sh
```

To evaluate deterministic ResNet on CIFAR10, run this command:

**Vanilla**
```test
sh scripts/test_deterministic_cifar.sh
```

## Citing

If you use this code, please cite as:
```sh
@software{krishnan2022bayesiantorch,
  author       = {Ranganath Krishnan and Pi Esposito and Mahesh Subedar},               
  title        = {Bayesian-Torch: Bayesian neural network layers for uncertainty estimation},
  month        = jan,
  year         = 2022,
  doi          = {10.5281/zenodo.5908307},
  url          = {https://doi.org/10.5281/zenodo.5908307}
  howpublished = {\url{https://github.com/IntelLabs/bayesian-torch}}
}
```
Accuracy versus Uncertainty Calibration (AvUC) loss
```sh
@inproceedings{NEURIPS2020_d3d94468,
 title = {Improving model calibration with accuracy versus uncertainty optimization},
 author = {Krishnan, Ranganath and Tickoo, Omesh},
 booktitle = {Advances in Neural Information Processing Systems},
 volume = {33},
 pages = {18237--18248},
 year = {2020},
 url = {https://proceedings.neurips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf}
 
}
```
MOdel Priors with Empirical Bayes using DNN (MOPED)
```sh
@inproceedings{krishnan2020specifying,
  title={Specifying weight priors in bayesian deep neural networks with empirical bayes},
  author={Krishnan, Ranganath and Subedar, Mahesh and Tickoo, Omesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={4477--4484},
  year={2020},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/5875}
}
```

This library and code is intended for researchers and developers, enables to quantify principled uncertainty estimates from deep learning model predictions using stochastic variational inference in Bayesian neural networks. 
Feedbacks, issues and contributions are welcome. Email to <ranganath.krishnan@intel.com> for any questions.
