# Bayesian-Torch: Bayesian neural network layers for uncertainty estimation
**[Get started](https://github.com/IntelLabs/bayesian-torch#installation)** | **[Example usage](https://github.com/IntelLabs/bayesian-torch#usage)** | **[Documentation](https://github.com/IntelLabs/bayesian-torch/blob/main/doc/bayesian_torch.layers.md)** | **[License](https://github.com/IntelLabs/bayesian-torch/blob/main/LICENSE)** | **[Citing](https://github.com/IntelLabs/bayesian-torch#citing)** 

### Bayesian layers and utilities to perform stochastic variational inference in PyTorch

Bayesian-Torch is a library of neural network layers and utilities extending the core of PyTorch to enable the user to perform stochastic variational inference in Bayesian deep neural networks.
Bayesian-Torch is designed to be flexible and seamless in extending a deterministic deep neural network architecture to corresponding Bayesian form by simply replacing the deterministic layers with Bayesian layers. 


The repository has implementations for the following Bayesian layers:
- [x] **[Variational layers with reparameterized Monte Carlo estimators](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/variational_layers)** [[Blundell et al. 2015](https://arxiv.org/abs/1505.05424)]

      
      LinearVariational 
      Conv1dVariational, Conv2dVariational, Conv3dVariational, ConvTranspose1dVariational, ConvTranspose2dVariational, ConvTranspose3dVariational
      LSTMVariational
      
- [x] **[Variational layers with Flipout Monte Carlo estimators](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/flipout_layers)** [[Wen et al. 2018](https://arxiv.org/abs/1803.04386)]
      
      LinearFlipout 
      Conv1dFlipout, Conv2dFlipout, Conv3dFlipout, ConvTranspose1dFlipout, ConvTranspose2dFlipout, ConvTranspose3dFlipout
      LSTMFlipout

<!--
- [ ] **[Radial BNN layers](bayesian_torch/layers/radial_layers)** [[Farquhar et al. 2020](https://arxiv.org/abs/1907.00865)]

      LinearRadial
      Conv1dRadial, Conv2dRadial, Conv3dRadial, ConvTranspose1dRadial, ConvTranspose2dRadial, ConvTranspose3dRadial
      LSTMRadial

- [ ] **Variational layers with Gaussian mixture model (GMM) posteriors using reparameterized Monte Carlo estimators** (in `pre-alpha`)

      LinearMixture
      Conv1dMixture, Conv2dMixture, Conv3dMixture, ConvTranspose1dMixture, ConvTranspose2dMixture, ConvTranspose3dMixture
      LSTMMixture
-->

<!--
Please refer to [documentation](doc/bayesian_torch.layers.md#layers) of Bayesian layers for details.
-->

Other features include:
- [x] [dnn_to_bnn()](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/models/dnn_to_bnn.py#L127): An API to convert deterministic deep neural network (dnn) model of any architecture to Bayesian deep neural network (bnn) model, simplifying the model definition i.e. drop-in replacements  of Convolutional, Linear and LSTM layers to corresponding Bayesian layers. This will enable seamless conversion of existing topology of larger models to Bayesian deep neural network models for extending towards uncertainty-aware applications. 
- [x] [MOPED](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/utils/util.py#L72): Specifying weight priors and variational posteriors in Bayesian neural networks with Empirical Bayes [[Krishnan et al. 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5875)]
- [x] [AvUC](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/utils/avuc_loss.py): Accuracy versus Uncertainty Calibration loss [[Krishnan and Tickoo 2020](https://proceedings.neurips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf)]

## Installation
<!--
**To install from PyPI:**
```
pip install bayesian-torch
```
-->
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
1. Convert an existing deterministic deep neural network (dnn) model to Bayesian deep neural network (bnn) model with dnn_to_bnn()
2. Define your custom model using the Bayesian layers ([Flipout](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/flipout_layers) or [Reparameterization](https://github.com/IntelLabs/bayesian-torch/tree/main/bayesian_torch/layers/variational_layers))

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
To use MOPED method, setting the prior and initializing variational parameters from a pretrained deterministic model (helps training convergence of larger models):
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
@misc{krishnan2020bayesiantorch,
    author = {Ranganath Krishnan and Piero Esposito and Mahesh Subedar},
    title = {Bayesian-Torch: Bayesian neural network layers for uncertainty estimation},
    year = {2020},
    publisher = {GitHub},
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

This code is intended for researchers and developers, enables to quantify principled uncertainty estimates from deep neural network predictions using stochastic variational inference in Bayesian neural networks. 
Feedbacks, issues and contributions are welcome. Email to <ranganath.krishnan@intel.com> for any questions.
 

