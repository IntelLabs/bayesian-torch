# Bayesian-Torch: Bayesian neural network layers for uncertainty estimation
**[Get started](#Installation)** | **[Example usage](#example-usage)** | **[Documentation](doc/bayesian_torch.layers.md)** | **[License](LICENSE)** | **[Citing](#citing)** 

### Bayesian layers and utilities to perform stochastic variational inference in PyTorch

Bayesian-Torch is a library of neural network layers and utilities extending the core of PyTorch to enable the user to perform stochastic variational inference in Bayesian deep neural networks.
Bayesian-Torch is designed to be flexible and seamless in extending a deterministic deep neural network architecture to corresponding Bayesian form by simply replacing the deterministic layers with Bayesian layers. 


The repository has implementations for the following Bayesian layers:
- [x] **[Variational layers with reparameterized Monte Carlo estimators](bayesian_torch/layers/variational_layers)** [[Blundell et al. 2015](https://arxiv.org/abs/1505.05424)]

      
      LinearVariational 
      Conv1dVariational, Conv2dVariational, Conv3dVariational, ConvTranspose1dVariational, ConvTranspose2dVariational, ConvTranspose3dVariational
      LSTMVariational
      
- [x] **[Variational layers with Flipout Monte Carlo estimators](bayesian_torch/layers/flipout_layers)** [[Wen et al. 2018](https://arxiv.org/abs/1803.04386)]
      
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

Please refer to [documentation](doc/bayesian_torch.layers.md#layers) of Bayesian layers for details.

Other features include:
- [x] [AvUC](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/utils/avuc_loss.py): Accuracy versus Uncertainty Calibration loss [[Krishnan and Tickoo 2020](https://proceedings.neurips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf)]
- [x] [MOPED](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/utils/util.py#L72): specifying weight priors and variational posteriors with Empirical Bayes [[Krishnan et al. 2019](https://arxiv.org/abs/1906.05323)]
- [x] [dnn_to_bnn](https://github.com/IntelLabs/bayesian-torch/blob/main/bayesian_torch/models/dnn_to_bnn.py#L127): An API to convert deterministic deep neural network (dnn) model of any architecture to Bayesian deep neural network (bnn) model, simplifying the model definition i.e. drop-in replacements  of Convolutional, Linear and LSTM layers to corresponding Bayesian layers. This will enable seamless conversion of existing topology of larger models to Bayesian deep neural network models for extending towards uncertainty-aware applications. 

## Installation


**Install from source:**
```sh
git clone https://github.com/IntelLabs/bayesian-torch
cd bayesian-torch
pip install .
```
This code has been tested on PyTorch v1.8.1 LTS.

Dependencies:

- Create conda environment with python=3.7
- Install PyTorch and torchvision packages within conda environment following instructions from [PyTorch install guide](https://pytorch.org/get-started/locally/)
- conda install -c conda-forge accimage
- pip install tensorboard
- pip install scikit-learn

## Example usage
We have provided [example model implementations](bayesian_torch/models/bayesian) using the Bayesian layers.

We also provide [example usages](bayesian_torch/examples) and [scripts](bayesian_torch/scripts) to train/evaluate the models. The instructions for CIFAR10 examples is provided below, similar scripts for ImageNet and MNIST are available.

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

**Contributors**
- Ranganath Krishnan 
- Piero Esposito 
- Mahesh Subedar

This code is intended for researchers and developers, enables to quantify principled uncertainty estimates from deep neural network predictions using stochastic variational inference in Bayesian neural networks. 
Feedbacks, issues and contributions are welcome. Email to <ranganath.krishnan@intel.com> for any questions.
 

