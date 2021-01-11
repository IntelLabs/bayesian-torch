# bayesian_torch.layers module
A set of Bayesian neural network layers to perform stochastic variational inference

- Variational layers with reparameterized Monte Carlo estimators [[Blundell et al. 2015](https://arxiv.org/abs/1505.05424)]
- Variational layers with Flipout Monte Carlo estimators [[Wen et al. 2018](https://arxiv.org/abs/1803.04386)]
- Radial BNN layers [[Farquhar et al. 2020](https://arxiv.org/abs/1907.00865)]
- Variational layers with Gaussian mixture model (GMM) posteriors using reparameterized Monte Carlo estimators (in pre-alpha)

# Layers


 * [BaseVariationalLayer_](#class-basevariationallayer_torchnnmodule)
 * [LinearReparameterization](#class-linearreparameterization)
 * [Conv1dReparameterization](#class-conv1dreparameterization)
 * [Conv2dReparameterization](#class-conv2dreparameterization)
 * [Conv3dReparameterization](#class-conv3dreparameterization)
 * [ConvTranspose1dReparameterization](#class-convtranspose1dreparameterization)
 * [ConvTranspose2dReparameterization](#class-convtranspose2dreparameterization)
 * [ConvTranspose3dReparameterization](#class-convtranspose3dreparameterization)
 * [LSTMReparameterization](#class-lstmreparameterization)


 * [LinearFlipout](#class-linearflipout)
 * [Conv1dFlipout](#class-conv1dflipout)
 * [Conv2dFlipout](#class-conv2dflipout)
 * [Conv3dFlipout](#class-conv3dflipout)
 * [ConvTranspose1dFlipout](#class-convtranspose1dflipout)
 * [ConvTranspose2dFlipout](#class-convtranspose2dflipout)
 * [ConvTranspose3dFlipout](#class-convtranspose3dflipout)
 * [LSTMFlipout](#class-lstmflipout)
  
 
 * [LinearRadial](#class-linearradial)
 * [Conv1dRadial](#class-conv1dradial)
 * [Conv2dRadial](#class-conv2dradial)
 * [Conv3dRadial](#class-conv3dradial)
 * [ConvTranspose1dRadial](#class-convtranspose1dradial)
 * [ConvTranspose2dRadial](#class-convtranspose2dradial)
 * [ConvTranspose3dRadial](#class-convtranspose3dradial)
 * [LSTMRadial](#class-lstmradial)
 
 
 * [LinearMixture](#class-linearmixture)
 * [Conv1dMixture](#class-conv1dmixture)
 * [Conv2dMixture](#class-conv2dmixture)
 * [Conv3dMixture](#class-conv3dmixture)
 * [ConvTranspose1dMixture](#class-convtranspose1dmixture)
 * [ConvTranspose2dMixture](#class-convtranspose2dmixture)
 * [ConvTranspose3dMixture](#class-convtranspose3dmixture)
 * [LSTMMixture](#class-lstmmixture)




## class BaseVariationalLayer_(torch.nn.Module)
Abstract class which inherits from torch.nn.Module
#### kl_div(mu_q, sigma_q, mu_p, sigma_p)
Calculates the Kullback-Leibler divergence from distribution normal Q (parametrized mu_q, sigma_q) to distribution normal P (parametrized mu_p, sigma_p)

##### Parameters:
 * mu_q: torch.Tensor -> mu parameter of distribution Q
 * sigma_q: torch.Tensor -> sigma parameter of distribution Q
 * mu_p: float -> mu parameter of distribution P
 * sigma_p: float -> sigma parameter of distribution P

##### Returns
torch.Tensor of shape 0

## class BaseMixtureLayer_(torch.nn.Module)
Abstract class which inherits from BaseVariationalLayer_, powered with method to calculate the a KL divergence sample between two mixture of gaussians. 

#### mixture_kl_div(self, mu_m1_d1, sigma_m1_d1, mu_m1_d2, sigma_m1_d2, mu_m2_d1, sigma_m2_d1,mu_m2_d2, sigma_m2_d2, eta, w)
Calculates a sample of KL divergence between two mixture of gaussians (Q || P), given a sample from the first
##### Parameters:
 * mu_m1_d1: torch.Tensor -> mu 1 parameter of distribution Q,
 * sigma_m1_d1 : torch.Tensor -> sigma 1 parameter of distribution Q,
 * mu_m1_d2 : torch.Tensor -> mu 2 parameter of distribution Q,
 * sigma_m1_d2 : torch.Tensor -> sigma 2 parameter of distribution Q,
 * mu_m2_d1: torch.Tensor -> mu 1 parameter of distribution P,
 * sigma_m2_d1: torch.Tensor -> sigma 1 parameter of distribution P,
 * mu_m2_d2: torch.Tensor -> mu 2 parameter of distribution P,
 * sigma_m2_d2: torch.Tensor -> sigma 2 parameter of distribution P,
 * eta: torch.Tensor -> mixture proportions of distribution Q,
 * w: sample from distribution Q,
 
##### Returns
torch.Tensor of shape 0

## class LinearReparameterization
### bayesian_torch.layers.LinearReparameterization(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)

#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
       
#### forward(X)

Samples the weights with reparameterization and performs `torch.nn.functional.linear`. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`

##### Returns:
 * torch.Tensor with shape = `(X.shape[0], out_features)`, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv1dReparameterization
### bayesian_torch.layers.Conv1dReparameterization(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

---

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#conv1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

## class Conv2dReparameterization
### bayesian_torch.layers.Conv2dReparameterization(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior


---

## class Conv3dReparameterization
### bayesian_torch.layers.Conv3dReparameterization(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose1dReparameterization
### bayesian_torch.layers.ConvTranspose1dReparameterization(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose2dReparameterization
### bayesian_torch.layers.ConvTranspose2dReparameterization(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose3dReparameterization
### bayesian_torch.layers.ConvTranspose3dReparameterization(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class LSTMReparameterization
### bayesian_torch.layers.LSTMReparameterization(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)

#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
       
#### forward(X, hidden_states=None)

Samples the weights with reparameterzation and performs LSTM feedforward operation.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`
 * hidden_states: None or tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`) 

##### Returns:
 * tuple: (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`))
    , float corresponding to KL divergence from the samples weights distribution to the prior


---

## class LinearFlipout
### bayesian_torch.layers.LinearFlipout(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
      
#### forward(X)

Samples the weights with flipout reparameterzation and performs `torch.nn.functional.linear`. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`

##### Returns:
 * torch.Tensor with shape = `(X.shape[0], out_features)`, float corresponding to KL divergence from the samples weights distribution to the prior
      

---

## class Conv1dFlipout
### bayesian_torch.layers.Conv1dFlipout(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with flipout reparameterzation and performs `torch.nn.functional.conv1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv2dFlipout
### bayesian_torch.layers.Conv2dFlipout(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
#### forward(X)

Samples the weights with flipout reparameterzation and performs `torch.nn.functional.conv2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv3dFlipout
### bayesian_torch.layers.Conv3dFlipout(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with flipout reparameterzation and performs `torch.nn.functional.conv3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose1dFlipout
### bayesian_torch.layers.ConvTranspose1dFlipout(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
 
#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose2dFlipout
### bayesian_torch.layers.ConvTranspose2dFlipout(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose3dFlipout
### bayesian_torch.layers.ConvTranspose3dFlipout(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
 
#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior


---

## class LSTMFlipout
### bayesian_torch.layers.LSTMFlipout(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)

#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
       
#### forward(X, hidden_states=None)

Samples the weights with Flipout and performs LSTM feedforward operation. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`
 * hidden_states: None or tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`) 

##### Returns:
 * tuple: (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`))
    , float corresponding to KL divergence from the samples weights distribution to the prior

---

## class LinearRadial
### bayesian_torch.layers.LinearRadial(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with radial reparameterzation and performs `torch.nn.functional.linear`. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`

##### Returns:
 * torch.Tensor with shape = `(X.shape[0], out_features)`, float corresponding to KL divergence from the samples weights distribution to the prior


---

## class Conv1dRadial
### bayesian_torch.layers.Conv1dRadial(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with radial reparameterzation and performs `torch.nn.functional.conv1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv2dRadial
### bayesian_torch.layers.Conv2dRadial(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv3dRadial
### bayesian_torch.layers.Conv3dRadial(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with radial reparameterzation and performs `torch.nn.functional.conv3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose1dRadial
### bayesian_torch.layers.ConvTranspose1dRadial(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with radial reparameterzation and performs `torch.nn.functional.conv_transpose1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspsoe1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose2dRadial
### bayesian_torch.layers.ConvTranspose2dRadial(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with radial reparameterzation and performs `torch.nn.functional.conv_transpose2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose3dRadial
### bayesian_torch.layers.ConvTranspose3dRadial(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with reparameterzation and performs `torch.nn.functional.conv_transpose3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class LSTMRadial
### bayesian_torch.layers.LSTMRadial(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)

#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
 * posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus σ = log(1 + exp(ρ)),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
       
#### forward(X, hidden_states=None)

Samples the weights with radial reparameterzation and performs LSTM feedforward operation. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`
 * hidden_states: None or tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`) 

##### Returns:
 * tuple: (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`))
    , float corresponding to KL divergence from the samples weights distribution to the prior

---

## class LinearMixture
### bayesian_torch.layers.LinearMixture(in_features, out_features, prior_mean_1, prior_variance_1, prior_mean_2, prior_variance_2, posterior_mu_init_1, posterior_rho_init_1, posterior_mu_init_2, posterior_rho_init_2, bias=True)
#### Parameters:
   * in_features: int -> size of each input sample,
   * out_features: int -> size of each output sample,
   * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
   * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
   * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
   * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
   * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
   * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
   * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
   * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
   * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

      
#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterzation and performs `torch.nn.functional.linear`. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`

##### Returns:
 * torch.Tensor with shape = `(X.shape[0], out_features)`, float corresponding to KL divergence from the samples weights distribution to the prior
      

---

## class Conv1dMixture
### bayesian_torch.layers.Conv1dMixture(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
 * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
 * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
 * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
 * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
 * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterization and performs `torch.nn.functional.conv1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv2dMixture
### bayesian_torch.layers.Conv2dMixture(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
 * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
 * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
 * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
 * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
 * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterization and performs `torch.nn.functional.conv2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class Conv3dMixture
### bayesian_torch.layers.Conv3dMixture(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
 * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
 * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
 * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
 * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
 * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterization and performs `torch.nn.functional.conv3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#conv3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose1dMixture
### bayesian_torch.layers.ConvTranspose1dMixture(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
 * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
 * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
 * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
 * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
 * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterization and performs `torch.nn.functional.conv_transpose1d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose1d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose2dMixture
### bayesian_torch.layers.ConvTranspose2dMixture(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
 * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
 * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
 * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
 * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
 * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterization and performs `torch.nn.functional.conv_transpose2d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose2d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior

---

## class ConvTranspose3dMixture
### bayesian_torch.layers.ConvTranspose3dMixture(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)
#### Parameters:
 * in_channels: int -> number of channels in the input image,
 * out_channels: int -> number of channels produced by the convolution,
 * kernel_size: int -> size of the convolving kernel,
 * stride: int -> stride of the convolution. Default: 1,
 * padding: int -> zero-padding added to both sides of the input. Default: 0,
 * dilation: int -> spacing between kernel elements. Default: 1,
 * groups: int -> number of blocked connections from input channels to output channels,
 * prior_mean_1: float -> mean of the prior arbitrary distribution 1 to be used on the complexity cost,
 * prior_variance_1: float -> variance of the prior arbitrary distribution 1 to be used on the complexity cost,
 * posterior_mu_init_1: float -> init std for the trainable mu 1 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_1: float -> init std for the trainable rho 1 parameter, sampled from N(0, posterior_rho_init),
 * prior_mean_2: float -> mean of the prior arbitrary distribution 2 to be used on the complexity cost,
 * prior_variance_2: float -> variance of the prior arbitrary distribution 2 to be used on the complexity cost,
 * posterior_mu_init_2: float -> init std for the trainable mu 2 parameter, sampled from N(0, posterior_mu_init),
 * posterior_rho_init_2: float -> init std for the trainable rho 2 parameter, sampled from N(0, posterior_rho_init),
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,

#### forward(X)

Samples the weights with mixture (bimodal gaussian) reparameterization and performs `torch.nn.functional.conv_transpose3d`. Check [PyTorch official documentation](https://pytorch.org/docs/stable/nn.html#convtranspose3d) for tensor output shape.

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, C, H, W, L)`

##### Returns:
 * torch.Tensor, float corresponding to KL divergence from the samples weights distribution to the prior


---

## class LSTMMixture
### bayesian_torch.layers.LSTMMixture(in_features, out_features, prior_mean, prior_variance, posterior_mu_init, posterior_rho_init, bias=True)

#### Parameters:
 * in_features: int -> size of each input sample,
 * out_features: int -> size of each output sample,
 * prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
 * prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
 * posterior_mu_init: float,
 * posterior_rho_init: float,
 * bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
       
#### forward(X, hidden_states=None)

Samples the weights with mixture (gaussian bimodal) reparameterization and performs LSTM feedforward operation. 

##### Parameters:

 * X: torch.Tensor with shape `(batch_size, in_features)`
 * hidden_states: None or tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`) 

##### Returns:
 * tuple: (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, tuple (torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`, torch.Tensor with shape = `(X.shape[0], seq_len, out_features)`))
    , float corresponding to KL divergence from the samples weights distribution to the prior

---
