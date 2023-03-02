# import torch 
# import bayesian_torch
# from bayesian_torch.ao.quantization import prepare, convert
# import bayesian_torch.models.bayesian.resnet_variational_large as resnet
# from bayesian_torch.models.bnn_to_qbnn import bnn_to_qbnn

# model = resnet.__dict__['resnet50']()

# input = torch.randn(1,3,224,224)
# mp = prepare(model)
# mp(input) # haven't replaced the batchnorm layer
# qmodel = torch.quantization.convert(mp)
# bnn_to_qbnn(qmodel)


import torch
import bayesian_torch
import bayesian_torch.models.bayesian.resnet_variational_large as resnet

m = resnet.__dict__['resnet50']()
# alternative way to construct a bnn model
# from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
# m = torchvision.models.resnet50(weights="IMAGENET1K_V1")
# dnn_to_bnn(m)



mp = bayesian_torch.quantization.prepare(m)
input = torch.randn(1,3,224,224)
mp(input) # calibration
mq = bayesian_torch.quantization.convert(mp)



