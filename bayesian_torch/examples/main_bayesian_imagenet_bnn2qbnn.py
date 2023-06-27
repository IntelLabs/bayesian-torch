import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import bayesian_torch
import bayesian_torch.models.bayesian.resnet_variational_large as resnet
import numpy as np
from bayesian_torch.models.bnn_to_qbnn import bnn_to_qbnn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
# import bayesian_torch.models.bayesian.quantized_resnet_variational_large as qresnet
import bayesian_torch.models.bayesian.quantized_resnet_flipout_large as qresnet

torch.cuda.is_available = lambda : False
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.backends.quantized.engine='onednn'
model_names = sorted(
    name
    for name in resnet.__dict__
    if name.islower() and not name.startswith("__") and name.startswith("resnet") and callable(resnet.__dict__[name])
)

print(model_names)
best_acc1 = 0
len_trainset = 1281167
len_valset = 50000


parser = argparse.ArgumentParser(description="ImageNet")
parser.add_argument('data',
                    metavar='DIR',
                    default='data/imagenet',
                    help='path to dataset')
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)"
)
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=1000, type=int, metavar="N", help="mini-batch size (default: 512)")
parser.add_argument('--val_batch_size', default=1000, type=int)
parser.add_argument("--lr", "--learning-rate", default=0.001, type=float, metavar="LR", help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay", "--wd", default=1e-4, type=float, metavar="W", help="weight decay (default: 5e-4)"
)
parser.add_argument("--print-freq", "-p", default=50, type=int, metavar="N", help="print frequency (default: 20)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--half", dest="half", action="store_true", help="use half-precision(16-bit) ")
parser.add_argument(
    "--save-dir",
    dest="save_dir",
    help="The directory used to save the trained models",
    default="../../bayesian-torch-20221214/bayesian_torch/checkpoint/bayesian",
    type=str,
)
parser.add_argument(
    "--moped-init-model",
    dest="moped_init_model",
    help="DNN model to intialize MOPED method",
    default="",
    type=str,
)
parser.add_argument(
    "--moped-delta-factor",
    dest="moped_delta_factor",
    help="MOPED delta scale factor",
    default=0.2,
    type=float,
)

parser.add_argument(
    "--bnn-rho-init",
    dest="bnn_rho_init",
    help="rho init for bnn layers",
    default=-3.0,
    type=float,
)

parser.add_argument(
    "--use-flipout-layers",
    type=bool,
    default=False,
    metavar="use_flipout_layers",
    help="Use Flipout layers for BNNs, default is Reparameterization layers",
)

parser.add_argument(
    "--save-every",
    dest="save_every",
    help="Saves checkpoints at every specified number of epochs",
    type=int,
    default=10,
)
parser.add_argument("--mode", type=str, required=True, help="train | test")

parser.add_argument(
    "--num_monte_carlo",
    type=int,
    default=20,
    metavar="N",
    help="number of Monte Carlo samples to be drawn during inference",
)
parser.add_argument("--num_mc", type=int, default=1, metavar="N", help="number of Monte Carlo runs during training")
parser.add_argument(
    "--tensorboard",
    type=bool,
    default=True,
    metavar="N",
    help="use tensorboard for logging and visualization of training progress",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="./logs/cifar/bayesian",
    metavar="N",
    help="use tensorboard for logging and visualization of training progress",
)

def evaluate(args, model, val_loader, calibration=False):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        begin = time.time()
        i=0
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(args.num_monte_carlo):
                output = model.forward(data)
                output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_list.append(output_)
            labels_list.append(target)
            i+=1
            end = time.time()
            print("inference throughput: ", i*args.val_batch_size / (end - begin), " images/s")
            # break
            if calibration and i==3:
                break

        output = torch.cat(output_list, 1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)
        pred_mean = output.mean(dim=0)
        Y_pred = torch.argmax(pred_mean, axis=1)
        print("Test accuracy:", (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() * 100)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    moped_enable = False
    if len(args.moped_init_model) > 0:  # use moped method if trained dnn model weights are provided
        moped_enable = True

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if moped_enable:
        checkpoint = torch.load(args.moped_init_model)
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    tb_writer = None

    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.val_batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

    print('len valset: ', len(val_dataset))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.mode == "test":
        const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": args.bnn_rho_init,
        "type": "Flipout" if args.use_flipout_layers else "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": moped_enable,  # initialize mu/sigma from the dnn weights
        "moped_delta": args.moped_delta_factor,
        }
        quantizable_model = torchvision.models.quantization.resnet50()
        dnn_to_bnn(quantizable_model, const_bnn_prior_parameters)
        model = torch.nn.DataParallel(quantizable_model)
    
        
        checkpoint_file = args.save_dir + "/bayesian_{}_imagenet.pth".format(args.arch)

        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        model.module = model.module.cpu()

        mp = bayesian_torch.quantization.prepare(model)
        evaluate(args, mp, val_loader, calibration=True) # calibration
        qmodel = bayesian_torch.quantization.convert(mp)
        evaluate(args, qmodel, val_loader)

        # save weights
        save_checkpoint(
                    {
                        'epoch': None,
                        'state_dict': qmodel.state_dict(),
                        'best_prec1': None,
                    },
                    True,
                    filename=os.path.join(
                        args.save_dir,
                        'quantized_bayesian_{}_imagenetv2.pth'.format(args.arch)))

        # reconstruct (no calibration)
        quantizable_model = torchvision.models.quantization.resnet50()
        dnn_to_bnn(quantizable_model, const_bnn_prior_parameters)
        model = torch.nn.DataParallel(quantizable_model)
        mp = bayesian_torch.quantization.prepare(model)
        qmodel1 = bayesian_torch.quantization.convert(mp)

        # load
        checkpoint_file = args.save_dir + "/quantized_bayesian_{}_imagenetv2.pth".format(args.arch)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        qmodel1.load_state_dict(checkpoint["state_dict"])
        evaluate(args, qmodel1, val_loader)


        return mp, qmodel, qmodel1

if __name__ == "__main__":
    mp, qmodel, qmodel1 = main()
