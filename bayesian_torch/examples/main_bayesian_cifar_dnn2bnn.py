import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import bayesian_torch.models.deterministic.resnet as resnet
import numpy as np
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

from bayesian_torch.ao.quantization.quantize import enable_prepare, convert
from bayesian_torch.models.bnn_to_qbnn import bnn_to_qbnn

model_names = sorted(
    name
    for name in resnet.__dict__
    if name.islower() and not name.startswith("__") and name.startswith("resnet") and callable(resnet.__dict__[name])
)

print(model_names)
len_trainset = 50000
len_testset = 10000

parser = argparse.ArgumentParser(description="CIFAR10")
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet20",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
)
parser.add_argument(
    "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 8)"
)
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size (default: 512)")
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
    default="./checkpoint/bayesian",
    type=str,
)
parser.add_argument(
    "--model-checkpoint",
    dest="model_checkpoint",
    help="Saved checkpoint for evaluating model",
    default="",
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
parser.add_argument("--mode", type=str, required=True, help="train | test | ptq")

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

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    moped_enable = False
    if len(args.moped_init_model) > 0:  # use moped method if trained dnn model weights are provided
        moped_enable = True

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": args.bnn_rho_init,
        "type": "Flipout" if args.use_flipout_layers else "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": moped_enable,  # initialize mu/sigma from the dnn weights
        "moped_delta": args.moped_delta_factor,
    }

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda() if torch.cuda.is_available() else model.cpu()
    if moped_enable:
        checkpoint = torch.load(args.moped_init_model)
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    dnn_to_bnn(model, const_bnn_prior_parameters)  # only replaces linear and conv layers
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, "tb_logger")
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    calib_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(random.sample(range(1, 50000), 100)),
        num_workers=args.workers,
        pin_memory=True,
    )


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    if args.half:
        model.half()
        criterion.half()

    if args.arch in ["resnet110"]:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr * 0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.mode == "train":

        for epoch in range(args.start_epoch, args.epochs):

            lr = args.lr
            if epoch >= 80 and epoch < 120:
                lr = 0.1 * args.lr
            elif epoch >= 120 and epoch < 160:
                lr = 0.01 * args.lr
            elif epoch >= 160 and epoch < 180:
                lr = 0.001 * args.lr
            elif epoch >= 180:
                lr = 0.0005 * args.lr

            optimizer = torch.optim.Adam(model.parameters(), lr)

            # train for one epoch
            print("current lr {:.5e}".format(optimizer.param_groups[0]["lr"]))
            train(args, train_loader, model, criterion, optimizer, epoch, tb_writer)

            prec1 = validate(args, val_loader, model, criterion, epoch, tb_writer)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_prec1": best_prec1,
                    },
                    is_best,
                    filename=os.path.join(args.save_dir, "bayesian_{}_cifar.pth".format(args.arch)),
                )

    elif args.mode == "test":
        checkpoint_file = args.save_dir + "/bayesian_{}_cifar.pth".format(args.arch)
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        evaluate(args, model, val_loader)

    elif args.mode == "ptq":
        if len(args.model_checkpoint) > 0:
           checkpoint_file = args.model_checkpoint
        else:
           print("please provide valid model-checkpoint")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

        '''
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        print('load checkpoint...')
        '''
        model.load_state_dict(checkpoint['state_dict'])

        # post-training quantization
        model_int8 = quantize(model, calib_loader, args)
        model_int8.eval()
        model_int8.cpu()

        print('Evaluating quantized INT8 model....')
        evaluate(args, model_int8, val_loader)

        #for i, (data, target) in enumerate(calib_loader):
        #    data = data.cpu()

        #with torch.no_grad():
        #    traced_model = torch.jit.trace(model_int8, data)
        #    traced_model = torch.jit.freeze(traced_model)

        #save_path = os.path.join(
        #                   args.save_dir,
        #                   'quantized_bayesian_{}_cifar.pth'.format(args.arch))
        #traced_model.save(save_path)
        #print('INT8 model checkpoint saved at ', save_path)
        #print('Evaluating quantized INT8 model....')
        #evaluate(args, traced_model, val_loader)

    '''
    elif args.mode =='test_ptq':
        print('load model...')
        if len(args.model_checkpoint) > 0:
           checkpoint_file = args.model_checkpoint
        else:
           print("please provide valid quantized model checkpoint")
        model_int8 = torch.jit.load(checkpoint_file)
        model_int8.eval()
        model_int8.cpu()
        model_int8 = torch.jit.freeze(model_int8)
        print('Evaluating the INT8 model....')
        evaluate(args, model_int8, val_loader)
     '''

def train(args, train_loader, model, criterion, optimizer, epoch, tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
        else:
            target = target.cpu()
            input_var = input.cpu()
            target_var = target

        if args.half:
            input_var = input_var.half()

        # compute output
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output = model(input_var)
            kl = get_kl_loss(model)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        cross_entropy_loss = criterion(output, target_var)
        scaled_kl = kl / args.batch_size

        # ELBO loss
        loss = cross_entropy_loss + scaled_kl

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1
                )
            )

        if tb_writer is not None:
            tb_writer.add_scalar("train/cross_entropy_loss", cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar("train/kl_div", scaled_kl.item(), epoch)
            tb_writer.add_scalar("train/elbo_loss", loss.item(), epoch)
            tb_writer.add_scalar("train/accuracy", prec1.item(), epoch)
            tb_writer.flush()


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            if args.half:
                input_var = input_var.half()

            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output = model(input_var)
                kl = get_kl_loss(model)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            cross_entropy_loss = criterion(output, target_var)
            # scaled_kl = kl / len_trainset
            scaled_kl = kl / args.batch_size
            # scaled_kl = 0.2 * (kl / len_trainset)

            # ELBO loss
            loss = cross_entropy_loss + scaled_kl

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                    )
                )

            if tb_writer is not None:
                tb_writer.add_scalar("val/cross_entropy_loss", cross_entropy_loss.item(), epoch)
                tb_writer.add_scalar("val/kl_div", scaled_kl.item(), epoch)
                tb_writer.add_scalar("val/elbo_loss", loss.item(), epoch)
                tb_writer.add_scalar("val/accuracy", prec1.item(), epoch)
                tb_writer.flush()

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def evaluate(args, model, val_loader):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        begin = time.time()
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
        end = time.time()
        print("inference throughput: ", len_testset / (end - begin), " images/s")

        output = torch.stack(output_list)
        output = output.permute(1, 0, 2, 3)
        output = output.contiguous().view(args.num_monte_carlo, len_testset, -1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)
        pred_mean = output.mean(dim=0)
        Y_pred = torch.argmax(pred_mean, axis=1)
        print("Test accuracy:", (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() * 100)
        np.save("./probs_cifar_mc.npy", output.data.cpu().numpy())
        np.save("./cifar_test_labels_mc.npy", labels.data.cpu().numpy())


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Save the training model
    """
    torch.save(state, filename)

def quantize(model, calib_loader, args, **kwargs):
    model.eval()
    model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig("onednn")
    print('Preparing model for quantization....')
    enable_prepare(model)
    prepared_model = torch.quantization.prepare(model)
    print('Calibrating...')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(calib_loader):
            data = data.cpu()
            _ = prepared_model(data)
    print('Calibration complete....')
    quantized_model = convert(prepared_model)
    return quantized_model

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


if __name__ == "__main__":
    main()
