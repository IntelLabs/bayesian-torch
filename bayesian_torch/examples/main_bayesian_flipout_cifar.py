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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.bayesian.resnet_flipout as resnet
import numpy as np

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

print(model_names)
len_trainset = 50000
len_testset = 10000
num_classes = 10

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 20)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half',
                    dest='half',
                    action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/bayesian',
                    type=str)
parser.add_argument(
    '--save-every',
    dest='save_every',
    help='Saves checkpoints at every specified number of epochs',
    type=int,
    default=10)
parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument(
    '--num_monte_carlo',
    type=int,
    default=20,
    metavar='N',
    help='number of Monte Carlo samples to be drawn during inference')
parser.add_argument('--num_mc',
                    type=int,
                    default=1,
                    metavar='N',
                    help='number of Monte Carlo runs during training')
parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/cifar/bayesian',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    if args.half:
        model.half()
        criterion.half()
    '''
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
    if args.arch in ['resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1
    '''

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.mode == 'train':
        for epoch in range(args.start_epoch, args.epochs):

            lr = args.lr
            if (epoch >= 80 and epoch < 120):
                lr = 0.1 * args.lr
            elif (epoch >= 120 and epoch < 160):
                lr = 0.01 * args.lr
            elif (epoch >= 160 and epoch < 180):
                lr = 0.001 * args.lr
            elif (epoch >= 180):
                lr = 0.0005 * args.lr

            optimizer = torch.optim.Adam(model.parameters(), lr)

            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(args, train_loader, model, criterion, optimizer, epoch,
                  tb_writer)
            #lr_scheduler.step()

            prec1 = validate(args, val_loader, model, criterion, epoch,
                             tb_writer)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if is_best:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best,
                    filename=os.path.join(
                        args.save_dir,
                        'bayesian_flipout_{}_cifar.pth'.format(args.arch)))

    elif args.mode == 'test':
        checkpoint_file = args.save_dir + '/bayesian_flipout_{}_cifar.pth'.format(
            args.arch)
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file,
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        evaluate(args, model, val_loader)


def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

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

        input_var = torch.cat([input_var for _ in range(args.num_mc)], 0)

        output_mc = []
        kl_mc = []
        output, kl = model(input_var)
        output_mc.append(output)
        kl_mc.append(kl)
        output_ = torch.stack(output_mc)
        output_mean = output_.reshape(args.num_mc, -1, num_classes).mean(dim=0)
        kl = torch.stack(kl_mc)
        cross_entropy_loss = criterion(output_mean, target_var)
        scaled_kl = kl / args.batch_size 
        #ELBO loss
        loss = cross_entropy_loss + scaled_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_mean.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', prec1.item(), epoch)
            tb_writer.flush()


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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

            input_var = torch.cat([input_var for _ in range(args.num_mc)], 0)

            output_mc = []
            kl_mc = []
            output, kl = model(input_var)
            output_mc.append(output)
            kl_mc.append(kl)
            output_ = torch.stack(output_mc)
            output_mean = output_.reshape(args.num_mc, -1,
                                          num_classes).mean(dim=0)
            kl = torch.stack(kl_mc)
            cross_entropy_loss = criterion(output_mean, target_var)
            scaled_kl = kl / args.batch_size 
            #ELBO loss
            loss = cross_entropy_loss + scaled_kl

            output = output_mean.float()
            loss = loss.float()

            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1))

            if tb_writer is not None:
                tb_writer.add_scalar('val/cross_entropy_loss',
                                     cross_entropy_loss.item(), epoch)
                tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
                tb_writer.add_scalar('val/elbo_loss', loss.item(), epoch)
                tb_writer.add_scalar('val/accuracy', prec1.item(), epoch)
                tb_writer.flush()

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

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

            data = torch.cat([data for _ in range(args.num_monte_carlo)], 0)
            output_mc = []
            output, _ = model.forward(data)
            output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_ = output_.reshape(args.num_monte_carlo, -1, num_classes)
            output_list.append(output_)
            labels_list.append(target)
        end = time.time()
        print("inference throughput: ", len_testset / (end - begin),
              " images/s")

        output = torch.stack(output_list)
        output = output.permute(1, 0, 2, 3)
        output = output.contiguous().view(args.num_monte_carlo, len_testset,
                                          -1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)
        pred_mean = output.mean(dim=0)
        Y_pred = torch.argmax(pred_mean, axis=1)
        print('Test accuracy:',
              (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() *
              100)
        np.save('./probs_cifar_mc_flipout.npy', output.data.cpu().numpy())
        np.save('./cifar_test_labels_mc_flipout.npy',
                labels.data.cpu().numpy())


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
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


def accuracy(output, target, topk=(1, )):
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


if __name__ == '__main__':
    main()
