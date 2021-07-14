from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy
from scipy.special import softmax
import bayesian_torch.models.bayesian.simple_cnn_variational as simple_cnn

len_trainset = 60000
len_testset = 10000


def train(args, model, device, train_loader, optimizer, epoch, tb_writer=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output, kl = model(data)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        nll_loss = F.nll_loss(output, target)
        #ELBO loss
        loss = nll_loss + (kl / args.batch_size)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.flush()


def test(args, model, device, test_loader, epoch, tb_writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, kl = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() + (
                kl / args.batch_size)  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    val_accuracy = correct / len(test_loader.dataset)
    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', test_loss, epoch)
        tb_writer.add_scalar('val/accuracy', val_accuracy, epoch)
        tb_writer.flush()


def evaluate(args, model, device, test_loader):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_probs_mc = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for mc_run in range(args.num_monte_carlo):
                model.eval()
                output, _ = model.forward(data)
                #get probabilities from log-prob
                pred_probs = torch.exp(output)
                pred_probs_mc.append(pred_probs.cpu().data.numpy())

        target_labels = target.cpu().data.numpy()
        pred_mean = np.mean(pred_probs_mc, axis=0)
        Y_pred = np.argmax(pred_mean, axis=1)
        print('Test accuracy:', (Y_pred == target_labels).mean() * 100)
        np.save('./probs_mnist_mc.npy', pred_probs_mc)
        np.save('./mnist_test_labels_mc.npy', target_labels)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=10000,
                        metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./checkpoint/bayesian')
    parser.add_argument('--mode', type=str, required=True, help='train | test')
    parser.add_argument(
        '--num_monte_carlo',
        type=int,
        default=20,
        metavar='N',
        help='number of Monte Carlo samples to be drawn for inference')
    parser.add_argument('--num_mc',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of Monte Carlo runs during training')
    parser.add_argument(
        '--tensorboard',
        action="store_true",
        help=
        'use tensorboard for logging and visualization of training progress')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/mnist/bayesian',
        metavar='N',
        help=
        'use tensorboard for logging and visualization of training progress')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    tb_writer = None
    if args.tensorboard:

        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        print("yee")
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

        tb_writer = SummaryWriter(logger_dir)
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              **kwargs)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = simple_cnn.SCNN()
    model = model.to(device)

    print(args.mode)
    if args.mode == 'train':

        for epoch in range(1, args.epochs + 1):
            if (epoch < args.epochs / 2):
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            else:
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr / 10)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            train(args, model, device, train_loader, optimizer, epoch,
                  tb_writer)
            test(args, model, device, test_loader, epoch, tb_writer)
            scheduler.step()

            torch.save(model.state_dict(),
                       args.save_dir + "/mnist_bayesian_scnn.pth")

    elif args.mode == 'test':
        checkpoint = args.save_dir + '/mnist_bayesian_scnn.pth'
        model.load_state_dict(torch.load(checkpoint))
        evaluate(args, model, device, test_loader)


if __name__ == '__main__':
    main()
