import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt
import argparse

def get_args(default_filename):
    parser = argparse.ArgumentParser(description='args')

    parser.add_argument('--testmodel', default=default_filename, 
                    type=str, metavar='TESTM', help='path to test model')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the decomposed models',
                        default='decomposed', type=str)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    return parser.parse_args()

def prepare_train_validate(args, device):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.half:
        model.half()
        criterion.half()
    
    return val_loader, train_loader, criterion
    

def save_decomposition(state, filename):
    torch.save(state, filename)

def get_layers(model):
    layers = [
        (model.module.layer1[1], 1),
        (model.module.layer1[3], 3),
        (model.module.layer2[1], 6),
        (model.module.layer2[3], 8),
        (model.module.layer3[1], 11),
        (model.module.layer3[3], 13),
    ]
    return layers

def get_factor_torch(sigma):
    i = torch.arange(21).float()
    s = 2 * torch.pi / 2 + 2 * torch.arctan(sigma)
    data = ((1 / (torch.sqrt(2 * torch.pi * (s ** 2)))) ** 2) * torch.exp(-(i - 10) ** 2 / (2 * (s ** 2)))
    return data

def get_factor_np(sigma):
    array = np.arange(21.0)
    s = 2 * np.pi / 2 + 2 * np.arctan(sigma)
    data = ((1 / (np.sqrt(2 * np.pi * (s ** 2)))) ** 2)  *  np.exp(-(array - 10) ** 2 / (2 * (s ** 2)))
    return data

def get_sizes(sigmas):
    kernels_sz = []
    for sigma in sigmas:
        data = get_factor_np(sigma)
        sz = np.sum(data >= 0.001)
        kernels_sz.append(sz)
    return kernels_sz

def plot_losses_acc(train_losses, test_losses, train_accuracies, test_accuracies, filename):
    _, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.grid()
        ax.set_xlabel('epoch')
        ax.legend()

    plt.savefig(filename)
    plt.close()

def plot_sigmas(sigmas, filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, values in sigmas.items():
        plt.plot(values, label=name)
    
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Change in Sigmas Values During Training", fontsize=16)
    
    plt.savefig(filename)
    plt.close()