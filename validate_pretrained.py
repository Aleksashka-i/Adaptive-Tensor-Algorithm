import argparse
import os
import resnet
from resnet import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import time
from tune_validate import *

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='TEST')

parser.add_argument('--testmodel', default='./pretrained_models/resnet32-d509ac18.th', 
                    type=str, metavar='TESTM', help='path to test model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    global args
    args = parser.parse_args()

    model = resnet.__dict__[args.arch]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    
    validate(val_loader, model, criterion, device, args)
    return
        

if __name__ == '__main__':
    test()