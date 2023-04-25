import argparse
import os
import resnet
from resnet import *
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import time
from decompose import cp_decomposition_conv_layer
from tune_validate import train, validate, AverageMeter, accuracy
from prettytable import PrettyTable

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
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the decomposed models',
                    default='decomposed', type=str)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.00000001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_decomposition(state, filename):
    torch.save(state, filename)

def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            if (replace_layers(module)):
                return True
            
        if "decomposed" not in n:
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                if (max(module.weight.data.numpy().shape) == 64):
                    if n == "conv2":
                        rank = max(module.weight.data.numpy().shape)//3 # rank estimation how?
                        new = cp_decomposition_conv_layer(module, rank)
                        setattr(model, n, new)
                        return True
    return False

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':
    #load pre-trained model
    global args
    args = parser.parse_args()
    
    model = resnet.__dict__[args.arch]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    
    model.eval()
    model.to(device)
    count_parameters(model)
    
    #replace one layer
    while (replace_layers(model)):
        print(model, flush=True)
        
        for param in model.parameters():
            param.requires_grad = True
        
        # validate after replacement
        validate(val_loader, model, criterion, device, args)
        
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
        
        # fine-tune
        for epoch in range(0, 1):
            train(train_loader, model, criterion, optimizer, epoch, device, args)
        
        model.eval()
        count_parameters(model)
        
        # validate after fine-tuning
        validate(val_loader, model, criterion, device, args)
        
        # to replace just one layer
        break
    
    # saving decomposed model
    save_decomposition(model.state_dict(), filename=os.path.join(args.save_dir, 'decomposed_model.th'))
    
    