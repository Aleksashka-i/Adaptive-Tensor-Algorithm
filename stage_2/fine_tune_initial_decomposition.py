import sys
sys.path.append('../') 

import resnet
from resnet import *
import torch
import torch.nn as nn
from decompose_nls import cp_decomposition_conv_layer
from functional import *
from train_validate import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_set_of_layers(model):
    layers = get_layers(model)
    for (layer, pos) in layers:
        module = layer.conv2
        rank = module.in_channels // 2
        new = cp_decomposition_conv_layer(module, rank, pos, './weights/weights_nls.mat')
        setattr(layer, 'conv2', new)

if __name__ == '__main__':
    __spec__ = None
    args = get_args('./pretrained_models/resnet32-d509ac18.th')
    
    #load pre-trained model
    model = resnet.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])
    
    # replace every conv2 layer
    replace_set_of_layers(model)
    print(model, flush=True)
    
    val_loader, train_loader, criterion = prepare_train_validate(args, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    for param in model.parameters():
        param.requires_grad = True
    
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    _, best_prec = validate(val_loader, model, criterion, device, args)
    save_decomposition({
        'state_dict': model.state_dict(),
        'best_prec': best_prec,
    }, filename=os.path.join(args.save_dir, 'best_initial_decompose.th'))
    
    # fine-tune
    for epoch in range(0, 50):
        loss_train, prec_train = train(train_loader, model, criterion, optimizer, epoch, device, args)
        
        loss_val, prec_val = validate(val_loader, model, criterion, device, args)
        if prec_val > best_prec:
            best_prec = prec_val
            save_decomposition({
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
            }, filename=os.path.join(args.save_dir, 'best_initial_decompose.th'))
        
        train_losses.append(loss_train)
        train_acc.append(prec_train)
        test_losses.append(loss_val)
        test_acc.append(prec_val)
                              
        plot_losses_acc(train_losses, test_losses, train_acc, test_acc, "./stage_2/initial_decomposition.png")
        print("best precision: ", best_prec, flush=True)