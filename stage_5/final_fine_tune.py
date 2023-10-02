import sys
sys.path.append('../') 

import os
import resnet
from resnet import *
import resnet_with_sigmas
from resnet_with_sigmas import *
import torch
import torch.nn as nn
from train_validate import *
from stage_2.fine_tune_initial_decomposition import replace_set_of_layers
from stage_3.extend_decomposed_kernels import extend_set_of_layers
from functional import *
from scipy.io import loadmat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def undecompose(model):
    layers = get_layers(model)
    
    sigmas = []
    for (layer, _) in layers:
        sigmas.append(layer.sigma.item())
    kernels_sz = get_sizes(sigmas)
    
    new_weights = loadmat("./weights/weights_composed.mat")["composed_weights"][0]
    
    i = 0
    for (layer, pos) in layers:
        new_weight = new_weights[i]
        new = nn.Conv2d(new_weight.shape[0],
                        new_weight.shape[1],
                        kernel_size=kernels_sz[i],
                        stride=1,
                        padding=kernels_sz[i] // 2,
                        bias=False)
        new.weight.data = torch.from_numpy(new_weight.astype(np.float32))
        i += 1
        setattr(layer, 'conv2', new)
        
if __name__ == '__main__':
    __spec__ = None
    args = get_args('./decomposed/best_extended_decompose.th')
    
    #load pre-trained model
    model = resnet_with_sigmas.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    replace_set_of_layers(model)
    extend_set_of_layers(model, 21)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])
    
    undecompose(model)
    print(model, flush=True)
    
    val_loader, train_loader, criterion = prepare_train_validate(args, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # prepare loss acc plots
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    layers = get_layers(model)
    sigmas = []
    for (layer, _) in layers:
        sigmas.append(layer.sigma.item())
    kernels_sz = get_sizes(sigmas)
    print("kernels_sz: ", kernels_sz, flush=True)
    
    _, best_prec = validate(val_loader, model, criterion, device, args)
    save_decomposition({
        'state_dict': model.state_dict(),
        'best_prec': best_prec,
        'kernels_sz': kernels_sz,
    }, filename=os.path.join(args.save_dir, 'best_final.th'))
        
    
    # fine-tune
    for epoch in range(0, 50):
        loss_train, prec_train = train(train_loader, model, criterion, optimizer, epoch, device, args)
        
        loss_val, prec_val = validate(val_loader, model, criterion, device, args)
        if prec_val > best_prec:
            best_prec = prec_val
            save_decomposition({
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'kernels_sz': kernels_sz,
            }, filename=os.path.join(args.save_dir, 'best_final.th'))
        
        # plot loss acc
        train_losses.append(loss_train)
        train_acc.append(prec_train)
        test_losses.append(loss_val)
        test_acc.append(prec_val)                      
        plot_losses_acc(train_losses, test_losses, train_acc, test_acc, "./stage_5/final.png")
        print("best precision: ", best_prec, flush=True)
    