import sys
sys.path.append('../') 

import os
import resnet
from resnet import *
import resnet_with_sigmas
from resnet_with_sigmas import *
import torch
import torch.nn as nn
from decompose_nls import cp_decomposition_conv_layer
from functional import *
from train_validate import *
from stage_2.fine_tune_initial_decomposition import replace_set_of_layers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extend_set_of_layers(model, new_sz):
    layers = get_layers(model)
    for (layer, pos) in layers:
        # vertical
        module = layer.conv2[1]
        base_weight = module.weight.data
        base_shape = module.weight.data.shape
        
        new_shape = list(base_shape)
        new_shape[2] = new_sz
        new_weight = torch.zeros(torch.Size(new_shape))
        padding = new_sz // 2
        new_weight[:, :, (padding - 1):(padding + 2), :] = base_weight
        
        new = nn.Conv2d(module.in_channels,
                        module.out_channels,
                        kernel_size=(new_sz, 1),
                        stride=module.stride,
                        padding=(padding, 0), 
                        dilation=module.dilation,
                        groups=module.groups,
                        bias=module.bias)
        
        new.weight.data = new_weight
        setattr(layer.conv2, '1_decomposed', new)
        
        # horizontal
        module = layer.conv2[2]
        base_weight = module.weight.data
        base_shape = module.weight.data.shape
        
        new_shape = list(base_shape)
        new_shape[3] = new_sz
        new_weight = torch.zeros(torch.Size(new_shape))
        padding = new_sz // 2
        new_weight[:, :, :, (padding - 1):(padding + 2)] = base_weight
        
        new = nn.Conv2d(module.in_channels,
                        module.out_channels,
                        kernel_size=(1, new_sz),
                        stride=module.stride,
                        padding=(0, padding), 
                        dilation=module.dilation,
                        groups=module.groups,
                        bias=module.bias)
        
        new.weight.data = new_weight
        setattr(layer.conv2, '2_decomposed', new)
        
if __name__ == '__main__':
    __spec__ = None

    args = get_args('./decomposed/best_initial_decompose.th')
    
    # upload decomposed weights + add sigma to every layer
    param = torch.load(args.testmodel, map_location=device)
    
    model = resnet_with_sigmas.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    replace_set_of_layers(model)
    
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in param['state_dict'].items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    extend_set_of_layers(model, 21) # 21x21
    # model after layer extension
    print(model, flush=True)
    
    val_loader, train_loader, criterion = prepare_train_validate(args, device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    for param in model.parameters():
        param.requires_grad = True
    
    # prepare sigma plots
    layers = get_layers(model)
    sigmas_plots = {}
    i = 0
    for (layer, _) in layers:
        sigmas_plots[i] = [layer.sigma.item()]
        i += 1
    
    # prepare loss acc plots
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    
    _, best_prec = validate(val_loader, model, criterion, device, args)
    save_decomposition({
        'state_dict': model.state_dict(),
        'best_prec': best_prec,
    }, filename=os.path.join(args.save_dir, 'best_extended_decompose.th'))
    
    # fine-tune
    for epoch in range(0, 50):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss_train, prec_train = train(train_loader, model, criterion, optimizer, epoch, device, args)
        lr_scheduler.step()
        
        loss_val, prec_val = validate(val_loader, model, criterion, device, args)
        if prec_val > best_prec:
            best_prec = prec_val
            save_decomposition({
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
            }, filename=os.path.join(args.save_dir, 'best_extended_decompose.th'))
        
        # plot loss acc
        train_losses.append(loss_train)
        train_acc.append(prec_train)
        test_losses.append(loss_val)
        test_acc.append(prec_val)                      
        plot_losses_acc(train_losses, test_losses, train_acc, test_acc, 
                        "./stage_3/extended_kernels_decomposition.png")
        
        # plot sigmas
        sigmas = []
        i = 0
        for (layer, _) in layers:
            sigmas_plots[i].append(layer.sigma.item())
            i += 1
            sigmas.append(layer.sigma.item())
            
        kernels_sz = get_sizes(sigmas)
       
        plot_sigmas(sigmas_plots, "./stage_3/sigmas_values.png")
        
        print("sigmas: ", sigmas, flush = True)
        print("kernels_sz: ", kernels_sz, flush = True)
        print("best precision: ", best_prec, flush=True)
    