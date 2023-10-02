import sys
sys.path.append('../') 

import os
import resnet
from resnet import *
import torch
import torch.nn as nn
from train_validate import *
from functional import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_layers(model, kernels_sz):
    layers = get_layers(model)
    i = 0
    for (layer, pos) in layers:
        new = nn.Conv2d(layer.conv2.in_channels,
                        layer.conv2.out_channels,
                        kernel_size=kernels_sz[i],
                        stride=1,
                        padding=kernels_sz[i] // 2,
                        bias=False)
        i += 1
        setattr(layer, 'conv2', new)

if __name__ == '__main__':
    __spec__ = None
    args = get_args('./decomposed/best_final.th')
    
    param = torch.load(args.testmodel, map_location=device)
    
    model = resnet.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    replace_layers(model, param['kernels_sz'])
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in param['state_dict'].items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    print(model, flush=True)
    
    val_loader, _, criterion = prepare_train_validate(args, device)

    validate(val_loader, model, criterion, device, args)