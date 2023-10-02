import sys
sys.path.append('../') 

import resnet
from resnet import *
import resnet_with_sigmas
from resnet_with_sigmas import *
import torch
import torch.nn as nn
from stage_2.fine_tune_initial_decomposition import replace_set_of_layers
from stage_3.extend_decomposed_kernels import extend_set_of_layers
from functional import *
from scipy.io import savemat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    __spec__ = None
    args = get_args('./decomposed/best_extended_decompose.th')
    
    param = torch.load(args.testmodel, map_location=device)
    
    model = resnet_with_sigmas.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    replace_set_of_layers(model)
    extend_set_of_layers(model, 21)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])
    
    layers = get_layers(model)
    
    sigmas = []
    for (layer, _) in layers:
        sigmas.append(layer.sigma.item())
    kernels_sz = get_sizes(sigmas)
    
    print("kernels_sz: ", kernels_sz, flush=True)
    print("sigmas: ", sigmas, flush=True)
    
    weights_decomposed = []
    
    for (module, _), sz in zip(layers, kernels_sz):
        first = module.conv2[0].weight.data
        
        factor = get_factor_torch(module.sigma)
        
        vertical = module.conv2[1].weight.data
        factor_x = factor.view(1, 1, 21, 1)
        vertical = vertical * factor_x
        vertical_crop = vertical[:, :, (10 - sz // 2):(10 + sz // 2 + 1), :]
        
        horizontal = module.conv2[2].weight.data
        factor_y = factor.view(1, 1, 1, 21)
        horizontal = horizontal * factor_y
        horizontal_crop = horizontal[:, :, :, (10 - sz // 2):(10 + sz // 2 + 1)]
        
        last = module.conv2[3].weight.data

        first = torch.transpose(first.squeeze(-1).squeeze(-1), 0, 1).detach().cpu().numpy().tolist()
        vertical = torch.transpose(vertical_crop.squeeze(1).squeeze(-1), 0, 1).detach().cpu().numpy().tolist()
        horizontal = torch.transpose(horizontal_crop.squeeze(1).squeeze(1), 0, 1).detach().cpu().numpy().tolist()
        last = last.squeeze(-1).squeeze(-1).detach().cpu().numpy().tolist()
        
        weights_decomposed.append([last, first, vertical, horizontal])
    
    savemat('./weights/weights_base.mat', {'weights_decomposed': weights_decomposed})