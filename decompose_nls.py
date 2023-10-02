import numpy as np
import torch
import torch.nn as nn
import collections
from scipy.io import loadmat

def cp_decomposition_conv_layer(layer, rank, i, filename):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """
    
    weights = loadmat(filename)["weights_nls"][0]
    
    last, first, vertical, horizontal = weights[i][0]
    
    last = torch.from_numpy(last.astype(np.float32))
    first = torch.from_numpy(first.astype(np.float32))
    vertical = torch.from_numpy(vertical.astype(np.float32))
    horizontal = torch.from_numpy(horizontal.astype(np.float32))
    
    # U^s
    pointwise_s_to_r_layer = nn.Conv2d(in_channels=first.shape[0],
                                       out_channels=first.shape[1],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)
    
    depthwise_vertical_layer = nn.Conv2d(in_channels=vertical.shape[1], 
                                               out_channels=vertical.shape[1],
                                               kernel_size=(vertical.shape[0], 1),
                                               stride=1,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=vertical.shape[1],
                                               bias=False)

    depthwise_horizontal_layer = nn.Conv2d(in_channels=horizontal.shape[1],
                                                 out_channels=horizontal.shape[1], 
                                                 kernel_size=(1, horizontal.shape[0]),
                                                 stride=layer.stride,
                                                 padding=(0, layer.padding[0]), 
                                                 dilation=layer.dilation,
                                                 groups=horizontal.shape[1],
                                                 bias=False)
    
    pointwise_r_to_t_layer = nn.Conv2d(in_channels=last.shape[1],
                                       out_channels=last.shape[0],
                                       kernel_size=1,
                                       padding=0,
                                       bias=False)

    depthwise_horizontal_layer.weight.data = torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    
    depthwise_vertical_layer.weight.data = torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    
    pointwise_s_to_r_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    
    return nn.Sequential(
    collections.OrderedDict(
            [
                ("0_decomposed", pointwise_s_to_r_layer),
                ("1_decomposed", depthwise_vertical_layer),
                ("2_decomposed", depthwise_horizontal_layer),
                ("3_decomposed", pointwise_r_to_t_layer),
            ]
        )
    )
    