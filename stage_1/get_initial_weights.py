import sys
sys.path.append('../') 

import resnet
from resnet import *
import torch
import torch.nn as nn
from scipy.io import savemat
from functional import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_list_shape(lst):
    shape = []
    
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    
    return shape

weights_base = []

def get_weights(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            get_weights(module)
        if n == "conv2":
            array = module.weight.data.numpy().tolist()
            global weights_base
            weights_base.append(array)

if __name__ == '__main__':
    __spec__ = None
    __spec__ = None
    args = get_args('./pretrained_models/resnet32-d509ac18.th')
    
    param = torch.load(args.testmodel, map_location=device)
    
    model = resnet.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])

    file_name = './weights/weights_base.mat'
    savemat(file_name, {'weights_base': weights_base})