import sys
sys.path.append('../') 

import resnet
from resnet import *
import torch
import torch.nn as nn
from train_validate import *
from stage_2.fine_tune_initial_decomposition import replace_set_of_layers
from functional import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    __spec__ = None
    args = get_args('./decomposed/best_initial_decompose.th')
    
    #load pre-trained model
    model = resnet.__dict__["resnet32"]()
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device)
    replace_set_of_layers(model)
    param = torch.load(args.testmodel, map_location=device)
    model.load_state_dict(param['state_dict'])
    
    print(model, flush=True)
    
    val_loader, _, criterion = prepare_train_validate(args, device)

    validate(val_loader, model, criterion, device, args)