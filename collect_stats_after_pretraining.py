import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import pprint
import torch
from training_utils.helpers import parseargs, fixseed
from models.compute_feature_stats import compute_feature_stats


if __name__ == "__main__":
    args = parseargs()

    # Setup configuration
    # Load the model class dynamically
    imported_module = __import__(args.model)
    model_name = args.model.split(sep='.')[-1]
    target_model = imported_module.__dict__[model_name].__dict__[model_name]

    # Select the device that will run the experiments, should be a gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch._C.device("cuda:0")
    torch.set_default_dtype(torch.float32)
    
    config = dict()    
    config.update(args.__dict__)  
    config['model_name'] = model_name 
    print('Configuration for the run:')
    pprint.pprint(config, width=1)
    
    # Seed everything
    fixseed(config['seed'])
    
    # Compute Feature Stats
    compute_feature_stats(device, config)