import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import pprint
import torch
from training_utils.helpers import parseargs, fixseed, generate_runname
from models.personalizer import personalization


if __name__ == "__main__":
    args = parseargs()

    # Setup config
    # Load the model class dynamically
    imported_module = __import__(args.model)
    model_name = args.model.split(sep='.')[-1]
    target_model = imported_module.__dict__[model_name].__dict__[model_name]

    # Select the device that will run the experiments, should be a gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch._C.device("cuda:0")

    # Setup paths
    run_name = generate_runname(model_name=target_model.__name__, exp_name=args.expname)
    checkpoint_path = os.path.join("./checkpoints/", args.expname, run_name)
    tensorboard_path = os.path.join("./tensorboard/", args.expname, run_name)
    figure_path = os.path.join("./figs/", args.expname, run_name)
    logs_path = os.path.join("./logs/", args.expname, run_name)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    print(f'Dataset folder: {os.path.join(args.dataset_folder, args.dataset_name)}')
    print(f'Checkpoint folder: {checkpoint_path}')
    print(f'Tensorboard folder: {tensorboard_path}')
    print(f'Figure folder: {figure_path}')
    print(f'Logs folder: {logs_path}')

    # Load configuration into a dict
    config = dict()
    config.update(args.__dict__)  
    config['model_name'] = model_name 
    config['checkpoint_path'] = checkpoint_path
    config['tensorboard_path'] = tensorboard_path
    config['figure_path'] = figure_path
    config['logs_path'] = logs_path

    print('Configuration for the run:')
    pprint.pprint(config, width=1)

    # Seed everything
    fixseed(config['seed'])

    # Personalization
    personalization(tensorboard_path, config, device)
