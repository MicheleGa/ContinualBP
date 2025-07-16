import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import pprint
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.dataset import PhysioDataset
from data.dataset_ssl import PhysioDatasetSSL
from models.trainer import pretraining_training_validation_testing
from training_utils.helpers import fixseed, generate_runname, parseargs
from training_utils.metrics import call_metric


def pretraining(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device):
    r"""
    Pretrains a neural network model for Blood Pressure Estimation from PPG and ECG data with supervision or self-supervision.

    Parameters
    ------------
    
    save_name (str): 
        Name of the experiment/run, used for saving checkpoints and logs.
    model_name (str): 
        Name of the model name to run.
    dataset: 
        Pytorch  DataLoader to split the pretraining dataset into 'train', 'val', and 'test'.
    checkpoint_path (str): 
        Path to the directory where checkpoints should be saved.
    tensorboard_path (str): 
        Path to the directory where TensorBoard logs should be saved.
    config (dict): 
        Configuration dictionary containing hyperparameters and training settings
    """
    
    # Get train/val/test samplers and build the dataloaders
    (train_sampler, val_sampler, test_sampler) = dataset.get_pretraining_samplers()
    
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)    
    valid_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    
    all_targets, all_outputs = pretraining_training_validation_testing(
        save_name=save_name,
        checkpoint_path=checkpoint_path,
        tensorboard_path=tensorboard_path,
        model_name=model_name,
        dataloaders={
            'train': train_dataloader,
            'val': valid_dataloader,
            'test': test_dataloader
        },
        config=config,
        device=device
    )
    
    # Log test metrics (optionally, plot them) and return loss for validation
    _ = call_metric(all_targets, all_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'pretraining'), plot=True)    
    
    # Save the best model and configuration after training
    print(f"Pretraining completed, best model and configuration saved in {checkpoint_path}")


if __name__ == "__main__":
    global args
    args = parseargs()

    ## Setup configuration
    # Load the model class dynamically
    imported_module = __import__(args.model)
    model_name = args.model.split(sep='.')[-1]
    target_model = imported_module.__dict__[model_name].__dict__[model_name]

    # Select the gpu to be usesd
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch._C.device("cuda:0")

    # Setup paths
    run_name = generate_runname(model_name=target_model.__name__, exp_name=args.expname)
    checkpoint_path = os.path.join("./checkpoints/", args.expname, run_name)
    tensorboard_path = os.path.join("./tensorboard/", args.expname, run_name)
    figure_path = os.path.join("./figs/", args.expname, run_name)
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    print(f'Dataset folder: {os.path.join(args.dataset_folder, args.dataset_name)}')
    print(f'Checkpoint folder: {checkpoint_path}')
    print(f'Tensorboard folder: {tensorboard_path}')
    print(f'Figure folder: {tensorboard_path}')

    # Load configuration into a dict
    config = dict()    
    config.update(args.__dict__)  # add argparse
    config['model_name'] = model_name # add model name
    config['checkpoint_path'] = checkpoint_path
    config['tensorboard_path'] = tensorboard_path
    config['figure_path'] = figure_path
    
    print('Configuration for the run:')
    pprint.pprint(config, width=1)
    
    # Seed everything
    fixseed(config['seed'])
    pl.seed_everything(config['seed'])
    
    ## Build dataset
    
    # RESP is loaded only if ECG is also loaded
    if config['resp'] and not config['ecg']:
        raise ValueError('RESP can be loaded only along with ECG')
    
    if not config['ssl']:
        # Load data and annotation
        dataset = PhysioDataset(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
            pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
            mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig']
        )
    else:
        # Load data
        dataset = PhysioDatasetSSL(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
            pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
            mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            masking_ratio=config['masking_ratio'],
            augmentation_types=config['augmentation_types'].split(','),
            aug_prob=config['aug_prob']
        )
        
    ## Pretraining
    pretraining(args.expname, model_name, dataset, checkpoint_path, tensorboard_path, config, device)