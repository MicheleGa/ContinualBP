import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import argparse
import pprint
from distutils.util import strtobool
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.dataset import PhysioDataset
from models.trainer import pretraining_training_validation_testing
from training_utils.helpers import fixseed, generate_runname
from training_utils.metrics import call_metric


def pretraining(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device):
    r"""
    Pretrains a neural network model for Blood Pressure Estimation from PPG and ECG data using PyTorch Lightning.

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


def parseargs():
    parser = argparse.ArgumentParser(description="Multimodal Blood Pressure from PPG/ECG/RESP with NN - Pretraining")
    
    # Run setup
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--dataset_folder', default='./data/lmdb', type=str, help='path to dataset fodler')
    parser.add_argument('--dataset_name', default='test', type=str, help='name of the dataset')
    parser.add_argument('--expname', default='test', type=str, help='experiment name')
    parser.add_argument('--model', default="models.ResGRUNet", type=str, help='model name')
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # Generic training setup
    parser.add_argument('--max_training_epochs', default=75, type=int, help='maximum epochs')
    parser.add_argument('--eval_every_n_epochs', default=1, type=int, help='evaluate every n epochs')
    parser.add_argument('--es_patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--es_min_delta', default=0.01, type=float, help='early stopping minimum delta')
    parser.add_argument('--es_enable', default='False', type=lambda x: bool(strtobool(x)), help='enable early stopping')
    parser.add_argument('--enable_amp', default='False', type=lambda x: bool(strtobool(x)), help='enable automatic mixed precision')
    
    # Optimization setup
    parser.add_argument('--smoothl1loss_beta', default=5, type=int, help='beta for SmoothL1Loss')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_linear_probe', default=0.0005, type=float, help='learning rate for personalization')
    parser.add_argument('--lr_personalization', default=0.0005, type=float, help='learning rate for personalization')
    parser.add_argument('--l2norm', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--sgd_momentum', default=0.9, type=float, help='Momentum for SGD optimizer')
    parser.add_argument('--lrsched_step', default="5, 10, 15, 20, 40", type=str, help='learning rate scheduler steps')
    parser.add_argument('--lrsched_gamma', default=0.5, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--criterion', default="MSELoss", type=str, help='loss criterion')
    parser.add_argument('--lambda_supervised', default=0., type=float, help='scale supervised loss function contribution, set 0. to prevent its application')
    parser.add_argument('--lambda_ortho', default=0., type=float, help='induce feature orthogonality during pretraining, set 0. to prevent its application')
    parser.add_argument('--lambda_contrastive', default=0., type=float, help='induce contrastive loss contribution to the final loss during pretraining, set 0. to prevent its application')
    parser.add_argument('--temperature', default=0., type=float, help='temperature scalar for SimCLR loss')
    parser.add_argument('--optimizer_type', default="AdamW", type=str, help='optimizer type')
    parser.add_argument('--lr_scheduler_type', default="MultiStepLR", type=str, help='learning rate scheduler type')
    parser.add_argument('--lr_scheduler_enable', default='False', type=lambda x: bool(strtobool(x)), help='enable learning rate scheduler')
    parser.add_argument('--lr_scheduler_min_lr', default=0.0001, type=float, help='enable learning rate scheduler')
    parser.add_argument('--lr_scheduler_warmup', default=0, type=int, help='enable learning rate scheduler')
    
    # Dataset setup
    parser.add_argument('--pretraining_ratio', default=0.8, type=float, help='pretraining ratio of the whole dataset')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.1,0.2', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--personalization_sample_number', default=100, type=int, help='number of samples to take for personalization')
    parser.add_argument('--mix_pretraining_subject_samples', default='True', type=lambda x: bool(strtobool(x)), help='whether to mix pretraining subject samples among train/val/test or not')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg imfs or not')
    parser.add_argument('--ppg_freqs', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg freqs or not')
    parser.add_argument('--aug', default='False', type=lambda x: bool(strtobool(x)), help='whether to use data augmentations during pretraining or not')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    
    # Generic model setup
    parser.add_argument('--set_tunable_params', default='all', type=str, help='which model parameters to tune (all, only regressor, only encoder, etc.)')
    parser.add_argument('--return_embedding', default='True', type=lambda x: bool(strtobool(x)), help='whether to return the model embedding before the regressor or not')
    parser.add_argument('--proj_head_dim', default=256, type=int, help='dimension of the projection head after the feture extractor') 
    
    # ResGRUNet setup
    #parser.add_argument('--gru', default='True', type=lambda x: bool(strtobool(x)), help='whether to use a GRU after CNN or not')
    #parser.add_argument('--channels', default='1, 32, 64, 128', type=str, help='channels produced by the convolutional blocks')
    #parser.add_argument('--act', default='leaky_relu', type=str, help='which activation to use (ReLU or LeakyReLU)')
    #parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    #parser.add_argument('--pooling', default='avg', type=str, help='which poolng to use (average or max)')
    #parser.add_argument('--supervised', default='False', type=lambda x: bool(strtobool(x)), help='whether to use the model with pretrianing or not')
        
    ## PhysioFormer setup
    #parser.add_argument('--embed_dim', default=64, type=int, help='input embedding size')
    #parser.add_argument('--hidden_dim', default=256, type=int, help='transformer hidden dimension')
    #parser.add_argument('--num_layers', default=3, type=int, help='number of transformer layers')
    #parser.add_argument('--n_head', default=4, type=int, help='number of self-attention heads')
    #parser.add_argument('--head_dim', default=16, type=int, help='self-attention headd dimension')
    
    # UNet setup
    parser.add_argument('--channels', default='32, 64, 128, 256, 512', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--num_heads_attention', default=1, type=int, help='heads number of the final self-attention layer') 
    parser.add_argument('--dim_feedforward_attention', default=128, type=int, help='dimension of the final self-attention layer') 
    parser.add_argument('--kernel_size', default=3, type=int, help='convolutional layer kernel size')
    
    # GRU setup
    parser.add_argument('--hidden_dim', default=128, type=int, help='GRU hidden dimension size')
    parser.add_argument('--num_layers', default=2, type=int, help='number of GRU layers')
    parser.add_argument('--bidirectional', default=True, type=lambda x: bool(strtobool(x)), help='whether to use bidirectional GRU or not')
        
    # Transformer setup
    parser.add_argument('--embed_dim', default=32, type=int, help='transformer embedding dimension size')
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads for the self-attention mechanism')
    parser.add_argument('--num_encoder_layers', default=2, type=int, help='number of trasnformer layers')
    parser.add_argument('--dim_feedforward', default=128, type=int, help='feedforward dimension size in the transformer encoder') 
        
    args = parser.parse_args()
    return args


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

    print(f'Dataset folder: {args.dataset_folder}')
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
    
    # PPG derivatives/PPG EMD/PPG freqs are loaded only if ecg (and optionally resp) are not present
    if (config['ppg_derivatives'] or config['ppg_emd'] or config['ppg_freqs']) and config['ecg']: 
        raise ValueError('PPG derivatives/emd/scalogram can be loaded only without ECG (and optionally RESP)')  
    
    dataset = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_ratio=config['pretraining_ratio'],
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        personalization_sample_number=config['personalization_sample_number'],
        mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        resp=config['resp'],
        sig2sig=config['sig2sig'],
        ppg_derivatives=config['ppg_derivatives'],
        ppg_emd=config['ppg_emd'],
        ppg_freqs=config['ppg_freqs']
    )
    
    ## Pretraining
    pretraining(args.expname, model_name, dataset, checkpoint_path, tensorboard_path, config, device)