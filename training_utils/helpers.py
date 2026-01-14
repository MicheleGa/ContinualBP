import os
import sys
folders_to_add = ['models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import argparse
import time
import yaml
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from models import BIOT, TCN, Proto, ResGruNet
from models.component_factory import BPRegressor


def parseargs():
    parser = argparse.ArgumentParser(description="Multimodal Blood Pressure from PPG/ECG/RESP with NN - Pretraining")
    
    # Run Setup
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--expname', default='test', type=str, help='experiment name')
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # Optimization Setup
    parser.add_argument('--criterion', default="MSELoss", type=str, help='loss criterion')
    parser.add_argument('--smoothl1loss_beta', default=5, type=int, help='beta for SmoothL1Loss')
    parser.add_argument('--sgd_momentum', default=0.9, type=float, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    
    # Dataset Setup
    parser.add_argument('--dataset_folder', default='./data/lmdb', type=str, help='path to dataset fodler')
    parser.add_argument('--dataset_name', default='test', type=str, help='name of the dataset')
    parser.add_argument('--index_file_name', default='pulse_db_index.csv', type=str, help='name of the dataset index file')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.15,0.15', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--meta_train_split_ratio', default=0.2, type=float, help='percentage of subjects to extract from the training subjects for meta-learning')
    parser.add_argument('--min_subject_sample_number', default=0, type=int, help='minimum number of samples per subject to consider it valid, 0 means no limit; given the support and query sample number, it is set to 15')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='whether to load only ecg or not')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    parser.add_argument('--output_dim', default=3, type=int, help='output dimension (SBP/MAP/DBP))')
    
    # Pre-training Setup
    parser.add_argument('--pretrained_encoder_ckpt_path', default='', type=str, help='optional checkpoint path for the encoder')
    parser.add_argument('--stage1_epochs', default=10, type=int, help='epochs training head only')
    parser.add_argument('--stage1_freeze_epochs', default=30, type=int, help='epochs training head only')
    parser.add_argument('--stage1_pre_train_lr', default=1e-3, type=float, help='ssl learning rate for pre-training stage 1')
    parser.add_argument('--stage1_pre_train_scheduler_eta_min', default=1e-5, type=float, help='min learning rate for stage 1 pre-training stage 1')
    
    # Meta-learning Setup
    parser.add_argument('--max_meta_epochs', default=150, type=int, help='maximum number of meta epochs')
    parser.add_argument('--k_support', type=int, default=5, help='number of support samples in meta-learning')
    parser.add_argument('--k_query', type=int, default=10, help='number of query samples in meta-learning')
    parser.add_argument('--meta_batch_size', type=int, default=8, help='meta batch size')
    parser.add_argument('--msl_anneal_epochs', default=20, type=int, help='how many epochs until full MSL anneal (later inner steps get more weight)')   
    parser.add_argument('--msl_pre_base_weight', default=0.5, type=float, help='base weight for pre-adaptation loss if included')
    
    parser.add_argument('--meta_lr', default=1e-3, type=float, help='meta-learning learning rate')
    parser.add_argument('--meta_lr_schedule', default='cosine', type=str, choices=['constant', 'cosine', 'cosine_wr', 'multistep'], help='meta-learning learning rate schedule type')
    parser.add_argument('--meta_lr_steps', default=[100, 200, 300, 400], type=int, nargs='+', help='meta-learning learning rate steps for step decay')
    parser.add_argument('--meta_lr_gamma', default=0.1, type=float, help='meta-learning learning rate gamma for step decay')   
    parser.add_argument('--meta_lr_scheduler_T0', default=100, type=int, help='cosine wr scheduler T0 parmeter')   
    parser.add_argument('--meta_lr_scheduler_T_mult', default=1.5, type=float, help='cosine wr scheduler T mult parmeter')   
    parser.add_argument('--meta_lr_scheduler_eta_min', default=0.00001, type=float, help='cosine wr scheduler eta min parmeter')
    parser.add_argument('--meta_lr_scheduler_gamma', default=0.00001, type=float, help='cosine wr scheduler eta min parmeter')
    parser.add_argument('--meta_lr_scheduler_min_gamma', default=0.00001, type=float, help='cosine wr scheduler eta min gamma')
    parser.add_argument('--meta_lr_scheduler_max_cycles', default=3, type=int, help='cosine wr number of cycless')
    
    parser.add_argument('--inner_adapt', default='head', type=str, help='ANIL: adapt only head parameters in inner loop')
    parser.add_argument('--inner_opt', default='adam', type=str, help='inner loop optimizer type')
    parser.add_argument('--inner_lr', default=1e-2, type=float, help='inner learning rate for meta-learning')
    parser.add_argument('--inner_lr_min', default=1e-3, type=float, help='minimum inner learning rate (increased for ANIL stability)')
    parser.add_argument('--inner_lr_schedule', default='constant', type=str, choices=['constant', 'cosine'], help='inner learning rate schedule type (simplified for ANIL)')
    parser.add_argument('--inner_steps', default=5, type=int, help='number of inner steps for meta-learning (increased for ANIL)')
    parser.add_argument('--inner_steps_max', default=8, type=int, help='maximum inner steps for cosine schedule')
    parser.add_argument('--inner_steps_schedule', default='constant', type=str, choices=['constant', 'cosine'], help='inner steps schedule type (simplified for ANIL)')
    
    parser.add_argument('--eval_lr', default=5e-3, type=float, help='inner learning rate for meta-learning evaluation')
    parser.add_argument('--eval_steps', default=8, type=int, help='inner steps for meta-learning evaluation')
    
    # LoRA setup
    parser.add_argument('--use_lora', action=argparse.BooleanOptionalAction, default=False, help='enable Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning')
    parser.add_argument('--lora_r', default=8, type=int, help='LoRA rank (dimensionality of low-rank matrices A and B)')
    parser.add_argument('--lora_alpha', default=16, type=float, help='LoRA scaling parameter (alpha), controls the magnitude of LoRA updates')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='dropout probability applied to LoRA layers during training')
    
    # Model Setup
    parser.add_argument('--model', default="models.ResGRUNet", type=str, help='model name')
    parser.add_argument('--channels', default='1, 64, 128, 256', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    parser.add_argument('--act', default='leaky_relu', type=str, help='which activation to use (ReLU or LeakyReLU)')
    parser.add_argument('--pooling', default='avg', type=str, help='which poolng to use (average or max)')
    parser.add_argument('--embed_dim', default=256, type=int, help='embedding dimension')
    parser.add_argument('--num_groups', default=8, type=int, help='number of groups for group normalization')
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads for the self-attention mechanism')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='number of trasnformer layers')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='number of decoder layers')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability')
    parser.add_argument('--n_fft', default=200, type=int, help='number of elements for STFT')
    parser.add_argument('--hop_length', default=100, type=int, help='hop length for STFT')

    # Personalization Setup
    parser.add_argument('--pretrained_model_ckpt_path', default=None, type=str, help='checkpoint path to the pretrained model')
    parser.add_argument('--pretraining_feats_stats', default=None, type=str, help='path to the features statistics during pretraining')
    parser.add_argument('--num_personalization_subjects', default=0, type=int, help='number of subjects to for personalization')
    parser.add_argument('--personalization_steps', default=8, type=int, help='number of gradient steps for personalization')
    parser.add_argument('--personalization_lr', default=1e-2, type=float, help='learning rate for personalization')
    parser.add_argument('--plot_personalization', action=argparse.BooleanOptionalAction, default=False, help='whether to plot the subject annotation over the total windows or not')
    parser.add_argument('--personalization_batch_size', default=16, type=int, help='batch size for personalization')
    parser.add_argument('--validation_batch_size', default=16, type=int, help='batch size for personalization')
    parser.add_argument('--num_train_val', default=2, type=int, help='number of training and validation phases per block')
    parser.add_argument('--valid_runs_number', default=2, type=int, help='valid runs number')
    parser.add_argument('--setup_type', default='drift', type=str, choices=['drift', 'fixed'], help='whether to trigger adaptation after the distribution shift detector or not')
    parser.add_argument('--drift_threshold', default=0.5, type=float, help='threshold for drift detector')
    parser.add_argument('--replay_threshold', default=0.35, type=float, help='replay buffer threshold for drift detector')
    parser.add_argument('--replay_buffer_size', default=256, type=int, help='maximum size of the feature replay buffer')
    parser.add_argument('--replay_batch_size', default=16, type=int, help='batch size for feature replay')
    
    args = parser.parse_args()
    return args


def fixseed(SEED):
    r"""
    Fixes the random seed for reproducibility across runs.
    
    Parameters
    ------------    
        SEED (int): 
            The seed value to be set for random number generation.  
    
    Returns
    ------------
        None: 
            The function sets the random seed for various libraries to ensure reproducibility.
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_runname(model_name, exp_name):
    r"""
    Composes a unique run name based on the model name, experiment name, and current timestamp.
    
    Parameters
    ------------
        model_name (str): 
            Name of the model.
        exp_name (str): 
            Name of the experiment.
    Returns
    ------------
        str: 
            A unique run name formatted as "exp_name-model_name-YYYY_MM_DD-HH_MM_SS".    
    """
    
    exec_timestamp = time.localtime()
    exec_timestr = "{:4d}_{:02d}_{:02d}-{:02d}_{:02d}_{:02d}".format(
        exec_timestamp.tm_year, 
        exec_timestamp.tm_mon, 
        exec_timestamp.tm_mday, 
        exec_timestamp.tm_hour, 
        exec_timestamp.tm_min, 
        exec_timestamp.tm_sec)
    run_name = "{}-{}-{}".format(exp_name, model_name, exec_timestr)
    return run_name


def count_parameters(model):
    r"""
    Count parameters trainable and non trainable parameters of a model.
    
    Parameters
    ------------    
        model (torch.nn.Module): 
            The pytroch model with the .parameters() method.  
    
    Returns
    ------------
        trainable, non_trainable (tuple):
            A tuple containing the number of trainable and non-trainable parameters in the model.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable

    
def save_status(subject_id, epoch, model_name, save_name, model, optimizer, scheduler, meter, checkpoint_path, config):
    r"""
    Save model checkpoint, hyperparameters, and training configurations.
    
    Parameters
    ------------
        subject_id (int or None): 
            Subject ID for which the checkpoint is saved. If None, saves the global checkpoint.
        epoch (int): 
            Current epoch number.
        model_name (str): 
            Name of the model checkpoint file.
        save_name (str): 
            Name of the experiment or save directory.
        model (torch.nn.Module): 
            Model to save the state dictionary from.
        optimizer (torch.optim.Optimizer): 
            Optimizer to save the state dictionary from.
        scheduler (torch.optim.lr_scheduler or None): 
            Learning rate scheduler to save the state dictionary from.
        meter (AverageMeter): 
            Validation loss meter.
        checkpoint_path (str): 
            Path where checkpoints are saved.
        config (dict): 
            Configuration dictionary containing model parameters and settings.
            
    Returns
    ------------
        None: 
            The function saves the model, optimizer, scheduler state dictionaries, and validation metrics to a file.
    """
    to_save = dict()
    to_save['epoch'] = epoch
    to_save['model'] = model.state_dict()
    if optimizer is not None:
        to_save['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        to_save['lr_scheduler'] = scheduler.state_dict()
    if isinstance(meter, np.float32):
        to_save['val_loss'] = meter
    else:
        to_save['val_loss'] = meter.avg

    if subject_id is not None:
        out_path = os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt') 
    else:
        out_path = os.path.join(checkpoint_path, save_name, 'ckpt')
    
    os.makedirs(out_path, exist_ok=True)
    torch.save(to_save, os.path.join(out_path, model_name))
    with open(os.path.join(out_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=True, default_flow_style=False)
    
    print(f"Epoch {epoch + 1} - checkpoint {model_name} saved in {out_path}")
        
    
def load_status(subject_id, model_name, save_name, model, optimizer, scheduler, checkpoint_path, config):
    r"""
    Load model checkpoint.
    
    Parameters
    ------------
        subject_id (int or None): 
            Subject ID for which the checkpoint is loaded. If None, loads the global checkpoint.
        model_name (str): 
            Name of the model checkpoint file.
        save_name (str): 
            Name of the experiment or save directory.
        model (torch.nn.Module): 
            Model to load the state dictionary into.
        optimizer (torch.optim.Optimizer or None): 
            Optimizer to load the state dictionary into.
        scheduler (torch.optim.lr_scheduler or None): 
            Learning rate scheduler to load the state dictionary into.
        checkpoint_path (str): 
            Path where checkpoints are saved.
        config (dict): 
            Configuration dictionary containing model parameters and settings.
            
    Returns
    ------------
        None: 
            The function modifies the model, optimizer, and scheduler in place.    
    """
    assert model is not None, "Model is required for loading status"
    
    if subject_id is not None:
        checkpoint = torch.load(os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt', model_name), weights_only=False, map_location='cpu')
        print(f"Checkpoint loaded from {os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt', model_name)}", map_location='cpu')
    else:
        checkpoint = torch.load(os.path.join(checkpoint_path, save_name, 'ckpt', model_name), weights_only=False)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_path, save_name, 'ckpt', model_name)}")
        
    epoch = int(checkpoint['epoch'])
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    print(f"Loaded checkpoint from validation on epoch {epoch + 1}: {checkpoint['val_loss']}")

                
def get_encoder_architecture(config):
    r"""
    Function to get the model encoder architecture based on the configuration.
    
    Parameters
    ------------
        config (dict): 
            Configuration dictionary containing model parameters.
    
    Returns
    ------------
        encoder (torch.nn.Module): 
            The model encoder architecture initialized with the given configuration.   
    """
    encoder = None
    
    if config['model_name'] == 'ResGruNet':   
        encoder = ResGruNet.ResGruNet(
            ecg=config['ecg'], 
            fs=config['fs'], 
            input_seq_len_s=config['input_seq_len_s'],
            embed_dim=config['embed_dim']
        )
    elif config['model_name'] == 'TCN':
        input_size = 2 if config['ecg'] else 1
        input_length = config['input_seq_len_s'] * config['fs']
        output_size = config['embed_dim']
        channel_sizes = [input_size, 16, 32, 48, 64, 96, output_size]
        kernel_size = [7] * 7                           
        
        encoder = TCN.TCN(
            input_size=input_size,
            output_size=output_size,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            input_length=input_length
        )
    elif config['model_name'] == 'BIOT':
        encoder = BIOT.BIOT(
            ecg=config['ecg'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            embed_dim=config['embed_dim'],
            pretrained_path=config['pretrained_encoder_ckpt_path']
        )
    elif config['model_name'] == 'Proto':
        encoder = Proto.Proto(
            ecg=config['ecg'], 
            fs=config['fs'], 
            input_seq_len_s=config['input_seq_len_s'],
            embed_dim=config['embed_dim']
        )
    else:
        raise ValueError("Invalid model name ...")
    
    if encoder is None:
        raise ValueError("Error occurred during model arhcitecture intialization ...")
    
    return encoder


def get_prediction_head_architecture(config):
    r"""
    Function to get the model prediction head architecture based on the configuration.
    
    Parameters
    ------------
        config (dict): 
            Configuration dictionary containing model parameters.
    
    Returns
    ------------
        model (torch.nn.Module): 
            The model prediction head architecture initialized with the given configuration.   
    """
    
    return BPRegressor(input_dim=2 * config['embed_dim'] if config['ecg'] else config['embed_dim'], output_dim=config['output_dim'])

    
def get_meta_lr(epoch, config):
    r"""
    Get meta-learning learning rate based on the specified schedule.
    
    Parameters
    ------------
        epoch (int): 
            Current epoch number.
        config (dict): 
            Configuration dictionary containing meta-learning parameters.
    
    Returns
    ------------
        float: 
            The computed meta-learning learning rate for the given epoch.    
    """
    
    meta_lr_schedule = config['meta_lr_schedule']
    meta_epochs = config['max_meta_epochs']
    base_meta_lr = config['meta_lr']

    if meta_lr_schedule == 'constant':
        return base_meta_lr
    elif meta_lr_schedule == 'cosine':
        eta_min = float(config.get('meta_lr_scheduler_eta_min'))
        return eta_min + (base_meta_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
    elif meta_lr_schedule == 'cosine_wr':
        # Extended version with decaying peaks
        T0 = int(config.get('meta_lr_scheduler_T0'))
        T_mult = float(config.get('meta_lr_scheduler_T_mult'))
        eta_min0 = float(config.get('meta_lr_scheduler_eta_min'))
        gamma = float(config.get('meta_lr_scheduler_gamma'))
        min_gamma = float(config.get('meta_lr_scheduler_min_gamma'))
        max_cycles = config.get('meta_lr_scheduler_max_cycles')
        
        cycle = 0
        length = T0
        e = epoch
        while e >= length:
            e -= length
            cycle += 1
            length = int(length * T_mult)

        if (max_cycles is not None) and (cycle >= int(max_cycles)):
            # After max cycles, stay at final eta_min
            final_eta_min = eta_min0 * (min_gamma ** (int(max_cycles) - 1))
            return final_eta_min
        
        # Calculate current cycle parameters
        eta_max_cycle = base_meta_lr * (gamma ** cycle)
        eta_min_cycle = eta_min0 * (min_gamma ** cycle)
        
        # Ensure cycle ends exactly at eta_min by using (length-1) for full cycle
        # When e = length-1, cos(π) = -1, giving eta_min exactly
        cycle_progress = math.pi * e / max(1, length - 1)
        cosine_factor = 0.5 * (1 + math.cos(cycle_progress))
        
        return eta_min_cycle + (eta_max_cycle - eta_min_cycle) * cosine_factor
    elif meta_lr_schedule == 'multistep':
        # MultiStep LR: piecewise decay at specified milestones
        milestones = config.get("meta_lr_steps")  # epochs where decay happens
        gamma = float(config.get("meta_lr_gamma")) # decay factor
        lr = base_meta_lr
        for m in milestones:
            if epoch >= m:
                lr *= gamma
        return lr

    else:
        return base_meta_lr


def get_inner_lr(epoch, config):
    r"""
    Get inner learning rate based on the specified schedule.
    
    Parameters
    ------------
        epoch (int): 
            Current epoch number.
        config (dict): 
            Configuration dictionary containing inner learning parameters.
    
    Returns
    ------------
        float: 
            The computed inner learning rate for the given epoch.
    """
    
    schedule = config['inner_lr_schedule']
    meta_epochs = config['max_meta_epochs']
    base_inner_lr = config['inner_lr']
    inner_lr_min = config['inner_lr_min']
    
    if schedule == 'constant':
        return base_inner_lr
    elif schedule == 'cosine':
        return inner_lr_min + (base_inner_lr - inner_lr_min) * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
    else:
        return base_inner_lr


def get_inner_steps(epoch, config):
    r"""
    Get number of inner steps based on the specified schedule.
    
    Parameters
    ------------
        epoch (int): 
            Current epoch number.
        config (dict): 
            Configuration dictionary containing inner steps parameters. 
    
    Returns
    ------------
        int: 
            The computed number of inner steps for the given epoch.
    """
    
    meta_epochs = config['max_meta_epochs']
    schedule = config['inner_steps_schedule']
    base_steps = config['inner_steps']
    max_steps = config['inner_steps_max']
    
    if schedule == 'constant':
        return base_steps
    elif schedule == 'cosine':
        progress = epoch / meta_epochs
        return int(round(base_steps + 0.5 * (1 - math.cos(math.pi * progress)) * (max_steps - base_steps)))
    else:
        return base_steps

    
def build_inner_optimizer(adapted_encoder, adapted_head, base_lr, config):
    r"""
    Build inner optimizer for meta-learning algorithms evaluation and deployment (not training, learn2learn library with autograd is employed for training).
    Behaviors:
    - inner_adapt='all'  : adapt backbone + regressor
    - inner_adapt='head' : freeze backbone, adapt only regressor and LoRA params if applicable.
    
    Parameters
    ------------
        adapted_encoder (torch.nn.Module): 
            The model encoder to be adapted.
        adapted_head (torch.nn.Module): 
            The model prediction head to be adapted.
        base_lr (float): 
            The base learning rate for the inner optimizer.
        config (dict): 
            Configuration dictionary containing inner adaptation parameters.
            
    Returns
    ------------
        inner_opt (torch.optim.Optimizer): 
            The constructed inner optimizer for meta-learning evaluation and deployment.    
    """
    mode = config.get('inner_adapt')   
    opt_type = config.get('inner_opt').lower()  
    
    params = []
    
    # Default to 'head' - ANIL
    if mode == 'all':
        # Mode all not supported for LoRA
        if config['use_lora']:
            raise ValueError("inner_adapt='all' not supported for LoRA models")
        
        # Train both encoder and head parameters
        params = list(adapted_encoder.parameters()) + list(adapted_head.parameters())
    else:    
        # Do not train backbone parameters in ANIL - only head + LoRA params if applicable
        params = list(adapted_head.parameters())
        if config['use_lora']:
            for name, param in adapted_encoder.named_parameters():
                if 'lora_' in name:
                    params.append(param)
    
    # Build optimizer, default to Adam
    if opt_type == 'sgd':
        momentum = float(config.get('sgd_momentum'))
        inner_opt = torch.optim.SGD(params, lr=base_lr, momentum=momentum)
    else:
        inner_opt = torch.optim.Adam(params, lr=base_lr)
    
    return inner_opt
    

def get_msl_weights(epoch, config, num_inner_steps):
    r"""
    Returns a list of weights (length num_inner_steps), normalized to sum to 1. 
    The weights are annealed over epochs so later inner steps get more weight as training proceeds.
    
    Parameters
    ------------    
        epoch (int): 
            Current epoch number.
        config (dict): 
            Configuration dictionary containing MSL parameters.
        num_inner_steps (int): 
            Number of inner steps in the meta-learning algorithm.

    Returns
    ------------
        weights (list of float): 
            A list of weights for each inner step, normalized to sum to 1.
    """
    # anneal control: 0 -> no anneal (uniform), 1 -> full anneal (linear bias to later steps)
    anneal_epochs = config.get('msl_anneal_epochs')
    anneal_factor = float(min(epoch, anneal_epochs)) / max(1.0, anneal_epochs)

    # baseline: uniform on the post-update steps
    steps_idx = list(range(1, num_inner_steps + 1))  # 1..S
    # Raw weight for step s = 1 + anneal_factor * s  (so later steps get higher weight)
    raw = [1.0 + anneal_factor * float(s) for s in steps_idx]

    # Normalize
    total = sum(raw)
    weights = [r / total for r in raw]
    return weights


def compute_embedding_stats(encoder, dataloader, device):
    r"""
    Compute mean and std of embeddings over a dataloader.
    
    Parameters
    ------------    
        encoder (torch.nn.Module): 
            The model encoder to compute embeddings.
        dataloader (torch.utils.data.DataLoader): 
            DataLoader providing the data to compute embeddings.
        device (torch.device): 
            Device to perform computations on.

    Returns
    ------------
        mean (np.ndarray): 
            Mean of the embeddings.
        std (np.ndarray): 
            Standard deviation of the embeddings.
    """
    encoder.eval()
    all_embs = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (Xs, _), (_, _), _ = batch
            meta_batch = Xs.shape[0]
            
            for t in range(meta_batch):
                sX = Xs[t].to(device).float()
                z = encoder(sX)
                all_embs.append(z.detach().cpu().numpy())
            
    all_embs = np.concatenate(all_embs, axis=0)
    mean = np.mean(all_embs, axis=0)
    std = np.std(all_embs, axis=0)
    std[std == 0] = np.finfo(np.float32).eps
    return mean, std
