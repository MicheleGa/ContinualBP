import os
import sys
folders_to_add = ['models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import argparse
from distutils.util import strtobool
import time
import yaml
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from models import ResGruNet, BIOT


def parseargs():
    parser = argparse.ArgumentParser(description="Multimodal Blood Pressure from PPG/ECG/RESP with NN - Pretraining")
    
    # Run Setup
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--expname', default='test', type=str, help='experiment name')
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # Generic training Setup
    parser.add_argument('--max_training_epochs', default=75, type=int, help='maximum epochs')
    parser.add_argument('--max_lp_training_epochs', default=75, type=int, help='maximum linear probing epochs')
    parser.add_argument('--eval_every_n_epochs', default=1, type=int, help='evaluate every n epochs')
    parser.add_argument('--es_patience', default=10, type=int, help='early stopping patience')
    parser.add_argument('--es_min_delta', default=0.01, type=float, help='early stopping minimum delta')
    parser.add_argument('--es_enable', default='False', type=lambda x: bool(strtobool(x)), help='enable early stopping')
    parser.add_argument('--enable_amp', default='False', type=lambda x: bool(strtobool(x)), help='enable automatic mixed precision')
    
    # Optimization Setup
    parser.add_argument('--criterion', default="MSELoss", type=str, help='loss criterion')
    parser.add_argument('--smoothl1loss_beta', default=5, type=int, help='beta for SmoothL1Loss')
    parser.add_argument('--sgd_momentum', default=0.9, type=float, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--grad_clip', default=10.0, type=float, help='gradient clipping')
    
    # Dataset Setup
    parser.add_argument('--dataset_folder', default='./data/lmdb', type=str, help='path to dataset fodler')
    parser.add_argument('--pulse_db', default='False', type=lambda x: bool(strtobool(x)), help='whether to load the MIMIC III/VitalDB from the PulseDB or not')
    parser.add_argument('--dataset_name', default='test', type=str, help='name of the dataset')
    parser.add_argument('--index_file_name', default='pulse_db_index.csv', type=str, help='name of the dataset index file')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.1,0.2', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--mix_pretraining_subject_samples', default='True', type=lambda x: bool(strtobool(x)), help='whether to mix pretraining subject samples among train/val/test or not')
    parser.add_argument('--min_subject_sample_number', default=0, type=int, help='minimum number of samples per subject to consider it valid, 0 means no limit; given the support and query sample number, it is set to 15')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    
    # Personalization Setup
    parser.add_argument('--pretrained_model_ckpt_path', default=None, type=str, help='checkpoint path to the pretrained model')
    parser.add_argument('--pretraining_feats_stats', default=None, type=str, help='path to the features statistics during pretraining')
    parser.add_argument('--min_run_length', default=128, type=float, help='number of samples to take for trainng and testing personalization must be greater than training_samples')
    parser.add_argument('--num_personalization_subjects', default=100, type=int, help='number of subjects to for personalization')
    parser.add_argument('--personalization_steps', default=8, type=int, help='number of gradient steps for personalization')
    parser.add_argument('--personalization_lr', default=1e-2, type=float, help='learning rate for personalization')
    parser.add_argument('--plot_personalization', default='False', type=lambda x: bool(strtobool(x)), help='whether to plot the subject annotation over the total windows or not')
    parser.add_argument('--personalization_batch_size', default=16, type=int, help='batch size for personalization')
    parser.add_argument('--setup_type', default='drift', type=str, choices=['drift', 'fixed'], help='whether to trigger adaptation after the distribution shift detector or not')
    parser.add_argument('--replay_buffer_size', default=256, type=int, help='maximum size of the feature replay buffer')
    parser.add_argument('--replay_batch_size', default=16, type=int, help='batch size for feature replay')
    
    # Pre-training Setup
    parser.add_argument('--pretrained_encoder_ckpt_path', default=None, type=str, help='optional checkpoint path for the encoder')
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
    parser.add_argument('--msl_include_pre', default=False, type=lambda x: bool(strtobool(x)), help='whether to include the pre-adaptation (step-0) query loss into the meta-loss')
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
    parser.add_argument('--inner_head_lr_mult', default=1.0, type=float, help='learning rate multiplier for head parameters in inner loop')
    parser.add_argument('--inner_backbone_lr_mult', default=0.1, type=float, help='learning rate multiplier for backbone parameters')
    parser.add_argument('--inner_steps', default=5, type=int, help='number of inner steps for meta-learning (increased for ANIL)')
    parser.add_argument('--inner_steps_max', default=8, type=int, help='maximum inner steps for cosine schedule')
    parser.add_argument('--inner_steps_schedule', default='constant', type=str, choices=['constant', 'cosine'], help='inner steps schedule type (simplified for ANIL)')
    
    parser.add_argument('--eval_lr', default=5e-3, type=float, help='inner learning rate for meta-learning evaluation')
    parser.add_argument('--eval_steps', default=8, type=int, help='inner steps for meta-learning evaluation')
    
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
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--n_fft', default=200, type=int)
    parser.add_argument('--hop_length', default=100, type=int)

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
            channels=config['channels'], 
            kernel_size=config['kernel_size'], 
            act=config['act'], 
            pooling=config['pooling'], 
            embed_dim=config['embed_dim'], 
            num_groups=config['num_groups'],
            fs=config['fs'], 
            input_seq_len_s=config['input_seq_len_s']
        )
    elif config['model_name'] == 'BIOT':
        encoder = BIOT.BIOT(
            ecg=config['ecg'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            pretrained_path=config['pretrained_encoder_ckpt_path']
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
    prediction_head = None
    
    if config['model_name'] == 'ResGruNet':   
        prediction_head = ResGruNet.ResGruNetPredictionHead(
            embed_dim=config['embed_dim'],
            output_dim=config['input_seq_len_s'] * config['fs'] if config['sig2sig'] else 3  # Full waveform or SBP/DBP/MAP
        )            
    elif config['model_name'] == 'BIOT':
        total_input_channels = 2 if config['ecg'] else 1
        prediction_head = BIOT.BIOTPredictionHead(
            embed_dim=config['embed_dim'] * total_input_channels,
            output_dim=config['input_seq_len_s'] * config['fs'] if config['sig2sig'] else 3  # Full waveform or SBP/DBP/MAP
        )
    else:
        raise ValueError("Invalid model name ...")
    
    if prediction_head is None:
        raise ValueError("Error occurred during model arhcitecture intialization ...")
    
    return prediction_head

    
def get_meta_lr(epoch, config):
    
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


class LSLRStepSize(torch.nn.Module):
    def __init__(self, model, init_lr=0.01):
        super().__init__()
        self.lrs = torch.nn.ParameterDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.lrs[name] = torch.nn.Parameter(torch.ones(1) * init_lr)

    def forward(self, grads, params):
        # grads, params are dicts with same keys
        updated = {}
        for name in params:
            updated[name] = params[name] - self.lrs[name] * grads[name]
        return updated
    
    
def build_inner_optimizer(adapted_model, adapted_regressor, base_lr, config):
    """
    Build inner optimizer for meta-learning algorithms.
    Behaviors:
    - inner_adapt='all'  : adapt backbone + regressor
    - inner_adapt='head' : freeze backbone, adapt only regressor
    Also supports per-group LR multipliers and opt type.
    """
    mode = config.get('inner_adapt')   
    opt_type = config.get('inner_opt').lower()  
    head_mult = float(config.get('inner_head_lr_mult'))
    bb_mult   = float(config.get('inner_backbone_lr_mult'))
    
    # Decide which params to adapt
    backbone_params = [p for p in adapted_model.parameters()
                       if p.is_floating_point() or p.is_complex()]
    head_params = [p for p in adapted_regressor.parameters()
                   if p.is_floating_point() or p.is_complex()]

    # Default to 'head' - ANIL
    if mode == 'all':
        # train both
        for p in backbone_params:
            p.requires_grad = True
        for p in head_params:
            p.requires_grad = True
        params = [
            {'params': backbone_params, 'lr': base_lr * bb_mult},
            {'params': head_params,     'lr': base_lr * head_mult},
        ]
    else:    
        # freeze backbone
        for p in backbone_params:
            p.requires_grad = False
        params = [
            {'params': head_params, 'lr': base_lr * head_mult},
        ]
    
    # Build optimizer, default to Adam
    if opt_type == 'sgd':
        momentum = float(config.get('sgd_momentum'))
        inner_opt = torch.optim.SGD(params, lr=base_lr, momentum=momentum)
    else:
        inner_opt = torch.optim.Adam(params, lr=base_lr)
    
    return inner_opt
    

def get_backbone_and_head_params(model, regressor):
        backbone_params, head_params = [], []
        for _, p in model.named_parameters():
            backbone_params.append(p)
        for _, p in regressor.named_parameters():
            head_params.append(p)
        return backbone_params, head_params
    

def linear_warmup(current_epoch, warmup_epochs, base_lr):
    if current_epoch >= warmup_epochs:
        return base_lr
    return base_lr * (0.1 + 0.9 * (current_epoch / warmup_epochs))

                
def supcon_loss(features, config, labels=None, mask=None):
    """
    Supervised Contrastive Loss / SimCLR Loss
    """
    device = features.device
    
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                        'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    contrast_count = features.shape[1]  # Should be 2 for SimCLR
    
    # Reshape: [batch_size * n_views, embedding_dim]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    
    # Normalize embeddings (crucial for contrastive learning)
    contrast_feature = F.normalize(contrast_feature, dim=1)
    
    # Compute cosine similarity matrix
    cos_sim = torch.matmul(contrast_feature, contrast_feature.T)

    # Create self-mask to exclude self-comparisons
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)

    # Create positive pair mask
    if labels is not None:
        # Supervised case: Use labels to create positive pairs
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        labels = torch.cat([labels] * contrast_count, dim=0)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        # Remove self-comparisons from positive mask
        pos_mask = pos_mask * (~self_mask).float()
    else:
        # Unsupervised SimCLR case: Create positive pairs for augmentations
        pos_mask = torch.zeros((batch_size * contrast_count, batch_size * contrast_count), dtype=torch.float, device=device)
        
        # For each original sample, its augmentations form positive pairs
        for i in range(batch_size):
            for j in range(contrast_count):
                for k in range(contrast_count):
                    if j != k:  # Different augmentations of same sample
                        pos_mask[i * contrast_count + j, i * contrast_count + k] = 1.0

    # Scale by temperature
    logits = cos_sim / config['temperature']
    
    # For numerical stability, subtract max
    logits_max = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - logits_max.detach()

    # Create negative mask (all except self-comparisons)
    neg_mask = (~self_mask).float()

    # Compute exp(logits) only for valid negatives
    exp_logits = torch.exp(logits) * neg_mask

    # Compute log probabilities
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Compute mean log probability over positive pairs
    pos_pairs_per_sample = pos_mask.sum(dim=1)
    
    # Handle case where no positive pairs exist (shouldn't happen in proper SimCLR)
    valid_samples = pos_pairs_per_sample > 0
    
    if valid_samples.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), cos_sim, pos_mask
    
    # Average log prob over positive pairs for each sample
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_pairs_per_sample + 1e-8)
    
    # Only consider samples with positive pairs
    mean_log_prob_pos = mean_log_prob_pos[valid_samples]
    
    # Final loss (negative log likelihood)
    loss = -mean_log_prob_pos.mean()

    return loss, cos_sim, pos_mask


def get_msl_weights(epoch, config, num_inner_steps, include_pre=False):
    """
    Returns a list of weights (length num_inner_steps [+1 if include_pre]),
    normalized to sum to 1. The weights are annealed over epochs so later
    inner steps get more weight as training proceeds.
    """
    # anneal control: 0 -> no anneal (uniform), 1 -> full anneal (linear bias to later steps)
    anneal_epochs = config.get('msl_anneal_epochs')
    anneal_factor = float(min(epoch, anneal_epochs)) / max(1.0, anneal_epochs)

    # baseline: uniform on the post-update steps
    steps_idx = list(range(1, num_inner_steps + 1))  # 1..S
    # Raw weight for step s = 1 + anneal_factor * s  (so later steps get higher weight)
    raw = [1.0 + anneal_factor * float(s) for s in steps_idx]

    if include_pre:
        # give pre-adapt a small base weight (0.5) that also decays when anneal grows
        pre_base = config.get('msl_pre_base_weight')
        raw = [pre_base] + raw

    # Normalize
    total = sum(raw)
    weights = [r / total for r in raw]
    return weights


def compute_embedding_stats(encoder, dataloader, device):
    """
    Compute mean and std of embeddings over a dataloader.
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
