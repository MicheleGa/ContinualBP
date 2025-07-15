import os
import sys
folders_to_add = ['models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import argparse
from distutils.util import strtobool
import time
import yaml
import numpy as np
import random
import torch
from models import ResGRUNet, PhysioFormer, SSLUNet, UNet, GRU, Transformer, EUNet, SSLEUNet
from models.UNet import AttentionGate1D, SelfAttentionBlock1D


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
    parser.add_argument('--smoothl1loss_beta', default=5, type=int, help='beta for SmoothL1Loss')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--l2norm', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--sgd_momentum', default=0.9, type=float, help='Momentum for SGD optimizer')
    parser.add_argument('--lrsched_step', default="5, 10, 15, 20, 40", type=str, help='learning rate scheduler steps')
    parser.add_argument('--lrsched_gamma', default=0.5, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--optimizer_type', default="AdamW", type=str, help='optimizer type')
    parser.add_argument('--lr_scheduler_type', default="MultiStepLR", type=str, help='learning rate scheduler type')
    parser.add_argument('--lr_scheduler_enable', default='False', type=lambda x: bool(strtobool(x)), help='enable learning rate scheduler')
    parser.add_argument('--lr_scheduler_min_lr', default=0.0001, type=float, help='enable learning rate scheduler')
    parser.add_argument('--lr_scheduler_warmup', default=0, type=int, help='enable learning rate scheduler')
    
    # Loss function Setup (supervised and self-supervised)
    parser.add_argument('--criterion', default="MSELoss", type=str, help='loss criterion')
    parser.add_argument('--lambda_supervised', default=0., type=float, help='scale supervised loss function contribution, set 0. to prevent its application')
    parser.add_argument('--lambda_ortho', default=0., type=float, help='induce feature orthogonality during pretraining, set 0. to prevent its application')
    parser.add_argument('--lambda_contrastive', default=0., type=float, help='induce contrastive loss contribution to the final loss during pretraining, set 0. to prevent its application')
    parser.add_argument('--temperature', default=0., type=float, help='temperature scalar for SimCLR loss')
    parser.add_argument('--lambda_msr', default=0., type=float, help='induce masked signal loss contribution to the final loss during pretraining, set 0. to prevent its application')
    parser.add_argument('--lambda_cwg', default=0., type=float, help='induce prev/next signal reconstruction loss contribution to the final loss during pretraining, set 0. to prevent its application')
    
    # Dataset Setup
    parser.add_argument('--dataset_folder', default='./data/lmdb', type=str, help='path to dataset fodler')
    parser.add_argument('--dataset_name', default='test', type=str, help='name of the dataset')
    parser.add_argument('--lp_dataset_name', default='test', type=str, help='name of the dataset for linear probing')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.1,0.2', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--mix_pretraining_subject_samples', default='True', type=lambda x: bool(strtobool(x)), help='whether to mix pretraining subject samples among train/val/test or not')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    
    # Personalization Setup
    parser.add_argument('--personalization_sample_number', default=100, type=float, help='number of samples to take for personalization (-1.0 to use all samples)')
    parser.add_argument('--num_personalization_subjects', default=100, type=int, help='number of subjects to for personalization')
    parser.add_argument('--num_passes', default=3, type=int, help='number of passes on the same batch during personalization')
    parser.add_argument('--tune', default='all', type=str, help='which part of the model to training during pretraining')

    # Self-supervision Setup
    parser.add_argument('--apply_masking', default='False', type=lambda x: bool(strtobool(x)), help='whether to apply masking for self-supervision or not')
    parser.add_argument('--masking_ratio', default=0.08, type=float, help='Ratio of signal length to mask for MSR task')
    parser.add_argument('--masking_strategy', default='physiological', type=str, help='which type of masking to apply for self-supervision')
    parser.add_argument('--augmentation_types', default='jitter,scaling,magnitude_warp,flip', type=str, help='Comma-separated list of augmentation types for SimCLR')
    parser.add_argument('--aug_prob', default=0.5, type=float, help='Probability for each individual augmentation in RandomAugmentor')
    parser.add_argument('--data_aug', default='False', type=lambda x: bool(strtobool(x)), help='whether to use data augmentations also for the MSR and CWG SSL tasks or not')
    parser.add_argument('--ssl', default='False', type=lambda x: bool(strtobool(x)), help='whether to do self-supervised pretraining or not')
        
    # Model Setup
    parser.add_argument('--model', default="models.ResGRUNet", type=str, help='model name')
    parser.add_argument('--pretrained_model_checkpoint', default="./checkpoints", type=str, help='pretrained model path')
    parser.add_argument('--proj_head_dim', default=256, type=int, help='dimension of the projection head after the feture extractor') 
    
    # ResGRUNet Setup
    #parser.add_argument('--gru', default='True', type=lambda x: bool(strtobool(x)), help='whether to use a GRU after CNN or not')
    #parser.add_argument('--channels', default='1, 32, 64, 128', type=str, help='channels produced by the convolutional blocks')
    #parser.add_argument('--act', default='leaky_relu', type=str, help='which activation to use (ReLU or LeakyReLU)')
    #parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    #parser.add_argument('--pooling', default='avg', type=str, help='which poolng to use (average or max)')
    #parser.add_argument('--supervised', default='False', type=lambda x: bool(strtobool(x)), help='whether to use the model with pretrianing or not')
        
    ## PhysioFormer Setup
    #parser.add_argument('--embed_dim', default=64, type=int, help='input embedding size')
    #parser.add_argument('--hidden_dim', default=256, type=int, help='transformer hidden dimension')
    #parser.add_argument('--num_layers', default=3, type=int, help='number of transformer layers')
    #parser.add_argument('--n_head', default=4, type=int, help='number of self-attention heads')
    #parser.add_argument('--head_dim', default=16, type=int, help='self-attention headd dimension')
    
    # UNet Setup
    parser.add_argument('--channels', default='32, 64, 128, 256, 512', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--num_heads_attention', default=1, type=int, help='heads number of the final self-attention layer') 
    parser.add_argument('--dim_feedforward_attention', default=128, type=int, help='dimension of the final self-attention layer') 
    parser.add_argument('--kernel_size', default=3, type=int, help='convolutional layer kernel size')
    
    # Efficient UNet Setup
    parser.add_argument('--attention_type', default='self_attention', type=str, choices=['self_attention', 'nystrom_attention', 'gru'], help='Type of final processing layer: "self_attention", "nystrom_attention", or "gru"')
    
    # SSL UNet additional Setup
    parser.add_argument('--proj_hidden_dim', default=512, type=int, help='Dimension of the projection head hidden dim for SimCLR')
    
    # GRU Setup
    parser.add_argument('--hidden_dim', default=128, type=int, help='GRU hidden dimension size')
    parser.add_argument('--num_layers', default=2, type=int, help='number of GRU layers')
    parser.add_argument('--bidirectional', default=True, type=lambda x: bool(strtobool(x)), help='whether to use bidirectional GRU or not')
        
    # Transformer Setup
    parser.add_argument('--embed_dim', default=32, type=int, help='transformer embedding dimension size')
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads for the self-attention mechanism')
    parser.add_argument('--num_encoder_layers', default=2, type=int, help='number of trasnformer layers')
    parser.add_argument('--dim_feedforward', default=128, type=int, help='feedforward dimension size in the transformer encoder') 
        
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


def numpy_mse_loss(outputs, targets):
    r"""
    Calculates MSE loss using NumPy.
    
    Parameters
    ------------
        outputs (np.ndarray): 
            Predicted values.
        targets (np.ndarray): 
            Ground truth values.
    
    Returns
    ------------
        float: 
            The mean squared error loss.
    """
    return np.mean((outputs - targets)**2)


def numpy_smooth_l1_loss(outputs, targets, beta=1.0):
    """Calculates Smooth L1 loss using NumPy."""
    absolute_error = np.abs(outputs - targets)
    quadratic_error = 0.5 * absolute_error**2 / beta
    linear_error = absolute_error - 0.5 * beta
    return np.mean(np.where(absolute_error < beta, quadratic_error, linear_error))

    
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
    to_save['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        to_save['lr_scheduler'] = scheduler.state_dict()
    to_save['val_loss'] = meter.avg

    if subject_id is not None:
        out_path = os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt') 
    else:
        out_path = os.path.join(checkpoint_path, save_name, 'ckpt')
    
    os.makedirs(out_path, exist_ok=True)
    torch.save(to_save, os.path.join(out_path, model_name))
    with open(os.path.join(out_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=True, default_flow_style=False)
    
    print(f"Epoch {epoch} - checkpoint saved in {out_path}")
        
    
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
        checkpoint = torch.load(os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt', model_name), weights_only=False)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt', model_name)}")
    else:
        checkpoint = torch.load(os.path.join(checkpoint_path, save_name, 'ckpt', model_name), weights_only=False)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_path, save_name, 'ckpt', model_name)}")
        
    epoch = int(checkpoint['epoch'])
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    print(f"Loaded checkpoint from validation on epoch {epoch}: {checkpoint['val_loss']}")
    

def configure_optimizer_and_scheduler(model, config):
    r"""
    Configure the optimizer and learning rate scheduler.

    Parameters
    ------------
        parameters (iterable): 
            Model parameters to optimize.
        config (dict): 
            Configuration dictionary containing optimizer and scheduler settings.

    Returns
    ------------
        dict: 
            A dictionary containing the optimizer and optionally the scheduler.
    """
    
    # Configure optimizer
    if config['optimizer_type'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=config['l2norm'])
    elif config['optimizer_type'] == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=config['l2norm'])
    elif config['optimizer_type'] == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=config['l2norm'], momentum=config['sgd_momentum'])
    elif config['optimizer_type'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=config['l2norm'])
    else:
        raise ValueError("Invalid optimizer type specified in config.")

    # Configure learning rate scheduler
    # TODO: decide where to put the MultistepLR in the training loop (after optimizer.step() or after the loop over batches just before validation?)
    if config['lr_scheduler_enable']:
        if config['lr_scheduler_type'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=tuple([int(s) for s in config['lrsched_step'].split(sep=",")]),
                gamma=config['lrsched_gamma']
            )
            return {"optimizer": optimizer, 'lr_scheduler': scheduler}
        
        elif config['lr_scheduler_type'] == 'ExponentialLR':
            lr_start = config['lr']
            lr_end = config['lr_scheduler_min_lr'] 
            gamma = (lr_end / lr_start)**(1 / config['max_training_epochs'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=gamma
            )
            return {"optimizer": optimizer, 'lr_scheduler': scheduler}
        
        elif config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
            warmup_steps = config['lr_scheduler_warmup'] * config['steps_per_epoch']
            annealing_steps = (config['max_training_epochs'] - config['lr_scheduler_warmup']) * config['steps_per_epoch']

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config['lr_scheduler_min_lr'] / config['lr'],
                total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=annealing_steps,
                eta_min=config['lr_scheduler_min_lr']
            )

            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            raise ValueError("Invalid learning rate scheduler type specified in config.")
    else:
        return {"optimizer": optimizer}


def set_trainable_parameters(model, tune, config):
    r""" 
    Set the parameters of the model to be trainable or not based on the tuning strategy.
    
    Parameters
    ------------
        model (torch.nn.Module): 
            The model to be tuned.
        tune (str): 
            The tuning strategy. Options are 'all', 'gru' or 'transformer', 'projection_head', 'last_layer', or 'none'.
        config (dict): 
            Configuration dictionary containing model settings.
    
    Returns
    ------------
        None: 
            The function modifies the model parameters in place.
    """
    
    # Model-specific parameter freezing
    if config['model_name'] == 'ResGRUNet':
        _handle_resgru_tuning(model, tune, config)
    elif config['model_name'] == 'PhysioFormer':
        _handle_physioformer_tuning(model, tune, config)
    elif config['model_name'] == 'GRU':
        _handle_gru_tuning(model, tune, config)
    elif config['model_name'] == 'UNet':
        _handle_unet_tuning(model, tune, config)
    elif config['model_name'] == 'EUNet':
        _handle_eunet_tuning(model, tune, config)
    elif config['model_name'] == 'SSLUNet':
        _handle_sslunet_tuning(model, tune, config)
    elif config['model_name'] == 'SSLEUNet':
        _handle_ssl_eunet_tuning(model, tune, config)
    else:
        raise ValueError(f"Invalid model name: {config['model_name']}")


def _freeze_resgru_feature_generators(model, config):
    """Freeze feature generators for ResGRUNet based on config."""
    for p in model.ppg_feature_gen.parameters():
        p.requires_grad = False
    
    if config['ecg']:
        for p in model.ecg_feature_gen.parameters():
            p.requires_grad = False
    
    if config['resp']:
        for p in model.resp_feature_gen.parameters():
            p.requires_grad = False


def _freeze_physioformer_projections(model, config):
    """Freeze linear projections for PhysioFormer based on config."""
    for p in model.ppg_linear_projection.parameters():
        p.requires_grad = False
    
    if config['ecg']:
        for p in model.ecg_linear_projection.parameters():
            p.requires_grad = False
    
    if config['resp']:
        for p in model.resp_linear_projection.parameters():
            p.requires_grad = False


def _handle_resgru_tuning(model, tune, config):
    """Handle tuning strategy for ResGRUNet model."""
    
    # Special cases first all/none parameters tuned
    if tune == 'all': 
        return
    elif tune == 'none':
        for p in model.parameters():
            p.requires_grad = False
        return
    
    # Model-specfic tune, look into the corresponding script where the architecture is defined
    if tune in ['gru', 'transformer']:
        _freeze_resgru_feature_generators(model, config)
    
    elif tune == 'projection_head':
        _freeze_resgru_feature_generators(model, config)
        for p in model.t_gru.parameters():
            p.requires_grad = False
    
    elif tune == 'last_layer':
        _freeze_resgru_feature_generators(model, config)
        for p in model.t_gru.parameters():
            p.requires_grad = False
        for p in model.projection_head.parameters():
            p.requires_grad = False
    
    else:
        raise ValueError(f"Invalid tuning strategy for ResGRUNet: {tune}")


def _handle_physioformer_tuning(model, tune, config):
    """Handle tuning strategy for PhysioFormer model."""
    
    # Special cases first all/none parameters tuned
    if tune == 'all': 
        return
    elif tune == 'none':
        for p in model.parameters():
            p.requires_grad = False
        return
    
    # Model-specfic tune, look into the corresponding script where the architecture is defined
    if tune in ['gru', 'transformer']:
        _freeze_physioformer_projections(model, config)
    
    elif tune == 'projection_head':
        _freeze_physioformer_projections(model, config)
        for p in model.transformer.parameters():
            p.requires_grad = False
    
    elif tune == 'last_layer':
        _freeze_physioformer_projections(model, config)
        for p in model.transformer.parameters():
            p.requires_grad = False
        for p in model.projection_head.parameters():
            p.requires_grad = False
    
    else:
        raise ValueError(f"Invalid tuning strategy for PhysioFormer: {tune}")
    
    
def _handle_gru_tuning(model, tune, config):
    """Handle tuning strategy for GRU model."""
    
    if tune == 'all': 
        # Train all parameters
        for p in model.parameters():
            p.requires_grad = True
    elif tune == 'none':
        # Train no parameters
        for p in model.parameters():
            p.requires_grad = False
    elif tune == 'last_layer':
        # Freeze the GRU layers (look into GRU.py for the layers'name)
        for p in model.gru.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for GRU: {tune}")


def _handle_unet_tuning(model, tune, config):
    """Handle tuning strategy for UNet model."""
    
    if tune == 'all': 
        # Train all parameters
        for p in model.parameters():
            p.requires_grad = True
            
    elif tune == 'none':
        # Train no parameters
        for p in model.parameters():
            p.requires_grad = False
            
    elif tune == 'last_layer':
        # Freeze everything and only leave final_conv with require_grads True
        for p in model.parameters():
            p.requires_grad = False
        for p in model.final_conv.parameters():
            p.requires_grad = True
            
    elif tune == 'attention':
        # Freeze everything except the AttentionGate1D and SelfAttentionBlock1D modules
        for p in model.parameters():
            p.requires_grad = False
        for module in model.modules():
            if isinstance(module, (AttentionGate1D, SelfAttentionBlock1D)):
                for p in module.parameters():
                    p.requires_grad = True
        # Also unfreeze the final convolution layer, as it's typically part of the "head"
        for p in model.final_conv.parameters():
            p.requires_grad = True
            
    elif tune == 'decoder':
        # Freeze encoder and bottleneck, unfreeze decoder and attention gates, and final self-attention
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze all decoder blocks
        for block in model.decoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        
        # Unfreeze all attention gates
        for gate in model.att_gates: # Iterate directly over the nn.ModuleList
            for p in gate.parameters():
                p.requires_grad = True
        
        # Unfreeze self-attention stacks
        for p in model.self_attention_stack1.parameters():
            p.requires_grad = True
        for p in model.self_attention_stack2.parameters():
            p.requires_grad = True
        
        # Unfreeze final convolution
        for p in model.final_conv.parameters():
            p.requires_grad = True
    elif tune == 'encoder':
        # Freeze decoder and attention gates, unfreeze encoder and bottleneck
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze all encoder blocks
        for block in model.encoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        
        # Unfreeze bottleneck
        for p in model.bottleneck.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for UNet: {tune}")
        

def _handle_eunet_tuning(model, tune, config):
    """Handle tuning strategy for UNet model."""
    
    if tune == 'all': 
        # Train all parameters
        for p in model.parameters():
            p.requires_grad = True
    elif tune == 'none':
        # Train no parameters
        for p in model.parameters():
            p.requires_grad = False
    elif tune == 'last_layer':
        # Freeze everything and only leave final_conv with require_grads True
        for p in model.parameters():
            p.requires_grad = False
        for p in model.final_conv.parameters():
            p.requires_grad = True
    elif tune == 'attention':
        # Freeze everything except the AttentionGate1D and the final attention/GRU modules
        for p in model.parameters():
            p.requires_grad = False
        for module in model.att_gates: # Iterate through the ModuleList of AttentionGates
            for p in module.parameters():
                p.requires_grad = True
        
        # Unfreeze the selected final processing layer
        if model.attention_type == 'self_attention':
            for p in model.self_attention_stack1.parameters():
                p.requires_grad = True
            for p in model.self_attention_stack2.parameters():
                p.requires_grad = True
        elif model.attention_type == 'nystrom_attention':
            for p in model.nystrom_attention_block.parameters():
                p.requires_grad = True
        elif model.attention_type == 'gru':
            for p in model.gru_layer.parameters():
                p.requires_grad = True
            for p in model.gru_proj.parameters():
                p.requires_grad = True
        
        # Also unfreeze the final convolution layer, as it's typically part of the "head"
        for p in model.final_conv.parameters():
            p.requires_grad = True

    elif tune == 'decoder':
        # Freeze encoders and bottleneck, unfreeze decoder blocks, attention gates, and final attention/GRU
        for p in model.parameters():
            p.requires_grad = False
        
        for block in model.decoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        for gate in model.att_gates: # Iterate through the ModuleList of AttentionGates
            for p in gate.parameters():
                p.requires_grad = True
        
        # Unfreeze the selected final processing layer
        if model.attention_type == 'self_attention':
            for p in model.self_attention_stack1.parameters():
                p.requires_grad = True
            for p in model.self_attention_stack2.parameters():
                p.requires_grad = True
        elif model.attention_type == 'nystrom_attention':
            for p in model.nystrom_attention_block.parameters():
                p.requires_grad = True
        elif model.attention_type == 'gru':
            for p in model.gru_layer.parameters():
                p.requires_grad = True
            for p in model.gru_proj.parameters():
                p.requires_grad = True
        
        for p in model.final_conv.parameters():
            p.requires_grad = True
            
    elif tune == 'encoder':
        # Freeze decoder, attention gates, and final attention/GRU; unfreeze all modality-specific encoders and bottleneck
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze modality-specific encoders
        for encoder_blocks in [model.ppg_encoder_blocks, model.ecg_encoder_blocks, model.resp_encoder_blocks]:
            for block in encoder_blocks:
                for p in block.parameters():
                    p.requires_grad = True
        
        # Unfreeze bottleneck
        for p in model.bottleneck.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for EUNet: {tune}")
        

def _handle_sslunet_tuning(model, tune, config):
    """Handle tuning strategy for SSLUNet model."""

    # Ensure all parameters are initially set to requires_grad=False
    # This provides a clean slate for selective unfreezing given the particular SSLUNet definition
    for p in model.parameters():
        p.requires_grad = False

    if tune == 'all':
        for p in model.parameters():
            p.requires_grad = True
    elif tune == 'none':
        # All parameters already set to False 
        pass
    elif tune == 'last_layer':
        # Only tune the final supervised convolutional layer
        for p in model.final_conv_supervised.parameters():
            p.requires_grad = True
    elif tune == 'reconstruction_head':
        # Only tune the reconstruction head
        for p in model.reconstruction_head.parameters():
            p.requires_grad = True
    elif tune == 'encoder':
        # Only tune the encoder blocks and bottleneck
        for block in model.encoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.bottleneck.parameters():
            p.requires_grad = True
    elif tune == 'decoder':
        # Only tune the decoder blocks and attention gates
        for block in model.decoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        for gate in [model.att_gate1, model.att_gate2, model.att_gate3, model.att_gate4]:
            for p in gate.parameters():
                p.requires_grad = True
        for p in model.self_attention_stack1.parameters():
            p.requires_grad = True
        for p in model.self_attention_stack2.parameters():
            p.requires_grad = True
    elif tune == 'attention_modules':
        # Tune only the attention gates and self-attention stacks
        for gate in [model.att_gate1, model.att_gate2, model.att_gate3, model.att_gate4]:
            for p in gate.parameters():
                p.requires_grad = True
        for p in model.self_attention_stack1.parameters():
            p.requires_grad = True
        for p in model.self_attention_stack2.parameters():
            p.requires_grad = True
    elif tune == 'shared_backbone':
        # Tune the entire shared U-Net backbone (encoder, bottleneck, decoder, attention, self-attention, and the final_conv_supervised)
        # This means everything *except* the specific SSL heads
        for name, param in model.named_parameters():
            if not any(head_name in name for head_name in [
                "reconstruction_head"
            ]):
                param.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for SSLUNet: {tune}")
    

def _handle_ssl_eunet_tuning(model, tune, config):
    """Handle tuning strategy for SSLEUNet model, considering SSL heads."""
    
    if tune == 'all': 
        # Train all parameters
        for p in model.parameters():
            p.requires_grad = True
    elif tune == 'none':
        # Train no parameters
        for p in model.parameters():
            p.requires_grad = False
    elif tune == 'last_layer':
        # Freeze everything and only leave final_conv_supervised and SSL heads with require_grads True
        for p in model.parameters():
            p.requires_grad = False
        
        # Also unfreeze the final convolution layer for supervised task, and SSL heads
        for p in model.final_conv_supervised.parameters():
            p.requires_grad = True
        for p in model.reconstruction_head.parameters():
            p.requires_grad = True

    elif tune == 'attention':
        # Freeze everything except the AttentionGate1D and the final attention/GRU modules
        for p in model.parameters():
            p.requires_grad = False
        for module in model.att_gates: # Iterate through the ModuleList of AttentionGates
            for p in module.parameters():
                p.requires_grad = True
        
        # Unfreeze the selected final processing layer
        if model.attention_type == 'self_attention':
            for p in model.self_attention_stack1.parameters():
                p.requires_grad = True
            for p in model.self_attention_stack2.parameters():
                p.requires_grad = True
        elif model.attention_type == 'nystrom_attention':
            for p in model.nystrom_attention_block.parameters():
                p.requires_grad = True
        elif model.attention_type == 'gru':
            for p in model.gru_layer.parameters():
                p.requires_grad = True
            for p in model.gru_proj.parameters():
                p.requires_grad = True
        
        # Also unfreeze the final convolution layer for supervised task, and SSL heads
        for p in model.final_conv_supervised.parameters():
            p.requires_grad = True
        for p in model.reconstruction_head.parameters():
            p.requires_grad = True

    elif tune == 'decoder':
        # Freeze encoders and bottleneck, unfreeze decoder blocks, attention gates, and final attention/GRU
        for p in model.parameters():
            p.requires_grad = False
        
        for block in model.decoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        for gate in model.att_gates: # Iterate through the ModuleList of AttentionGates
            for p in gate.parameters():
                p.requires_grad = True
        
        # Unfreeze the selected final processing layer
        if model.attention_type == 'self_attention':
            for p in model.self_attention_stack1.parameters():
                p.requires_grad = True
            for p in model.self_attention_stack2.parameters():
                p.requires_grad = True
        elif model.attention_type == 'nystrom_attention':
            for p in model.nystrom_attention_block.parameters():
                p.requires_grad = True
        elif model.attention_type == 'gru':
            for p in model.gru_layer.parameters():
                p.requires_grad = True
            for p in model.gru_proj.parameters():
                p.requires_grad = True
        
        # Also unfreeze the final convolution layer for supervised task, and SSL heads
        for p in model.final_conv_supervised.parameters():
            p.requires_grad = True
        for p in model.reconstruction_head.parameters():
            p.requires_grad = True
            
    elif tune == 'encoder':
        # Freeze decoder, attention gates, final attention/GRU, and all heads; unfreeze all modality-specific encoders and bottleneck
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze modality-specific encoders
        for encoder_blocks in [model.ppg_encoder_blocks, model.ecg_encoder_blocks, model.resp_encoder_blocks]:
            for block in encoder_blocks:
                for p in block.parameters():
                    p.requires_grad = True
        
        # Unfreeze bottleneck
        for p in model.bottleneck.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for SSLEUNet: {tune}")


class EarlyStopping:
    r"""
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, mode='min'):
        r"""
        Parameters
        ------------
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
            mode (str): One of {min, max}. In min mode, training will stop when the quantity
                        monitored has stopped decreasing. In max mode it will stop when the quantity
                        monitored has stopped increasing. Default: min
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val = None
        self.early_stop = False
        self.mode = mode
        if mode not in ['min', 'max']:
            raise ValueError(f"mode {mode} is unknown!")
        if self.mode == 'min':
            self.val_loss_min = np.inf
        else:
            self.val_loss_max = -np.inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val):
        # Check if validation loss is nan
        if np.isnan(val):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val is None:
            self.best_val = val
        elif (val < self.best_val - self.delta and self.mode == 'min') or (val > self.best_val + self.delta and self.mode == 'max'):
            # Significant improvement detected
            self.best_val = val
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
                
def get_model_architecture(config):
    r"""
    Function to get the model architecture based on the configuration.
    
    Parameters
    ------------
        config (dict): 
            Configuration dictionary containing model parameters.
    
    Returns
    ------------
        model (torch.nn.Module): 
            The model architecture initialized with the given configuration.   
    """
    
    
    if config['model_name'] == 'ResGRUNet':   
        model = ResGRUNet.ResGRUNet(
            ecg=config['ecg'],
            resp=config['resp'], 
            channels=config['channels'], 
            kernel_size=config['kernel_size'], 
            act=config['act'], 
            pooling=config['pooling'], 
            proj_head_dim=config['proj_head_dim'], 
            input_seq_len=int(config['input_seq_len_s'] * config['fs']),
            return_embedding=config['return_embedding']
        )
    elif config['model_name'] == 'PhysioFormer':
        model = PhysioFormer.PhysioFormer(
            ecg=config['ecg'],
            resp=config['resp'], 
            embed_dim=config['embed_dim'], 
            n_head=config['n_head'], 
            head_dim=config['head_dim'], 
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'], 
            input_seq_len=int(config['input_seq_len_s'] * config['fs']),
            return_embedding=config['return_embedding']
        )
    elif config['model_name'] == 'UNet':
        model = UNet.UNet(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            channels=config['channels'],
            kernel_size=config['kernel_size'],
            num_heads_attention=config['num_heads_attention'],
            dim_feedforward_attention=config['dim_feedforward_attention']
        )
    elif config['model_name'] == 'GRU':
        model = GRU.GRU(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional']
        )
    elif config['model_name'] == 'Transformer':
        model = Transformer.Transformer(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            num_encoder_layers=config['num_encoder_layers']
        )   
    elif config['model_name'] == 'EUNet':
        model = EUNet.EUNet(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            channels=config['channels'], 
            kernel_size=config['kernel_size'],
            num_heads_attention=config['num_heads_attention'],
            dim_feedforward_attention=config['dim_feedforward_attention'],
            attention_type=config['attention_type'] 
        ) 
    elif config['model_name'] == 'SSLUNet':
        model = SSLUNet.SSLUNet(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            channels=config['channels'],
            kernel_size=config['kernel_size'],
            num_heads_attention=config['num_heads_attention'],
            dim_feedforward_attention=config['dim_feedforward_attention'],
        )
    elif config['model_name'] == 'SSLEUNet':
        model = SSLEUNet.SSLEUNet(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            channels=config['channels'], 
            kernel_size=config['kernel_size'],
            num_heads_attention=config['num_heads_attention'],
            dim_feedforward_attention=config['dim_feedforward_attention'],
            attention_type=config['attention_type'] 
        )
    else:
        raise ValueError("Invalid model name ...")
    
    return model