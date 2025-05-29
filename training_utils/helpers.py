import os
import sys
folders_to_add = ['models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import time
import yaml
import itertools
import numpy as np
import random
import torch
from models import ResGRUNet, PhysioFormer, SSLUNet, UNet, GRU, Transformer
from models.UNet import AttentionGate1D, SelfAttentionBlock1D


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
            The tuning strategy. Options are 'all', 'gru' or 'transformer', 'projection_head', 'last_linear', or 'none'.
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
    elif config['model_name'] == 'SSLUNet':
        _handle_sslunet_tuning(model, tune, config)
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
    
    elif tune == 'last_linear':
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
    
    elif tune == 'last_linear':
        _freeze_physioformer_projections(model, config)
        for p in model.transformer.parameters():
            p.requires_grad = False
        for p in model.projection_head.parameters():
            p.requires_grad = False
    
    else:
        raise ValueError(f"Invalid tuning strategy for PhysioFormer: {tune}")
    
    
def _handle_gru_tuning(model, tune, config):
    """Handle tuning strategy for GRU model."""
    
    # Special cases first all/none parameters tuned
    if tune == 'all': 
        return
    elif tune == 'none':
        for p in model.parameters():
            p.requires_grad = False
        return
    
    # Model-specfic tune, look into the corresponding script where the architecture is defined
    # For GRU either you fine-tune all the model parameters or the last linear
    if tune == 'last_linear':
        # Freeze the GRU layers (look into GRU.py for the layers'name)
        for p in model.gru.parameters():
            p.requires_grad = False
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
    elif tune == 'last_linear':
        # Freeze everything and only leave final_conv with require_grads True
        # Model was already all tunable, but code is cleaner this way 
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
    elif tune == 'decoder':
        # Freeze encoder and bottleneck, unfreeze decoder and attention gates, and final self-attention
        for p in model.parameters():
            p.requires_grad = False
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
        for p in model.final_conv.parameters():
            p.requires_grad = True
    elif tune == 'encoder':
        # Freeze decoder and attention gates, unfreeze encoder and bottleneck
        for p in model.parameters():
            p.requires_grad = False
        for block in model.encoder_blocks:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.bottleneck.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for UNet: {tune}")
    

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
    elif tune == 'last_linear':
        # Only tune the final supervised convolutional layer
        for p in model.final_conv_supervised.parameters():
            p.requires_grad = True
    elif tune == 'projection_head':
        # Only tune the projection head
        for p in model.projection_head.parameters():
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
                "projection_head",
                "reconstruction_head",
                "permutation_head",
                "generation_prev_head",
                "generation_next_head"
            ]):
                param.requires_grad = True
    else:
        raise ValueError(f"Invalid tuning strategy for SSLUNet: {tune}")


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
            ppg_derivatives=config['ppg_derivatives'], 
            ppg_emd=config['ppg_emd'], 
            ppg_freqs=config['ppg_freqs'],
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
            ppg_derivatives=config['ppg_derivatives'], 
            ppg_emd=config['ppg_emd'], 
            ppg_freqs=config['ppg_freqs'],
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
            ppg_derivatives=config['ppg_derivatives'],
            ppg_emd=config['ppg_emd'],
            ppg_freqs=config['ppg_freqs'],
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
            ppg_derivatives=config['ppg_derivatives'],
            ppg_emd=config['ppg_emd'],
            ppg_freqs=config['ppg_freqs'],
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
            ppg_derivatives=config['ppg_derivatives'],
            ppg_emd=config['ppg_emd'],
            ppg_freqs=config['ppg_freqs'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            num_encoder_layers=config['num_encoder_layers']
        )   
    elif config['model_name'] == 'SSLUNet':
        model = SSLUNet.SSLUNet(
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            ppg_derivatives=config['ppg_derivatives'],
            ppg_emd=config['ppg_emd'],
            ppg_freqs=config['ppg_freqs'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            channels=config['channels'],
            kernel_size=config['kernel_size'],
            num_heads_attention=config['num_heads_attention'],
            dim_feedforward_attention=config['dim_feedforward_attention'],
            projection_hidden_dim=config['proj_hidden_dim'],
            projection_dim=config['proj_head_dim']
            )
    else:
        raise ValueError("Invalid model name ...")
    
    return model