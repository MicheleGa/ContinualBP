import os
import sys
folders_to_add = ['data', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import shutil
import gc
import copy
import itertools
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from data.dataset_ssl_2 import PhysioDataset
from training_utils.helpers import save_status, load_status, EarlyStopping, set_trainable_parameters, configure_optimizer_and_scheduler, get_model_architecture
from training_utils.metrics import AverageMeter, get_metric_values



def pretraining_training_validation_testing(save_name, checkpoint_path, tensorboard_path, model_name, dataloaders, config, device):

    # Define model architecture
    model = get_model_architecture(config)
    model = model.to(device)

    # Set the model to training mode
    set_trainable_parameters(model=model, tune='all', config=config)

    # Print trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params} / {trainable_params + non_trainable_params} ({trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params} / {trainable_params + non_trainable_params} ({non_trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")


    # Optimizer and Scheduler
    if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
        config['steps_per_epoch'] = len(dataloaders['train']) # Corrected from `train_dataloader` to `dataloaders['train']`

    optim_sched = configure_optimizer_and_scheduler(model, config)

    optimizer = optim_sched['optimizer']
    if config['lr_scheduler_enable']:
        scheduler = optim_sched['lr_scheduler']
    else:
        scheduler = None

    # Dataloaders
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']

    # Logging to TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    # N.B. the field in_channels is common to all models
    example_input_array_graph = torch.rand((config['batch_size'], int(config['input_seq_len_s'] * config['fs']), model.in_channels))
    writer.add_graph(model, example_input_array_graph.to(device))

    # Early stopping
    if config['es_enable']:
        early_stopping = EarlyStopping(
            patience=config['es_patience'],
            delta=config['es_min_delta'],
            verbose=True,
            mode='min'
            )
    else:
        early_stopping = None

    return self_supervised_pretraining_training_validation_testing(
            save_name=save_name,
            checkpoint_path=checkpoint_path,
            writer=writer,
            model_name=model_name,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            early_stopping=early_stopping,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device
            )

def create_masked_signal(signal, mask_indices):
    """
    Applies masking to a signal tensor based on boolean mask_indices.
    Assumes signal is [B, L, C] and mask_indices is [B, L] bool.
    Masked positions are set to 0.0.
    """
    masked_s = signal.clone()
    # Expand mask_indices to match the channel dimension of the signal
    mask_expanded = mask_indices.unsqueeze(-1).expand_as(masked_s)
    masked_s[mask_expanded] = 0.0 # Set masked positions to 0.0 (or other masking strategy)
    return masked_s


def self_supervised_pretraining_training_validation_testing(
    save_name,
    checkpoint_path,
    writer,
    model_name,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    early_stopping,
    optimizer,
    scheduler,
    config,
    device):

    # Best lowest validation loss for early stopping purposes (e.g., combined SSL loss)
    best_val_loss = float("+inf")
    
    # Self-supervised pretraining loop
    for epoch in range(config['max_training_epochs']):

        # Training loop
        model.train() # Set model to training mode
        set_trainable_parameters(model=model, tune='all', config=config) # Parameters to train (e.g., all encoder and SSL heads)
        
        train_losses = AverageMeter(name='train/loss')
        train_msr_losses = AverageMeter(name='train/msr_loss')
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            # Move data to device
            original_signal, _, mask_indices = batch # Original signal amd mask indices for MSR
            original_signal, mask_indices = original_signal.to(device), mask_indices.to(device)
            msr_target_signal = original_signal # Target for MSR is the original signal            
  
            optimizer.zero_grad() # Zero gradients for a new batch

            # --- Forward Passes and Loss Calculations for Each Task ---

            # 1. Masked Signal Reconstruction (MSR)
            msr_loss = torch.tensor(0.0, device=device)
            if config['lambda_msr'] != 0.:
                # Create the masked input from the chosen augmented signal
                masked_input_for_msr = create_masked_signal(original_signal, mask_indices)
                masked_embedding = model.encode(masked_input_for_msr)
                reconstructed_signal = model.reconstruct(masked_embedding)

                # Calculate loss only on masked positions against the chosen augmented target
                if config['criterion'] == 'MSELoss':
                    msr_loss = F.mse_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], msr_target_signal[mask_indices.unsqueeze(-1).expand_as(msr_target_signal)])
                elif config['criterion'] == 'SmoothL1Loss':
                    msr_loss = F.smooth_l1_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], msr_target_signal[mask_indices.unsqueeze(-1).expand_as(msr_target_signal)])
                else:
                    raise ValueError("Invalid criterion ...")

            # --- Total Loss ---
            loss = config.get('lambda_msr', 0.) * msr_loss
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch only for cosine-warmup scheduler
            if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
                scheduler.step()

            # Log training losses
            train_losses.update(loss.item(), original_signal.size(0))
            train_msr_losses.update(msr_loss.item(), original_signal.size(0))
            
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('train/msr_loss', msr_loss.item(), epoch * len(train_dataloader) + batch_idx)
            
        # Log epoch average training losses
        writer.add_scalar('train/loss_epoch', train_losses.avg, epoch)
        writer.add_scalar('train/msr_loss_epoch', train_msr_losses.avg, epoch)

        # Validation loop
        model.eval() # Set model to evaluation mode (siwtch off batch norm/dropout etc.)
        set_trainable_parameters(model=model, tune='none', config=config) # Ensure all params are non-trainable during eval

        val_losses = AverageMeter(name='val/loss')
        val_msr_losses = AverageMeter(name='val/msr_loss')
        
        with torch.no_grad(): # No gradient calculation during validation
            for batch_idx, batch in enumerate(val_dataloader):
                
                original_signal, _, mask_indices = batch.to(device) # Original signal amd mask indices for MSR
                original_signal, mask_indices = original_signal.to(device), mask_indices.to(device)
                msr_target_signal = original_signal # Target for MSR is the original signal         
            
                # --- Forward Passes and Loss Calculations for Each Task (Validation) ---
                val_msr_loss = torch.tensor(0.0, device=device)
                if config['lambda_msr'] != 0.:
                    # MSR input now uses assigned augmented signal in validation
                    val_masked_input_for_msr = create_masked_signal(original_signal, mask_indices)
                    masked_embedding = model.encode(val_masked_input_for_msr)
                    reconstructed_signal = model.reconstruct(masked_embedding)
                    
                    if config['criterion'] == 'MSELoss':
                        val_msr_loss = F.mse_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], msr_target_signal[mask_indices.unsqueeze(-1).expand_as(msr_target_signal)])
                    elif config['criterion'] == 'SmoothL1Loss':
                        val_msr_loss = F.smooth_l1_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], msr_target_signal[mask_indices.unsqueeze(-1).expand_as(msr_target_signal)])
                    else:
                        raise ValueError("Invalid criterion ...")
                    
                # Total validation loss
                val_loss = config.get('lambda_msr', 0.) * val_msr_loss

                # Update validation meters
                val_losses.update(val_loss.item(), original_signal.size(0))
                val_msr_losses.update(val_msr_loss.item(), original_signal.size(0))

        # Log epoch validation losses to TensorBoard
        writer.add_scalar('val/loss_epoch', val_losses.avg, epoch)
        writer.add_scalar('val/msr_loss_epoch', val_msr_losses.avg, epoch)

        # N.B. schedulers are usually called at the end of the epoch, except for cosine-warmup
        if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'ExponentialLR':
            scheduler.step()

        # Log LR
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{config["optimizer_type"]}', current_lr, epoch)

        # Print losses
        print(f"Epoch {epoch + 1}/{config['max_training_epochs']} - Train Loss {train_losses.avg:.4f}, Val Loss {val_losses.avg:.4f}")

        # Update best model (based on total validation loss)
        if val_losses.avg < best_val_loss:
            save_status(None, epoch, model_name, save_name, model, optimizer, scheduler, val_losses, checkpoint_path, config)
            best_val_loss = val_losses.avg

        # Early stopping
        if config['es_enable']:
            early_stopping(val_losses.avg)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # Load the best model after pretraining
    model = get_model_architecture(config)
    load_status(None, model_name, save_name, model, optimizer, scheduler, checkpoint_path, config)
    model = model.to(device)
    

    #### LINEAR PROBING PHASE ####
    # In case of self-supervised pretraining, we need to train the final linear regression model
    # on top of it to test the learned representation and proceed with the test phase.
    # This part uses the full labeled dataset and not the ssl one.

    print("\n--- Starting Linear Probing Phase ---")

    # Re-initialize dataset for supervised tasks (ensures correct `__getitem__` behavior)
    lp_dataset = PhysioDataset(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['lp_dataset_name']), # Linear probing dataset name
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
            ppg_freqs=config['ppg_freqs'],
            apply_masking=False # Masking is not applied during linear probing
        )

    # Get train/val/test samplers and build the dataloaders for linear probing
    (lp_train_sampler, lp_val_sampler, lp_test_sampler) = lp_dataset.get_pretraining_samplers()

    lp_train_dataloader = DataLoader(lp_dataset, sampler=lp_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    lp_val_dataloader = DataLoader(lp_dataset, sampler=lp_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    lp_test_dataloader = DataLoader(lp_dataset, sampler=lp_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)

    # Freeze encoder, unfreeze last linear layer
    set_trainable_parameters(model=model, tune='decoder', config=config)

    # Optimizer and scheduler for linear probing
    if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
        config['steps_per_epoch'] = len(lp_train_dataloader) # Update steps_per_epoch for LP
    lp_optim_sched = configure_optimizer_and_scheduler(model, config)
    lp_optimizer = lp_optim_sched['optimizer']
    lp_scheduler = lp_optim_sched['lr_scheduler'] if config['lr_scheduler_enable'] else None

    lp_best_val_loss = float("+inf")

    # Early stopping for linear probing
    if config['es_enable']:
        lp_early_stopping = EarlyStopping(
            patience=config['es_patience'],
            delta=config['es_min_delta'],
            verbose=True,
            mode='min'
            )

    # Linear probing loop
    for epoch in range(config['max_lp_training_epochs']): # Use a separate LP epoch count if desired
        
        # Training loop
        model.train() # Set model to training mode for LP
        set_trainable_parameters(model=model, tune='decoder', config=config) # Only train last linear layer
        train_lp_losses = AverageMeter(name='lp_train/loss')
        for batch_idx, batch in enumerate(lp_train_dataloader):
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)

            # Targets are in the supervised format
            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

            lp_optimizer.zero_grad()

            # Perform forward pass (encoder frozen, only last layer trained)
            outputs = model(signals)

            # Supervised loss for LP
            if config['criterion'] == 'MSELoss':
                loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion ...")

            loss.backward()
            lp_optimizer.step()

            if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
                lp_scheduler.step()

            train_lp_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('train/lp_loss', loss.item(), epoch * len(lp_train_dataloader) + batch_idx)

        writer.add_scalar('train/lp_loss_epoch', train_lp_losses.avg, epoch)

        # Validation loop for LP
        model.eval() # Set model to evaluation mode (siwtch off batch norm/dropout etc.)
        set_trainable_parameters(model=model, tune='none', config=config) # Freeze all params during validation
        
        lp_val_losses = AverageMeter(name='lp_val/loss')
        lp_val_sbp_maes = AverageMeter(name='lp_val/sbp_mae')
        lp_val_dbp_maes = AverageMeter(name='lp_val/dbp_mae')
        lp_val_sbp_mes = AverageMeter(name='lp_val/sbp_me')
        lp_val_dbp_mes = AverageMeter(name='lp_val/dbp_me')
        lp_val_sbp_mae_stds = AverageMeter(name='lp_val/sbp_mae_std')
        lp_val_dbp_mae_stds = AverageMeter(name='lp_val/dbp_mae_std')
        lp_val_sbp_me_stds = AverageMeter(name='lp_val/sbp_me_std')
        lp_val_dbp_me_stds = AverageMeter(name='lp_val/dbp_me_std')

        with torch.no_grad():
            for batch_idx, batch in enumerate(lp_val_dataloader):
                signals, targets = batch
                signals = signals.to(device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)

                if config['sig2sig']:
                    targets = targets.to(device)
                else:
                    targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

                outputs = model(signals)

                if config['criterion'] == 'MSELoss':
                    val_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    val_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")

                metric_values = get_metric_values(val_loss, outputs, targets, config)
                
                lp_val_losses.update(metric_values['loss'], signals.size(0))
                lp_val_sbp_maes.update(metric_values['sbp_mae'], signals.size(0))
                lp_val_dbp_maes.update(metric_values['dbp_mae'], signals.size(0))
                lp_val_sbp_mes.update(metric_values['sbp_me'], signals.size(0))
                lp_val_dbp_mes.update(metric_values['dbp_me'], signals.size(0))
                lp_val_sbp_mae_stds.update(metric_values['sbp_mae_std'], signals.size(0))
                lp_val_dbp_mae_stds.update(metric_values['dbp_mae_std'], signals.size(0))
                lp_val_sbp_me_stds.update(metric_values['sbp_me_std'], signals.size(0))
                lp_val_dbp_me_stds.update(metric_values['dbp_me_std'], signals.size(0))

        # Log epoch validation loss and metrics (at the end of validation)
        writer.add_scalar('lp_val/loss_epoch', lp_val_losses.avg, epoch)
        writer.add_scalar('lp_val/sbp_mae_epoch', lp_val_sbp_maes.avg, epoch)
        writer.add_scalar('lp_val/dbp_mae_epoch', lp_val_dbp_maes.avg, epoch)
        writer.add_scalar('lp_val/sbp_me_epoch', lp_val_sbp_mes.avg, epoch)
        writer.add_scalar('lp_val/dbp_me_epoch', lp_val_dbp_mes.avg, epoch)
        writer.add_scalar('lp_val/sbp_mae_std_epoch', lp_val_sbp_mae_stds.avg, epoch)
        writer.add_scalar('lp_val/dbp_mae_std_epoch', lp_val_dbp_mae_stds.avg, epoch)

        if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'ExponentialLR':
            if lp_scheduler: # Only step if scheduler exists
                lp_scheduler.step()

        current_lp_lr = lp_optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{config["optimizer_type"]}_lp_lr', current_lp_lr, epoch)

        print(f"LP Epoch {epoch + 1}/{config['max_training_epochs']} - Train Loss {train_lp_losses.avg:.4f}, Val Loss {lp_val_losses.avg:.4f}")

        if lp_val_losses.avg < lp_best_val_loss:
            save_status(None, epoch, model_name + "_lp", save_name, model, lp_optimizer, lp_scheduler, lp_val_losses, checkpoint_path, config)
            lp_best_val_loss = lp_val_losses.avg

        if config['es_enable']:
            lp_early_stopping(lp_val_losses.avg)
            if lp_early_stopping.early_stop:
                print("LP Early stopping")
                break

    ## FINAL TEST PHASE for Linear Probing and Self-Supervision
    # Load the best LP model
    model = get_model_architecture(config) # Re-instantiate to load best LP checkpoint
    load_status(None, model_name + "_lp", save_name, model, lp_optimizer, lp_scheduler, checkpoint_path, config)
    model = model.to(device)
    model.eval() # Set model to evaluation mode (siwtch off batch norm/dropout etc.))
    set_trainable_parameters(model=model, tune='none', config=config) # Freeze all params during validation

    # Meters    
    lp_test_losses = AverageMeter(name='lp_test/loss')
    lp_test_sbp_maes = AverageMeter(name='lp_test/sbp_mae')
    lp_test_dbp_maes = AverageMeter(name='lp_test/dbp_mae')
    lp_test_sbp_mes = AverageMeter(name='lp_test/sbp_me')
    lp_test_dbp_mes = AverageMeter(name='lp_test/dbp_me')
    lp_test_sbp_mae_stds = AverageMeter(name='lp_test/sbp_mae_std')
    lp_test_dbp_mae_stds = AverageMeter(name='lp_test/dbp_mae_std')
    lp_test_sbp_me_stds = AverageMeter(name='lp_test/sbp_me_std')
    lp_test_dbp_me_stds = AverageMeter(name='lp_test/dbp_me_std')
    
    if config['sig2sig']:
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
    else:
        all_test_outputs = np.empty((0, 2), dtype=float)
        all_test_targets = np.empty((0, 2), dtype=float)

    with torch.no_grad():
        for batch_idx, batch in enumerate(lp_test_dataloader):
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)

            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

            outputs = model(signals)

            if config['criterion'] == 'MSELoss':
                test_loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                test_loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion ...")

            all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
            all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)

            metric_values = get_metric_values(test_loss, outputs, targets, config)
            
            lp_test_losses.update(metric_values['loss'], signals.size(0))
            lp_test_sbp_maes.update(metric_values['sbp_mae'], signals.size(0))
            lp_test_dbp_maes.update(metric_values['dbp_mae'], signals.size(0))
            lp_test_sbp_mes.update(metric_values['sbp_me'], signals.size(0))
            lp_test_dbp_mes.update(metric_values['dbp_me'], signals.size(0))
            lp_test_sbp_mae_stds.update(metric_values['sbp_mae_std'], signals.size(0))
            lp_test_dbp_mae_stds.update(metric_values['dbp_mae_std'], signals.size(0))
            lp_test_sbp_me_stds.update(metric_values['sbp_me_std'], signals.size(0))
            lp_test_dbp_me_stds.update(metric_values['dbp_me_std'], signals.size(0))

    # Log test metrics
    # Note: `epoch` here is the last epoch of training, not ideal for test summary
    writer.add_scalar('lp_test/loss_final', lp_test_losses.avg, epoch)
    writer.add_scalar('lp_test/sbp_mae_final', lp_test_sbp_maes.avg, epoch)
    writer.add_scalar('lp_test/dbp_mae_final', lp_test_dbp_maes.avg, epoch)
    writer.add_scalar('lp_test/sbp_me_final', lp_test_sbp_mes.avg, epoch)
    writer.add_scalar('lp_test/dbp_me_final', lp_test_dbp_mes.avg, epoch)
    writer.add_scalar('lp_test/sbp_mae_std_final', lp_test_sbp_mae_stds.avg, epoch)
    writer.add_scalar('lp_test/dbp_mae_std_final', lp_test_dbp_mae_stds.avg, epoch)

    writer.close()

    return all_test_targets, all_test_outputs
    
    