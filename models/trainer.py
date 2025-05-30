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
from data.dataset import PhysioDataset
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

    if not config['ssl']: # Original supervised path
        return supervised_pretraining_training_validation_testing(
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
    else: # New self-supervised path
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


def supervised_pretraining_training_validation_testing(
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

    # Best lowest val loss
    best_val = float("+inf")

    # Supervised pretraining loop
    for epoch in range(config['max_training_epochs']):

        # ---- Training Loop ----
        set_trainable_parameters(model=model, tune='all', config=config)
        train_losses = AverageMeter(name='train/loss')
        for batch_idx, batch in enumerate(train_dataloader):

            # Prepare data
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2: # For 1D signals like pure PPG, expand to [Batch, Length, 1]
                signals = signals.unsqueeze(-1)
            
            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

            # Forward propagation
            # In supervised mode, model.forward typically returns outputs and optionally embeddings
            outputs = model(signals)

            # Supervised loss
            supervised_loss = 0.
            if config['lambda_supervised'] != 0.:
                if config['criterion'] == 'MSELoss':
                    supervised_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    supervised_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")

            # Total loss
            loss = config['lambda_supervised'] * supervised_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch only for cosine-warmup scheduler
            if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
                scheduler.step()

            # Log training loss
            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

        # Log epoch training loss
        writer.add_scalar('train/loss_epoch', train_losses.avg, epoch)

        # ---- Validation Loop ----
        model.eval() # Set model to evaluation mode (siwtch off batch norm/dropout etc.) for validation
        set_trainable_parameters(model=model, tune='none', config=config) # Ensure all params are non-trainable during eval
        
        val_losses = AverageMeter(name='val/loss')
        val_sbp_maes = AverageMeter(name='val/sbp_mae')
        val_dbp_maes = AverageMeter(name='val/dbp_mae')
        val_sbp_mes = AverageMeter(name='val/sbp_me')
        val_dbp_mes = AverageMeter(name='val/dbp_me')
        val_sbp_mae_stds = AverageMeter(name='val/sbp_mae_std')
        val_dbp_mae_stds = AverageMeter(name='val/dbp_mae_std')
        val_sbp_me_stds = AverageMeter(name='val/sbp_me_std')
        val_dbp_me_stds = AverageMeter(name='val/dbp_me_std')
        
        for batch_idx, batch in enumerate(val_dataloader):

            # Prepare data
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            #_, _, num_modalities = signals.shape

            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

            # Forward propagation
            with torch.no_grad(): # No gradient calculation during validation
                outputs = model(signals)

            # Supervised loss
            supervised_loss = 0
            if config['lambda_supervised'] != 0.:
                if config['criterion'] == 'MSELoss':
                    supervised_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    supervised_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")

            # Total loss for validation is just supervised loss
            val_loss = config['lambda_supervised'] * supervised_loss

            # Log validation loss and metrics
            metric_values = get_metric_values(val_loss, outputs, targets, config)
            
            val_losses.update(metric_values['loss'], signals.size(0))
            val_sbp_maes.update(metric_values['sbp_mae'], signals.size(0))
            val_dbp_maes.update(metric_values['dbp_mae'], signals.size(0))
            val_sbp_mes.update(metric_values['sbp_me'], signals.size(0))
            val_dbp_mes.update(metric_values['dbp_me'], signals.size(0))
            val_sbp_mae_stds.update(metric_values['sbp_mae_std'], signals.size(0))
            val_dbp_mae_stds.update(metric_values['dbp_mae_std'], signals.size(0))
            val_sbp_me_stds.update(metric_values['sbp_me_std'], signals.size(0))
            val_dbp_me_stds.update(metric_values['dbp_me_std'], signals.size(0))

        # Log epoch validation loss and metrics (at the end of validation)
        writer.add_scalar('val/loss_epoch', val_losses.avg, epoch)
        writer.add_scalar('val/sbp_mae_epoch', val_sbp_maes.avg, epoch)
        writer.add_scalar('val/dbp_mae_epoch', val_dbp_maes.avg, epoch)
        writer.add_scalar('val/sbp_me_epoch', val_sbp_mes.avg, epoch)
        writer.add_scalar('val/dbp_me_epoch', val_dbp_mes.avg, epoch)
        writer.add_scalar('val/sbp_mae_std_epoch', val_sbp_mae_stds.avg, epoch)
        writer.add_scalar('val/dbp_mae_std_epoch', val_dbp_mae_stds.avg, epoch)

        # N.B. schedulers are usually called at the end of the epoch except for ths cosine-warmup
        if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'ExponentialLR':
            scheduler.step()

        # Log LR
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{config["optimizer_type"]}', current_lr, epoch)

        # Print losses
        print(f"Epoch {epoch + 1}/{config['max_training_epochs']} - Train Loss {train_losses.avg:.4f}, Val Loss {val_losses.avg:.4f}")

        # Update best model
        if val_losses.avg < best_val:
            save_status(None, epoch, model_name, save_name, model, optimizer, scheduler, val_losses, checkpoint_path, config)
            best_val = val_losses.avg

        # Early stopping
        if config['es_enable']:
            early_stopping(val_losses.avg)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # Load the best model
    model = get_model_architecture(config)
    load_status(None, model_name, save_name, model, optimizer, scheduler, checkpoint_path, config)
    model = model.to(device)
    model.eval() # Set model to evaluation mode (siwtch off batch norm/dropout etc.) for testing
    set_trainable_parameters(model=model, tune='none', config=config)

    # ---- Test Loop ----
    
    # Meters    
    test_losses = AverageMeter(name='test/loss')
    test_sbp_maes = AverageMeter(name='test/sbp_mae')
    test_dbp_maes = AverageMeter(name='test/dbp_mae')
    test_sbp_mes = AverageMeter(name='test/sbp_me')
    test_dbp_mes = AverageMeter(name='test/dbp_me')
    test_sbp_mae_stds = AverageMeter(name='test/sbp_mae_std')
    test_dbp_mae_stds = AverageMeter(name='test/dbp_mae_std')
    test_sbp_me_stds = AverageMeter(name='test/sbp_me_std')
    test_dbp_me_stds = AverageMeter(name='test/dbp_me_std')
    
    # Record outputs and targets
    if config['sig2sig']:
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
    else:
        all_test_outputs = np.empty((0, 2), dtype=float) # Assuming SBP/DBP output shape is 2
        all_test_targets = np.empty((0, 2), dtype=float)
        
    
    for batch_idx, batch in enumerate(test_dataloader):

        # Prepare data
        signals, targets = batch
        signals = signals.to(device)
        if len(signals.shape) == 2:
            signals = signals.unsqueeze(-1)

        if config['sig2sig']:
            targets = targets.to(device)
        else:
            targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

        # Perform forward pass
        with torch.no_grad(): # No gradient calculation during testing
            outputs = model(signals)

        # Supervised loss (for metric logging)
        if config['criterion'] == 'MSELoss':
            test_loss = F.mse_loss(outputs, targets)
        elif config['criterion'] == 'SmoothL1Loss':
            test_loss = F.smooth_l1_loss(outputs, targets)
        else:
            raise ValueError("Invalid criterion ...")

        # Record predictions and ground truths
        all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
        all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)

        # Log metrics
        metric_values = get_metric_values(test_loss, outputs, targets, config)
        
        test_losses.update(metric_values['loss'], signals.size(0))
        test_sbp_maes.update(metric_values['sbp_mae'], signals.size(0))
        test_dbp_maes.update(metric_values['dbp_mae'], signals.size(0))
        test_sbp_mes.update(metric_values['sbp_me'], signals.size(0))
        test_dbp_mes.update(metric_values['dbp_me'], signals.size(0))
        test_sbp_mae_stds.update(metric_values['sbp_mae_std'], signals.size(0))
        test_dbp_mae_stds.update(metric_values['dbp_mae_std'], signals.size(0))
        test_sbp_me_stds.update(metric_values['sbp_me_std'], signals.size(0))
        test_dbp_me_stds.update(metric_values['dbp_me_std'], signals.size(0))


    # Log test metrics
    # Note: `epoch` here is the last epoch of training, not ideal for test summary
    writer.add_scalar('test/loss_final', test_losses.avg, epoch)
    writer.add_scalar('test/sbp_mae_final', test_sbp_maes.avg, epoch)
    writer.add_scalar('test/dbp_mae_final', test_dbp_maes.avg, epoch)
    writer.add_scalar('test/sbp_me_final', test_sbp_mes.avg, epoch)
    writer.add_scalar('test/dbp_me_final', test_dbp_mes.avg, epoch)
    writer.add_scalar('test/sbp_mae_std_final', test_sbp_mae_stds.avg, epoch)
    writer.add_scalar('test/dbp_mae_std_final', test_dbp_mae_stds.avg, epoch)
    
    writer.close()

    return all_test_targets, all_test_outputs


def supcon_loss(features, config, labels=None, mask=None):
    r"""
    Supervised Contrastive Loss
    from https://github.com/pulp-platform/fscil/blob/main/code/lib/torch_blocks.py#L114
    abd from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#SimCLR-implementation
    """
    device = features.device

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                        'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # Shape: [batch_size * n_views, embedding_dim]

    # Normalize embeddings
    contrast_feature = F.normalize(contrast_feature, dim=1)

    # Compute cosine similarity
    cos_sim = torch.matmul(contrast_feature, contrast_feature.T)  # Shape: [batch_size * n_views, batch_size * n_views]

    # Mask out self-contrast cases
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Create positive pair mask
    if labels is not None:
        # Supervised case: Use labels to create positive pairs
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        labels = torch.cat([labels] * contrast_count, dim=0)  # Repeat labels for all views
        pos_mask = torch.eq(labels, labels.T).float().to(device)
    else:
        # Unsupervised case: Assume positive pairs are `batch_size // 2` apart
        # This is for unsupervised SimCLR where (x_i, x_j) from same x are positive pairs
        # The structure is [x1_aug1, x1_aug2, x2_aug1, x2_aug2, ...]
        pos_mask = self_mask.roll(shifts=1, dims=0).float() # (i, i+1) and (i+1, i) for each pair
        pos_mask = pos_mask + self_mask.roll(shifts=-1, dims=0).float()
        pos_mask = pos_mask.fill_diagonal_(0) # Remove self-comparison after roll

    # Scale cosine similarity by temperature
    cos_sim = cos_sim / config['temperature']

    # Compute InfoNCE loss
    # log_prob = cos_sim - torch.logsumexp(cos_sim, dim=-1, keepdim=True)
    # Corrected for stability and typical InfoNCE formulation:
    logits = cos_sim
    labels_simclr = pos_mask.float() # positive mask as labels
    
    # Calculate log_probs manually
    exp_logits = torch.exp(logits) * (1 - self_mask.float()) # Exclude self-comparison from denominator
    log_prob = logits - torch.log(exp_logits.sum(dim=-1, keepdim=True))

    mean_log_prob_pos = (labels_simclr * log_prob).sum(1) / labels_simclr.sum(1)

    # Handle cases where labels_simclr.sum(1) == 0 (no positive pairs, should not happen in SimCLR batch)
    mean_log_prob_pos[labels_simclr.sum(1) == 0] = 0

    # Final loss
    loss = -mean_log_prob_pos.mean()

    # Return loss and metrics
    return loss, cos_sim, pos_mask


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
        train_cwg_losses = AverageMeter(name='train/cwg_loss')
        train_contrastive_losses = AverageMeter(name='train/contrastive_loss')
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            # Move data to device
            current_signal = batch['current_signal'].to(device) # [B, L, 2]
            masked_signal = batch['masked_signal'].to(device)   # [B, L, 2]
            mask_indices = batch['mask_indices'].to(device)     # [B, L] bool
            aug1_signal = batch['aug1_signal'].to(device)       # [B, L, 2]
            aug2_signal = batch['aug2_signal'].to(device)       # [B, L, 2]
            prev_signal = batch['prev_signal'] # [B, L, 2] 
            next_signal = batch['next_signal'] # [B, L, 2] 

            prev_signal = prev_signal.to(device)
            next_signal = next_signal.to(device)

            optimizer.zero_grad() # Zero gradients for a new batch

            # --- Forward Passes and Loss Calculations for Each Task ---

            # 1. Masked Signal Reconstruction (MSR)
            msr_loss = torch.tensor(0.0, device=device)
            if config['lambda_msr'] != 0.:
                # Encoder needs to handle the `masked_signal` as input
                # Model's `forward` or specific `forward_reconstruction` method
                # This depends on how your `model` is structured (e.g., `model(masked_signal, task='msr')`)
                # Assuming model has a `reconstruct` method
                masked_embedding = model.encode(masked_signal) # Assume encode is shared
                reconstructed_signal = model.reconstruct(masked_embedding)

                # Calculate loss only on masked positions
                # Ensure mask_indices is expanded to match signal dimensions if needed: [B, L] -> [B, L, Modalities]
                if config['criterion'] == 'MSELoss':
                    msr_loss = F.mse_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], current_signal[mask_indices.unsqueeze(-1).expand_as(current_signal)])
                elif config['criterion'] == 'SmoothL1Loss':
                    msr_loss = F.smooth_l1_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], current_signal[mask_indices.unsqueeze(-1).expand_as(current_signal)])
                else:
                    raise ValueError("Invalid criterion ...")

            # 2. Cross-Window Generation (CWG)
            cwg_loss = torch.tensor(0.0, device=device)
            if config['lambda_cwg'] != 0.:
                # Only compute CWG if next_signal is available for this batch
                current_embedding_for_cwg = model.encode(current_signal)
                generated_prev_signal = model.generate_prev_window(current_embedding_for_cwg)
                generated_next_signal = model.generate_next_window(current_embedding_for_cwg)

                # Calculate loss between generated and actual next signal
                if config['criterion'] == 'MSELoss':
                    cwg_loss = F.mse_loss(generated_prev_signal, prev_signal) + F.mse_loss(generated_next_signal, next_signal)
                elif config['criterion'] == 'SmoothL1Loss':
                    cwg_loss = F.smooth_l1_loss(generated_prev_signal, prev_signal) + F.smooth_l1_loss(generated_next_signal, next_signal)
                else:
                    raise ValueError("Invalid criterion ...")                
                
            # 3. Contrastive Learning (SimCLR)
            contrastive_loss = torch.tensor(0.0, device=device)
            if config['lambda_contrastive'] != 0. and config['temperature'] != 0.:
                
                # Get embeddings for augmented views
                aug1_embedding = model.encode(aug1_signal)
                aug2_embedding = model.encode(aug2_signal)

                # Pass through projection head
                z1 = model.project(aug1_embedding)
                z2 = model.project(aug2_embedding)

                # Stack for supcon_loss: [Batch_Size, N_Views, Embedding_Dim]
                features_for_supcon = torch.stack([z1, z2], dim=1) # n_views = 2

                contrastive_loss, _, _ = supcon_loss(features_for_supcon, config)


            # --- Total Loss ---
            loss = (config.get('lambda_msr', 0.) * msr_loss +
                    config.get('lambda_cwg', 0.) * cwg_loss +
                    config.get('lambda_contrastive', 0.) * contrastive_loss)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch only for cosine-warmup scheduler
            if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
                scheduler.step()

            # Log training losses
            train_losses.update(loss.item(), current_signal.size(0))
            train_msr_losses.update(msr_loss.item(), current_signal.size(0))
            train_cwg_losses.update(cwg_loss.item(), current_signal.size(0))
            train_contrastive_losses.update(contrastive_loss.item(), current_signal.size(0))
            
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('train/msr_loss', msr_loss.item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('train/cwg_loss', cwg_loss.item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('train/contrastive_loss', contrastive_loss.item(), epoch * len(train_dataloader) + batch_idx)

        # Log epoch average training losses
        writer.add_scalar('train/loss_epoch', train_losses.avg, epoch)
        writer.add_scalar('train/msr_loss_epoch', train_msr_losses.avg, epoch)
        writer.add_scalar('train/cwg_loss_epoch', train_cwg_losses.avg, epoch)
        writer.add_scalar('train/contrastive_loss_epoch', train_contrastive_losses.avg, epoch)

        # Validation loop
        model.eval() # Set model to evaluation mode (siwtch off batch norm/dropout etc.)
        set_trainable_parameters(model=model, tune='none', config=config) # Ensure all params are non-trainable during eval

        val_losses = AverageMeter(name='val/loss')
        val_msr_losses = AverageMeter(name='val/msr_loss')
        val_cwg_losses = AverageMeter(name='val/cwg_loss')
        val_contrastive_losses = AverageMeter(name='val/contrastive_loss')

        with torch.no_grad(): # No gradient calculation during validation
            for batch_idx, batch in enumerate(val_dataloader):
                current_signal = batch['current_signal'].to(device)
                masked_signal = batch['masked_signal'].to(device)
                mask_indices = batch['mask_indices'].to(device)
                aug1_signal = batch['aug1_signal'].to(device)
                aug2_signal = batch['aug2_signal'].to(device)
                prev_signal = batch['prev_signal'] # [B, L, 2] or None
                next_signal = batch['next_signal'] # [B, L, 2] or None

                prev_signal = prev_signal.to(device)
                next_signal = next_signal.to(device)

                # --- Forward Passes and Loss Calculations for Each Task (Validation) ---
                val_msr_loss = torch.tensor(0.0, device=device)
                if config['lambda_msr'] != 0.:
                    masked_embedding = model.encode(masked_signal)
                    reconstructed_signal = model.reconstruct(masked_embedding)
                    
                    if config['criterion'] == 'MSELoss':
                        val_msr_loss = F.mse_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], current_signal[mask_indices.unsqueeze(-1).expand_as(current_signal)])
                    elif config['criterion'] == 'SmoothL1Loss':
                        val_msr_loss = F.smooth_l1_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], current_signal[mask_indices.unsqueeze(-1).expand_as(current_signal)])
                    else:
                        raise ValueError("Invalid criterion ...")
                    
                val_cwg_loss = torch.tensor(0.0, device=device)
                if config['lambda_cwg'] != 0. and next_signal is not None:
                    current_embedding_for_cwg = model.encode(current_signal)
                    generated_prev_signal = model.generate_prev_window(current_embedding_for_cwg)
                    generated_next_signal = model.generate_next_window(current_embedding_for_cwg)
                    
                    if config['criterion'] == 'MSELoss':
                        val_cwg_loss = F.mse_loss(generated_prev_signal, prev_signal) + F.mse_loss(generated_next_signal, next_signal) 
                    elif config['criterion'] == 'SmoothL1Loss':
                        val_cwg_loss = F.smooth_l1_loss(generated_prev_signal, prev_signal) + F.smooth_l1_loss(generated_next_signal, next_signal) 
                    else:
                        raise ValueError("Invalid criterion ...")

                val_contrastive_loss = torch.tensor(0.0, device=device)
                if config['lambda_contrastive'] != 0. and config['temperature'] != 0.:
                    aug1_embedding = model.encode(aug1_signal)
                    aug2_embedding = model.encode(aug2_signal)
                    z1 = model.project(aug1_embedding)
                    z2 = model.project(aug2_embedding)
                    features_for_supcon = torch.stack([z1, z2], dim=1)
                    val_contrastive_loss, _, _ = supcon_loss(features_for_supcon, config)


                # Total validation loss
                val_loss = (config.get('lambda_msr', 0.) * val_msr_loss +
                            config.get('lambda_cwg', 0.) * val_cwg_loss +
                            config.get('lambda_contrastive', 0.) * val_contrastive_loss)


                # Update validation meters
                val_losses.update(val_loss.item(), current_signal.size(0))
                val_msr_losses.update(val_msr_loss.item(), current_signal.size(0))
                val_cwg_losses.update(val_cwg_loss.item(), current_signal.size(0))
                val_contrastive_losses.update(val_contrastive_loss.item(), current_signal.size(0))

        # Log epoch validation losses to TensorBoard
        writer.add_scalar('val/loss_epoch', val_losses.avg, epoch)
        writer.add_scalar('val/msr_loss_epoch', val_msr_losses.avg, epoch)
        writer.add_scalar('val/cwg_loss_epoch', val_cwg_losses.avg, epoch)
        writer.add_scalar('val/contrastive_loss_epoch', val_contrastive_losses.avg, epoch)

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
            ppg_freqs=config['ppg_freqs']
        )

    # Get train/val/test samplers and build the dataloaders for linear probing
    (lp_train_sampler, lp_val_sampler, lp_test_sampler) = lp_dataset.get_pretraining_samplers()

    lp_train_dataloader = DataLoader(lp_dataset, sampler=lp_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    lp_val_dataloader = DataLoader(lp_dataset, sampler=lp_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    lp_test_dataloader = DataLoader(lp_dataset, sampler=lp_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)

    # Freeze encoder, unfreeze last linear layer
    set_trainable_parameters(model=model, tune='last_linear', config=config)

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
        set_trainable_parameters(model=model, tune='last_linear', config=config) # Only train last linear layer
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
    
    