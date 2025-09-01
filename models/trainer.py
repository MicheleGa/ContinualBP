import os
import sys
folders_to_add = ['data', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Function
from data.dataset import PhysioDataset
from training_utils.helpers import save_status, load_status, EarlyStopping, set_trainable_parameters, configure_optimizer_and_scheduler, get_model_architecture
from training_utils.metrics import AverageMeter, get_metric_values, call_metric



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

    if not config['ssl']:
        if not config['meta_learning']:
            if config['model_name'] == 'BIOT' and config['pretrained_path'] is not None:
                supervised_finetune_with_pretrained_backbone(
                    save_name=save_name,
                    checkpoint_path=checkpoint_path,
                    writer=writer,
                    model_name=model_name,
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    early_stopping=early_stopping,
                    optimizer=optimizer,  # not used; we create our own optimizers inside
                    scheduler=scheduler,  # not used; created per stage below if requested
                    config=config,
                    device=device
                )
            else:
                supervised_pretraining_training_validation_testing(
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
        else:
            if config['meta_algorithm'] == 'maml':
                maml_meta_training(
                    save_name,
                    checkpoint_path,
                    writer,
                    model_name,
                    model,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader,
                    early_stopping,
                    config,
                    device
                )
            elif config['meta_algorithm'] == 'mann':
                # call Reptile meta-training (this returns the final model after meta-training)
                mann_meta_training(
                    save_name,
                    checkpoint_path,
                    writer,
                    model_name,
                    model,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader,
                    early_stopping,
                    config,
                    device
                )
            elif config['meta_algorithm'] == 'reptile':
                reptile_meta_training(
                    save_name=save_name,
                    checkpoint_path=checkpoint_path,
                    writer=writer,
                    model_name=model_name,
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    early_stopping=early_stopping,
                    config=config,
                    device=device
                )
            else:
                raise ValueError('Unsupported Meta-Learning algorithm')
    else: 
        self_supervised_pretraining_training_validation_testing(
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


def supervised_finetune_with_pretrained_backbone(
    save_name,
    checkpoint_path,
    writer,
    model_name,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    early_stopping,
    optimizer,            # unused, new optimizers per stage
    scheduler,            # unused, new schedulers per stage
    config,
    device
):
    """
    Two-stage supervised fine-tuning with:
      - Stage 1: head-only training (backbone frozen)
      - Stage 2: backbone + head with param-group LRs
      - Cosine LR schedule per stage with linear warmup
      - Original trainable mask to avoid unfreezing fixed params (e.g., encoder.index)
    """

    # ====== Store original trainable mask ======
    original_trainable = {n: p.requires_grad for n, p in model.named_parameters()}

    def set_requires_grad_safe(module, req, name_prefix=""):
        """Toggle requires_grad but respect original_trainable mask when unfreezing."""
        for n, p in module.named_parameters():
            full_name = f"{name_prefix}.{n}" if name_prefix else n
            if not req:  # freezing
                p.requires_grad = False
            else:        # unfreezing
                if original_trainable.get(full_name, True):
                    p.requires_grad = True

    def get_backbone_and_head_params(model):
        backbone_params, head_params = [], []
        for name, p in model.named_parameters():
            if name.startswith("encoder"):
                backbone_params.append(p)
            else:
                head_params.append(p)
        return backbone_params, head_params

    def linear_warmup(current_epoch, warmup_epochs, base_lr):
        if current_epoch >= warmup_epochs:
            return base_lr
        return base_lr * (0.1 + 0.9 * (current_epoch / warmup_epochs))

    # ====== Common config ======
    base_lr = config.get('base_lr', 3e-4)
    backbone_lr_mul = config.get('backbone_lr_multiplier', 0.05)
    weight_decay = config.get('weight_decay', 1e-2)
    grad_clip = config.get('grad_clip', 1.0)

    stage1_epochs = config.get('ft_stage1_epochs', 10)
    stage2_epochs = config.get('ft_stage2_epochs', 50)
    warmup_stage1 = config.get('warmup_epochs_stage1', 3)
    warmup_stage2 = config.get('warmup_epochs_stage2', 5)

    best_val = float("+inf")

    # ====== Stage 1: Head-only ======
    if config.get('freeze_backbone_first', True):
        if hasattr(model, 'encoder'):
            set_requires_grad_safe(model.encoder, False, name_prefix="encoder")
    if hasattr(model, 'channel_proj'):
        set_requires_grad_safe(model.channel_proj, True, name_prefix="channel_proj")
    if hasattr(model, 'decoder'):
        set_requires_grad_safe(model.decoder, True, name_prefix="decoder")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_stage1 = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=stage1_epochs)

    print(f"==== Stage 1: head-only, {stage1_epochs} epochs, base_lr={base_lr} ====")
    for epoch in range(stage1_epochs):
        model.train()
        # Warmup LR
        for pg in optimizer_stage1.param_groups:
            pg['lr'] = linear_warmup(epoch, warmup_stage1, base_lr)

        train_losses = AverageMeter(name='train/loss')
        for batch_idx, batch in enumerate(train_dataloader):
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

            optimizer_stage1.zero_grad()
            outputs = model(signals)

            if config['criterion'] == 'MSELoss':
                loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion")

            loss *= config.get('lambda_supervised', 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer_stage1.step()

            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('train/stage1_loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

        scheduler_stage1.step()

        # Validation
        model.eval()
        val_losses = AverageMeter(name='val/loss')
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                signals, targets = batch
                signals = signals.to(device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
                
                outputs = model(signals)
                
                vloss = F.mse_loss(outputs, targets) if config['criterion'] == 'MSELoss' else F.smooth_l1_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('val/stage1_loss_epoch', val_losses.avg, epoch)
        print(f"[Stage1] Epoch {epoch+1}/{stage1_epochs} - Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            save_status(None, epoch, model_name + "_ft_stage1", save_name, model, optimizer_stage1, scheduler_stage1, val_losses, checkpoint_path, config)
            best_val = val_losses.avg

        if config['es_enable']:
            early_stopping(val_losses.avg)
            if early_stopping.early_stop:
                print("Early stopping Stage 1")
                break

    # ====== Stage 2: Backbone + Head ======
    lr_backbone = base_lr * backbone_lr_mul
    set_requires_grad_safe(model.encoder, True, name_prefix="encoder")
    set_requires_grad_safe(model.channel_proj, True, name_prefix="channel_proj")
    set_requires_grad_safe(model.decoder, True, name_prefix="decoder")

    backbone_params, head_params = get_backbone_and_head_params(model)
    backbone_params = [p for p in backbone_params if p.requires_grad]
    head_params = [p for p in head_params if p.requires_grad]

    optimizer_stage2 = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': base_lr}
    ], weight_decay=weight_decay)
    scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=stage2_epochs)

    print(f"==== Stage 2: fine-tune backbone, {stage2_epochs} epochs, head_lr={base_lr}, backbone_lr={lr_backbone} ====")
    for epoch in range(stage2_epochs):
        model.train()
        # Warmup
        for i, pg in enumerate(optimizer_stage2.param_groups):
            target_lr = base_lr if i == 1 else lr_backbone
            pg['lr'] = linear_warmup(epoch, warmup_stage2, target_lr)

        train_losses = AverageMeter(name='train/loss')
        for batch_idx, batch in enumerate(train_dataloader):
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)

            optimizer_stage2.zero_grad()
            outputs = model(signals)
            
            if config['criterion'] == 'MSELoss':
                loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion")

            loss *= config.get('lambda_supervised', 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer_stage2.step()

            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('train/stage2_loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

        scheduler_stage2.step()

        # Validation
        model.eval()
        val_losses = AverageMeter(name='val/loss')
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                signals, targets = batch
                signals = signals.to(device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
                
                outputs = model(signals)
                
                vloss = F.mse_loss(outputs, targets) if config['criterion'] == 'MSELoss' else F.smooth_l1_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('val/stage2_loss_epoch', val_losses.avg, epoch)
        print(f"[Stage2] Epoch {epoch+1}/{stage2_epochs} - Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            save_status(None, epoch, model_name + "_ft_stage2", save_name, model, optimizer_stage2, scheduler_stage2, val_losses, checkpoint_path, config)
            best_val = val_losses.avg

        if config['es_enable']:
            early_stopping(val_losses.avg)
            if early_stopping.early_stop:
                print("Early stopping Stage 2")
                break

    # ====== Final test ======
    
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
        
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            
            signals, targets = batch
            signals = signals.to(device)
            
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            
            targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
            
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

class MAMLLearner(nn.Module):
    """
    Learner wrapper that cleanly supports MAML-style functional forward using
    torch.nn.utils.stateless.functional_call for both the BIOT encoder and the
    BPRegressor head. This avoids in-place param swapping and shape/order bugs.
    """
    def __init__(self, model, regressor):
        super().__init__()
        self.model = model
        self.regressor = regressor

        # Capture parameter names in a deterministic order (matches .parameters())
        self.model_named_params = list(self.model.named_parameters())
        self.reg_named_params = list(self.regressor.named_parameters())

        # Keep a flattened list of parameters with the same order as learner.parameters()
        self._flattened_params = list(p for _, p in self.model_named_params if p.is_floating_point() or p.is_complex())
        self._flattened_params += list(p for _, p in self.reg_named_params if p.is_floating_point() or p.is_complex())
        
        # Save names in the same order (to reconstruct dicts quickly)
        self._model_names_in_order = [n for n, _ in self.model_named_params]
        self._reg_names_in_order = [n for n, _ in self.reg_named_params]

        # Guardrail: the BPRegressor expects 256-dim embeddings as input.
        self.expected_feat_dim = None
        try:
            if hasattr(self.model, "embed_dim"):
                self.expected_feat_dim = int(self.model.embed_dim)
        except Exception:
            self.expected_feat_dim = None

    def _split_vars_to_dicts(self, vars_list):
        """
        Split a flat list of tensors into two param dicts matching model and regressor.
        The ordering *must* match how we created fast_weights.
        """
        n_model = len(self._model_names_in_order)
        model_vars = vars_list[:n_model]
        reg_vars = vars_list[n_model:]

        model_param_dict = {name: tensor for name, tensor in zip(self._model_names_in_order, model_vars)}
        reg_param_dict = {name: tensor for name, tensor in zip(self._reg_names_in_order, reg_vars)}
        return model_param_dict, reg_param_dict

    def forward(self, x, vars=None):
        """
        If vars is None -> regular forward (encoder -> head).
        If vars is not None -> functional forward using provided weights (MAML inner loop).
        """
        if vars is None:
            feats = self.model(x)            # Expect [B, 256] from BIOT
            assert feats.dim() == 2, f"Encoder output must be [B, D], got {list(feats.shape)}"
            if self.expected_feat_dim is not None:
                assert feats.shape[-1] == self.expected_feat_dim,                     f"Encoder features mismatch: expected {self.expected_feat_dim}, got {feats.shape[-1]}"
            out = self.regressor(feats)      # [B, 3]
            return out

        # Functional forward: use stateless.functional_call so autograd can build
        # a graph w.r.t. 'vars' (fast weights). This is the canonical PyTorch way.
        from torch.nn.utils.stateless import functional_call

        model_param_dict, reg_param_dict = self._split_vars_to_dicts(vars)

        # Run encoder with provided params
        feats = functional_call(self.model, model_param_dict, (x,))

        # Sanity checks to catch shape issues early
        if feats.dim() == 3 and feats.shape[1] != feats.shape[-1]:
            # If accidentally a sequence [B, T, C] got returned without pooling, try to pool
            feats = feats.mean(dim=1)

        assert feats.dim() == 2, f"Encoder output must be [B, D]; got {list(feats.shape)}"
        if self.expected_feat_dim is not None:
            assert feats.shape[-1] == self.expected_feat_dim,                 f"Encoder features mismatch: expected {self.expected_feat_dim}, got {feats.shape[-1]}"

        # Run head with provided params
        out = torch.func.functional_call(self.regressor, reg_param_dict, (feats,))
        return out


def maml_meta_training(
    save_name,
    checkpoint_path,
    writer,
    model_name,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    early_stopping,
    config,
    device
):
    
    # Load best model
    model = get_model_architecture(config)
    #checkpoint = torch.load('/data/users/mgaspari/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTencoder_ft_stage2', weights_only=False)
    checkpoint = torch.load('/home/michele/Documents/ContinualBP/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTencoder_ft_stage2', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    #checkpoint = torch.load('/data/users/mgaspari/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTregressor_ft_stage2', weights_only=False)
    checkpoint = torch.load('/home/michele/Documents/ContinualBP/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTregressor_ft_stage2', weights_only=False)
    bp_regressor.load_state_dict(checkpoint['model'])
    bp_regressor = bp_regressor.to(device)
    print("Loaded checkpoints")
    
    # Create enhanced learner wrapper
    learner = MAMLLearner(model, bp_regressor).to(device)
    
    # Configure functional forward approach based on your preference
    learner.use_pure_functional = config.get('use_pure_functional', False)  # Set to True for memory efficiency
    
    # ==== Meta-learning (MAML) Stage ====
    
    # ---- Setup ----
    
    # Hyperparams / defaults
    meta_epochs = config.get('max_training_epochs')
    meta_val_tasks = config.get('meta_val_tasks')
    grad_clip_norm = config.get('grad_clip', 10.0)  # Increased for MAML
    
    # Enhanced scheduling parameters
    meta_lr_schedule = config.get('meta_lr_schedule')
    inner_lr_schedule = config.get('inner_lr_schedule')
    inner_steps_schedule = config.get('inner_steps_schedule')
    
    # MAML specific parameters
    second_order = config.get('second_order_maml')
    
    print(f"==== Stage 3 Starting MAML meta-training ====")
    print(f"  - Meta LR schedule: {meta_lr_schedule}")
    print(f"  - Inner LR schedule: {inner_lr_schedule}")
    print(f"  - Inner steps schedule: {inner_steps_schedule}")
    print(f"  - Second-order gradients: {second_order}")
    print(f"  - Using pure functional: {learner.use_pure_functional}")
    
    # Meta-optimizer for the meta-parameters
    meta_optimizer = torch.optim.Adam(learner.parameters(), lr=get_meta_lr(0, config))
    
    # Learning rate scheduler for meta-optimizer
    if config.get('meta_lr_scheduler') == 'cosine':
        meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer, T_max=meta_epochs
        )
    elif config.get('meta_lr_scheduler') == 'step':
        meta_scheduler = torch.optim.lr_scheduler.StepLR(
            meta_optimizer, step_size=config.get('meta_lr_step_size', 100), 
            gamma=config.get('meta_lr_gamma', 0.5)
        )
    else:
        meta_scheduler = None

    best_val = float("+inf")
    global_step = 0

    # Outer loop: epochs
    for epoch in range(meta_epochs):
        learner.train()
        
        # Get current hyperparameters based on schedules
        current_meta_lr = get_meta_lr(epoch, config)
        current_inner_lr = get_inner_lr(epoch, config)
        current_inner_steps = get_inner_steps(epoch, config)
        
        # Update meta-optimizer learning rate if using manual schedule
        if meta_scheduler is None:
            for param_group in meta_optimizer.param_groups:
                param_group['lr'] = current_meta_lr
        
        epoch_query_loss_meter = AverageMeter(name='meta/train_query_loss')
        epoch_support_loss_meter = AverageMeter(name='meta/train_support_loss')
        epoch_meta_loss_meter = AverageMeter(name='meta/train_meta_loss')
        
        # Log current hyperparameters
        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)
                
        # Iterate over tasks provided by train_dataloader
        for batch_idx, batch in enumerate(train_dataloader):
            (Xs, Ys), (Xq, Yq), pid = batch
            meta_batch = Xs.shape[0]

            meta_optimizer.zero_grad()
            
            task_query_losses = []
            task_support_losses = []
            task_query_pre_losses = []
            meta_loss = 0.0

            for t in range(meta_batch):
                # Extract task-specific support/query
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()

                # --------------------
                # Query loss before adaptation (step 0)
                with torch.no_grad():
                    out_q0 = learner(qX)  # Use current meta-parameters
                    loss_q0 = F.smooth_l1_loss(out_q0, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q0, qY)
                task_query_pre_losses.append(loss_q0.item())
                # --------------------

                # Build fast_weights normally (all params)
                fast_weights = [p.clone() for p in learner.parameters()]
                support_losses = []                
                
                for step in range(current_inner_steps):
                    out_s = learner(sX, fast_weights)
                    loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)
                    support_losses.append(loss_s.item())
                    
                    # Select only differentiable (float + requires_grad) params
                    opt_idx, opt_params = zip(
                        *[(i, w) for i, w in enumerate(fast_weights)
                        if w.requires_grad and torch.is_floating_point(w)]
                    )

                    grads = torch.autograd.grad(
                        loss_s,
                        opt_params,
                        create_graph=second_order,
                        retain_graph=True,
                        allow_unused=True
                    )

                    # Clip grads
                    clipped = []
                    for g in grads:
                        if g is None:
                            clipped.append(None)
                        else:
                            norm = g.norm(p=2).clamp(min=1e-12)
                            scale = (grad_clip_norm / norm).clamp(max=1.0)
                            clipped.append(g * scale)

                    # Update only the differentiable params, leave others untouched
                    new_fast_weights = []
                    g_iter = iter(clipped)
                    for i, w in enumerate(fast_weights):
                        if i in opt_idx:
                            g = next(g_iter)
                            if g is not None:
                                new_fast_weights.append(w - current_inner_lr * g)
                            else:
                                new_fast_weights.append(w)
                        else:
                            new_fast_weights.append(w)  # carry over unchanged

                    fast_weights = new_fast_weights
                                    
                    # Log per-inner-step support loss
                    writer.add_scalar(f"inner/support_loss_step{step}", loss_s.item(), global_step)

                # Evaluate adapted model on query set (meta-loss computation)
                out_q = learner(qX, fast_weights)
                loss_q = F.smooth_l1_loss(out_q, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q, qY)

                meta_loss += loss_q / meta_batch  # Average over tasks
                
                task_query_losses.append(loss_q.item())
                task_support_losses.append(np.mean(support_losses))

                epoch_query_loss_meter.update(loss_q.item(), 1)
                epoch_support_loss_meter.update(np.mean(support_losses), 1)

            # Meta-backward pass
            meta_loss.backward()
            
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), grad_clip_norm)
            writer.add_scalar("meta/grad_norm", total_norm.item(), global_step)
            
            # Meta-optimization step
            meta_optimizer.step()
            
            epoch_meta_loss_meter.update(meta_loss.item(), 1)

            global_step += 1
            
            # Logging per meta step
            if (batch_idx + 1) % config.get('meta_log_step', 50) == 0:
                avg_q = float(np.mean(task_query_losses))
                avg_s = float(np.mean(task_support_losses))
                avg_q0 = float(np.mean(task_query_pre_losses))

                step_idx = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('meta/train_query_loss_step', avg_q, step_idx)
                writer.add_scalar('meta/train_support_loss_step', avg_s, step_idx)
                writer.add_scalar('meta/train_query_pre_loss_step', avg_q0, step_idx)
                writer.add_scalar('meta/train_meta_loss_step', meta_loss.item(), step_idx)
                writer.add_histogram("meta/task_query_losses", torch.tensor(task_query_losses), step_idx)

                print(f"[MAML] Epoch {epoch+1} Step {batch_idx+1}/{len(train_dataloader)} - "
                    f"support_loss {avg_s:.6f}, query_loss_pre {avg_q0:.6f}, "
                    f"query_loss_post {avg_q:.6f}, meta_loss {meta_loss.item():.6f} "
                    f"(meta_lr={current_meta_lr:.6f}, inner_lr={current_inner_lr:.6f}, "
                    f"inner_steps={current_inner_steps}, grad_norm={total_norm.item():.3f})")

        # Update learning rate scheduler
        if meta_scheduler is not None:
            meta_scheduler.step()

        # End epoch: run validation on val tasks
        val_loss = evaluate_meta_with_bp_metrics_maml(
            learner, 
            val_dataloader, 
            device, 
            config
        )
        
        writer.add_scalar('meta/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('meta/train_query_loss_epoch', epoch_query_loss_meter.avg, epoch)
        writer.add_scalar('meta/train_support_loss_epoch', epoch_support_loss_meter.avg, epoch)
        writer.add_scalar('meta/train_meta_loss_epoch', epoch_meta_loss_meter.avg, epoch)
        
        print(f"[MAML] Epoch {epoch+1}/{meta_epochs} - "
              f"train_support_loss {epoch_support_loss_meter.avg:.6f}, "
              f"train_query_loss {epoch_query_loss_meter.avg:.6f}, "
              f"train_meta_loss {epoch_meta_loss_meter.avg:.6f}, "
              f"val_loss {val_loss:.6f}")

        # Save best model according to validation loss
        if val_loss < best_val:
            save_ckpt = {
                'epoch': epoch,
                'learner_state_dict': learner.state_dict(),
                'meta_optimizer_state_dict': meta_optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss
            }
            best_model_path = os.path.join(checkpoint_path, f"{save_name}_best_maml")
            torch.save(save_ckpt, best_model_path)
            best_val = val_loss
            print(f"[MAML] New best validation loss: {val_loss:.6f}, saved to {best_model_path}")

        # Early stopping if configured
        if config.get('es_enable', False):
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping MAML meta-training")
                break

    # After meta-training, run final test evaluation
    print(f"[MAML] Starting final test evaluation with best model...")
    
    # Load best model
    best_ckpt = torch.load(os.path.join(checkpoint_path, f"{save_name}_best_maml"), weights_only=False)
    learner.load_state_dict(best_ckpt['learner_state_dict'])
    learner = learner.to(device)
    
    all_test_targets, all_test_outputs = evaluate_meta_with_bp_metrics_maml(
        learner, 
        test_dataloader, 
        device, 
        config, 
        test=True, 
        writer=writer
    )
    
    # Log test metrics and return loss for validation
    _ = call_metric(all_test_targets, all_test_outputs, config, 
                   figure_savepath=os.path.join(config['figure_path'], 'maml_meta_stage'), 
                   plot=True)  
    
    writer.close()
    
    
def evaluate_meta_with_bp_metrics_maml(learner, dataloader, device, config, test=False, writer=None):
    """
    Evaluation function adapted for MAML learner
    """
    learner.eval()
    
    current_inner_lr = config.get('inner_lr')
    current_inner_steps = config.get('inner_steps')
    second_order = config['second_order_maml']
    grad_clip_norm = config['grad_clip']
    
    if test:
        # Meters    
        test_losses = AverageMeter(name='pre_train_test/loss')
        test_sbp_maes = AverageMeter(name='pre_train_test/sbp_mae')
        test_dbp_maes = AverageMeter(name='pre_train_test/dbp_mae')
        test_map_maes = AverageMeter(name='pre_train_test/map_mae')
        test_sbp_mes = AverageMeter(name='pre_train_test/sbp_me')
        test_dbp_mes = AverageMeter(name='pre_train_test/dbp_me')
        test_map_mes = AverageMeter(name='pre_train_test/map_me')
        test_sbp_mae_stds = AverageMeter(name='pre_train_test/sbp_mae_std')
        test_dbp_mae_stds = AverageMeter(name='pre_train_test/dbp_mae_std')
        test_map_mae_stds = AverageMeter(name='pre_train_test/map_mae_std')
        test_sbp_me_stds = AverageMeter(name='pre_train_test/sbp_me_std')
        test_dbp_me_stds = AverageMeter(name='pre_train_test/dbp_me_std')
        test_map_me_stds = AverageMeter(name='pre_train_test/map_me_std')
    
        # Record outputs and targets
        if config['sig2sig']:
            all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
            all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        else:
            all_test_outputs = np.empty((0, 3), dtype=float) # Assuming SBP/DBP/MAP output shape is 3
            all_test_targets = np.empty((0, 3), dtype=float)
    else:
        val_losses = AverageMeter(name='meta/val_loss')

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            (Xs, Ys), (Xq, Yq), pid = batch
            meta_batch = Xs.shape[0]
            
            for t in range(meta_batch):
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()
                
                # Get current meta-parameters
                # Build fast_weights normally (all params)
                fast_weights = [p.clone() for p in learner.parameters()]
                support_losses = []                
                
                for step in range(current_inner_steps):
                    out_s = learner(sX, fast_weights)
                    loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)
                    support_losses.append(loss_s.item())
                    
                    # Select only differentiable (float + requires_grad) params
                    opt_idx, opt_params = zip(
                        *[(i, w) for i, w in enumerate(fast_weights)
                        if w.requires_grad and torch.is_floating_point(w)]
                    )

                    grads = torch.autograd.grad(
                        loss_s,
                        opt_params,
                        create_graph=second_order,
                        retain_graph=True,
                        allow_unused=True
                    )

                    # Clip grads
                    clipped = []
                    for g in grads:
                        if g is None:
                            clipped.append(None)
                        else:
                            norm = g.norm(p=2).clamp(min=1e-12)
                            scale = (grad_clip_norm / norm).clamp(max=1.0)
                            clipped.append(g * scale)

                    # Update only the differentiable params, leave others untouched
                    new_fast_weights = []
                    g_iter = iter(clipped)
                    for i, w in enumerate(fast_weights):
                        if i in opt_idx:
                            g = next(g_iter)
                            if g is not None:
                                new_fast_weights.append(w - current_inner_lr * g)
                            else:
                                new_fast_weights.append(w)
                        else:
                            new_fast_weights.append(w)  # carry over unchanged

                    fast_weights = new_fast_weights
                
                # Evaluate on query
                out_q = learner(qX, fast_weights)
                loss_q = F.smooth_l1_loss(out_q, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q, qY)
                
                metric_values = get_metric_values(loss_q, out_q, qY, config)
            
                if test:
                    # Record predictions and ground truths
                    all_test_outputs = np.concatenate((all_test_outputs, out_q.detach().cpu().numpy()), axis=0)
                    all_test_targets = np.concatenate((all_test_targets, qY.detach().cpu().numpy()), axis=0)

                    # Log metrics
                    metric_values = get_metric_values(loss_q, out_q, qY, config)
                    
                    test_losses.update(metric_values['loss'], qX.size(0))
                    test_sbp_maes.update(metric_values['sbp_mae'], qX.size(0))
                    test_dbp_maes.update(metric_values['dbp_mae'], qX.size(0))
                    test_map_maes.update(metric_values['map_mae'], qX.size(0))
                    test_sbp_mes.update(metric_values['sbp_me'], qX.size(0))
                    test_dbp_mes.update(metric_values['dbp_me'], qX.size(0))
                    test_map_mes.update(metric_values['map_me'], qX.size(0))
                    test_sbp_mae_stds.update(metric_values['sbp_mae_std'], qX.size(0))
                    test_dbp_mae_stds.update(metric_values['dbp_mae_std'], qX.size(0))
                    test_map_mae_stds.update(metric_values['map_mae_std'], qX.size(0))
                    test_sbp_me_stds.update(metric_values['sbp_me_std'], qX.size(0))
                    test_dbp_me_stds.update(metric_values['dbp_me_std'], qX.size(0))
                    test_map_me_stds.update(metric_values['map_me_std'], qX.size(0))
                    
                else:
                    val_losses.update(loss_q.item(), qX.size(0))
    
    if test:
        
        # Log test metrics
        writer.add_scalar('test/loss_final', test_losses.avg, config.get('max_training_epochs'))
        writer.add_scalar('test/sbp_mae_final', test_sbp_maes.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_mae_final', test_dbp_maes.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_mae_final', test_map_maes.avg, config['max_training_epochs'])
        writer.add_scalar('test/sbp_me_final', test_sbp_mes.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_me_final', test_dbp_mes.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_me_final', test_map_mes.avg, config['max_training_epochs'])
        writer.add_scalar('test/sbp_mae_std_final', test_sbp_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_mae_std_final', test_dbp_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_mae_std_final', test_map_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/sbp_me_std_final', test_sbp_me_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_me_std_final', test_dbp_me_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_me_std_final', test_map_me_stds.avg, config['max_training_epochs'])
        
        return all_test_targets, all_test_outputs

    else:
        return val_losses.avg
    

class BPRegressor(torch.nn.Module):
    """
    Blood pressure regressor for SBP/DBP/MAP prediction during Reptile stage.
    """
    def __init__(self, input_dim, output_dim=2):  # 2 for SBP/DBP, can be 3 for SBP/DBP/MAP
        super(BPRegressor, self).__init__()
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.regressor(x)


def evaluate_meta_with_bp_metrics(
    model,
    bp_regressor,
    dataloader,
    device,
    config,
    test=False,
    writer=None
):
    """
    Meta-evaluation with AAMI/BHS metrics.
    - For each task: adapt model on support, evaluate on query.
    - Use existing get_metric_values() for per-batch metrics.
    - Aggregate SBP/DBP errors -> AAMI / BHS summary.
    """
    model = model.to(device)
    model.eval()
    bp_regressor = bp_regressor.to(device)
    bp_regressor.eval()

    if test:
        # Meters    
        test_losses = AverageMeter(name='pre_train_test/loss')
        test_sbp_maes = AverageMeter(name='pre_train_test/sbp_mae')
        test_dbp_maes = AverageMeter(name='pre_train_test/dbp_mae')
        test_map_maes = AverageMeter(name='pre_train_test/map_mae')
        test_sbp_mes = AverageMeter(name='pre_train_test/sbp_me')
        test_dbp_mes = AverageMeter(name='pre_train_test/dbp_me')
        test_map_mes = AverageMeter(name='pre_train_test/map_me')
        test_sbp_mae_stds = AverageMeter(name='pre_train_test/sbp_mae_std')
        test_dbp_mae_stds = AverageMeter(name='pre_train_test/dbp_mae_std')
        test_map_mae_stds = AverageMeter(name='pre_train_test/map_mae_std')
        test_sbp_me_stds = AverageMeter(name='pre_train_test/sbp_me_std')
        test_dbp_me_stds = AverageMeter(name='pre_train_test/dbp_me_std')
        test_map_me_stds = AverageMeter(name='pre_train_test/map_me_std')
    
        # Record outputs and targets
        if config['sig2sig']:
            all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
            all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        else:
            all_test_outputs = np.empty((0, 3), dtype=float) # Assuming SBP/DBP/MAP output shape is 3
            all_test_targets = np.empty((0, 3), dtype=float)
    else:
        val_losses = AverageMeter(name='meta/val_loss')

    for batch_idx, batch in enumerate(dataloader):

        (Xs, Ys), (Xq, Yq), pid = batch
        
        meta_batch = Xs.shape[0]
        
        for t in range(meta_batch):
        
            # Extract task-specific support/query
            sX = Xs[t].to(device).float()
            qX = Xq[t].to(device).float()
            
            sY = Ys[t].to(device).float()
            qY = Yq[t].to(device).float()

            # --- Inner adaptation ---
            adapted = copy.deepcopy(model).to(device)
            adapted_regressor = copy.deepcopy(bp_regressor).to(device)
            
            adapted.train()
            adapted_regressor.train()
                
            # Combine parameters for optimization
            params_to_optimize = list(adapted.parameters())
            params_to_optimize.extend(list(adapted_regressor.parameters()))
                
            inner_opt = build_inner_optimizer(
                adapted,
                adapted_regressor,
                base_lr=config.get('lr_inner'),
                config=config
            )
            inner_steps = config['inner_steps_max']
            
            for _ in range(inner_steps):
                inner_opt.zero_grad()
                
                feats_s = adapted(sX)
                out_s = adapted_regressor(feats_s)
                    
                loss_s = F.mse_loss(out_s, sY) if config.get("criterion") == "MSELoss" else F.smooth_l1_loss(out_s, sY)
                
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=config['grad_clip'])
                torch.nn.utils.clip_grad_norm_(adapted_regressor.parameters(), max_norm=config['grad_clip'])
                inner_opt.step()

            # --- Evaluate on query ---
            adapted.eval()
            adapted_regressor.eval()
                
            with torch.no_grad():
                feats_q = adapted(qX)
                out_q = adapted_regressor(feats_q)
                    
                loss_q = F.mse_loss(out_q, qY) if config.get("criterion", "MSELoss") == "MSELoss" else F.smooth_l1_loss(out_q, qY)
                
                metric_values = get_metric_values(loss_q, out_q, qY, config)
            
                if test:
                    # Record predictions and ground truths
                    all_test_outputs = np.concatenate((all_test_outputs, out_q.detach().cpu().numpy()), axis=0)
                    all_test_targets = np.concatenate((all_test_targets, qY.detach().cpu().numpy()), axis=0)

                    # Log metrics
                    metric_values = get_metric_values(loss_q, out_q, qY, config)
                    
                    test_losses.update(metric_values['loss'], qX.size(0))
                    test_sbp_maes.update(metric_values['sbp_mae'], qX.size(0))
                    test_dbp_maes.update(metric_values['dbp_mae'], qX.size(0))
                    test_map_maes.update(metric_values['map_mae'], qX.size(0))
                    test_sbp_mes.update(metric_values['sbp_me'], qX.size(0))
                    test_dbp_mes.update(metric_values['dbp_me'], qX.size(0))
                    test_map_mes.update(metric_values['map_me'], qX.size(0))
                    test_sbp_mae_stds.update(metric_values['sbp_mae_std'], qX.size(0))
                    test_dbp_mae_stds.update(metric_values['dbp_mae_std'], qX.size(0))
                    test_map_mae_stds.update(metric_values['map_mae_std'], qX.size(0))
                    test_sbp_me_stds.update(metric_values['sbp_me_std'], qX.size(0))
                    test_dbp_me_stds.update(metric_values['dbp_me_std'], qX.size(0))
                    test_map_me_stds.update(metric_values['map_me_std'], qX.size(0))
                    
                else:
                    val_losses.update(loss_q.item(), qX.size(0))

            del adapted
            del adapted_regressor
            torch.cuda.empty_cache()

    if test:
        
        # Log test metrics
        writer.add_scalar('test/loss_final', test_losses.avg, config.get('max_training_epochs'))
        writer.add_scalar('test/sbp_mae_final', test_sbp_maes.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_mae_final', test_dbp_maes.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_mae_final', test_map_maes.avg, config['max_training_epochs'])
        writer.add_scalar('test/sbp_me_final', test_sbp_mes.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_me_final', test_dbp_mes.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_me_final', test_map_mes.avg, config['max_training_epochs'])
        writer.add_scalar('test/sbp_mae_std_final', test_sbp_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_mae_std_final', test_dbp_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_mae_std_final', test_map_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/sbp_me_std_final', test_sbp_me_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/dbp_me_std_final', test_dbp_me_stds.avg, config['max_training_epochs'])
        writer.add_scalar('test/map_me_std_final', test_map_me_stds.avg, config['max_training_epochs'])
        
        return all_test_targets, all_test_outputs

    else:
        return val_losses.avg
    

def get_meta_lr(epoch, config):
    
    meta_lr_schedule = config['meta_lr_schedule']
    meta_epochs = config['max_training_epochs']
    base_meta_lr = config['meta_lr']
    
    if meta_lr_schedule == 'constant':
        return base_meta_lr
    elif meta_lr_schedule == 'cosine':
        return base_meta_lr * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
    elif meta_lr_schedule == 'step':
        lr = base_meta_lr
        meta_lr_steps = config['meta_lr_steps']
        meta_lr_gamma = config['meta_lr_gamma']
    
        for step in meta_lr_steps:
            if epoch >= step:
                lr *= meta_lr_gamma
        return lr
    elif meta_lr_schedule == 'exponential':
        meta_lr_decay = config['meta_lr_decay']
        
        return base_meta_lr * (meta_lr_decay ** epoch)
    else:
        return base_meta_lr

def get_inner_lr(epoch, config):
    
    inner_lr_schedule = config['inner_lr_schedule']
    meta_epochs = config['max_training_epochs']
    base_lr_inner = config['lr_inner']
    inner_lr_min = config['inner_lr_min']
    
    if inner_lr_schedule == 'constant':
        return base_lr_inner
    elif inner_lr_schedule == 'cosine':
        return inner_lr_min + (base_lr_inner - inner_lr_min) * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
    else:
        return base_lr_inner

def get_inner_steps(epoch, config):
    
    meta_epochs = config['max_training_epochs']
    inner_steps_schedule = config['inner_steps_schedule']
    base_inner_steps = config['inner_steps']
    inner_steps_max = config['inner_steps_max']
    
    if inner_steps_schedule == 'constant':
        return base_inner_steps
    elif inner_steps_schedule == 'increasing':
        progress = epoch / meta_epochs
        return int(base_inner_steps + progress * (inner_steps_max - base_inner_steps))
    else:
        return base_inner_steps


# === Inner optimizer builder (two behaviors: all vs head-only) ===
def build_inner_optimizer(adapted_model, adapted_regressor, base_lr, config):
    """
    Build inner optimizer for Reptile.
    Behaviors:
    - inner_adapt='all'  : adapt backbone + regressor
    - inner_adapt='head' : freeze backbone, adapt only regressor
    Also supports per-group LR multipliers and opt type.
    """
    mode = config.get('inner_adapt')   
    opt_type = config.get('inner_opt').lower()  
    head_mult = float(config.get('inner_head_lr_mult'))
    bb_mult   = float(config.get('inner_backbone_lr_mult'))
    weight_decay = float(config.get('weight_decay'))
    momentum = float(config.get('sgd_momentum'))

    # Decide which params to adapt
    backbone_params = [p for p in adapted_model.parameters()
                       if p.is_floating_point() or p.is_complex()]
    head_params = [p for p in adapted_regressor.parameters()
                   if p.is_floating_point() or p.is_complex()]

    if mode == 'head':
        # freeze backbone
        for p in backbone_params:
            p.requires_grad = False
        params = [
            {'params': head_params, 'lr': base_lr * head_mult},
        ]
    elif mode == 'all':
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
        raise ValueError("config['inner_adapt'] must be 'all' or 'head'")

    # Build optimizer
    if opt_type == 'adam':
        inner_opt = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        inner_opt = torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError("config['inner_opt'] must be 'adam' or 'sgd'")

    return inner_opt
    

def pre_training(
    save_name,
    checkpoint_path,
    writer,
    model_name,
    model,
    early_stopping,
    config,
    device
    ):
    
    # Optionally load a pretrained checkpoint
    if config.get('pretrained_path') is not None:
        ckpt = torch.load(config['pretrained_path'], map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model_state = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and any(k.startswith('encoder') for k in ckpt.keys()):
            model_state = ckpt
        else:
            model_state = ckpt
        
        model.load_state_dict(model_state, strict=False)
        print(f"[Reptile] Loaded pretrained weights (all) from {config['pretrained_path']}")

    model = model.to(device)
    
    # ---- Pre-training task with improved loss handling ----
    
    pretrain_ds = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        sig2sig=config['sig2sig'],
        bp_pattern=False,
        min_subject_sample_number=config['min_subject_sample_number'],
    )
    
    (pre_train_sampler, pre_val_sampler, pre_test_sampler) = pretrain_ds.get_pretraining_samplers()
    
    pre_train_dataloader = DataLoader(pretrain_ds, sampler=pre_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)    
    pre_valid_dataloader = DataLoader(pretrain_ds, sampler=pre_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    pre_test_dataloader = DataLoader(pretrain_ds, sampler=pre_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    
    """
    Two-stage supervised fine-tuning with:
      - Stage 1: head-only training (backbone frozen)
      - Stage 2: backbone + head with param-group LRs
      - Cosine LR schedule per stage with linear warmup
      - Original trainable mask to avoid unfreezing fixed params (e.g., encoder.index)
    """

    # ====== Store original trainable mask ======
    original_trainable = {n: p.requires_grad for n, p in model.named_parameters()}

    def set_requires_grad_safe(module, req, name_prefix=""):
        """Toggle requires_grad but respect original_trainable mask when unfreezing."""
        for n, p in module.named_parameters():
            full_name = f"{name_prefix}.{n}" if name_prefix else n
            if not req:  # freezing
                p.requires_grad = False
            else:        # unfreezing
                if original_trainable.get(full_name, True):
                    p.requires_grad = True

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

    # ====== Common config ======
    base_lr = config.get('base_lr')
    backbone_lr_mul = config.get('backbone_lr_multiplier')
    weight_decay = config.get('weight_decay')
    grad_clip = config.get('grad_clip')

    stage1_epochs = config.get('ft_stage1_epochs')
    stage2_epochs = config.get('ft_stage2_epochs')
    warmup_stage1 = config.get('warmup_epochs_stage1')
    warmup_stage2 = config.get('warmup_epochs_stage2')

    best_val = float("+inf")

    # ====== Stage 1: Head-only ======
    if config.get('freeze_backbone_first'):
        if hasattr(model, 'encoder'):
            set_requires_grad_safe(model.encoder, False, name_prefix="encoder")
    if hasattr(model, 'channel_proj'):
        set_requires_grad_safe(model.channel_proj, True, name_prefix="channel_proj")
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad] + [p for p in bp_regressor.parameters() if p.requires_grad]
    optimizer_stage1 = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=stage1_epochs)

    print(f"==== Stage 1: head-only, {stage1_epochs} epochs, base_lr={base_lr} ====")
    for epoch in range(stage1_epochs):
        model.train()
        bp_regressor.train()
        
        # Warmup LR
        for pg in optimizer_stage1.param_groups:
            pg['lr'] = linear_warmup(epoch, warmup_stage1, base_lr)

        train_losses = AverageMeter(name='pre_train_stage_1/train_loss')
        for batch_idx, batch in enumerate(pre_train_dataloader):
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device), batch[1][2].to(device)), dim=-1)

            optimizer_stage1.zero_grad()
            
            feats = model(signals)
            outputs = bp_regressor(feats)

            if config['criterion'] == 'MSELoss':
                loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion")

            loss *= config.get('lambda_supervised')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            torch.nn.utils.clip_grad_norm_(bp_regressor.parameters(), max_norm=grad_clip)
            optimizer_stage1.step()

            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('pre_train_stage_1/train_loss', loss.item(), epoch * len(pre_train_dataloader) + batch_idx)

        writer.add_scalar('pre_train_stage_1/train_loss_epoch', train_losses.avg, epoch)
        scheduler_stage1.step()

        # Validation
        model.eval()
        bp_regressor.eval()
        val_losses = AverageMeter(name='pre_train_stage_1/val_loss')
        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_valid_dataloader):
                signals, targets = batch
                signals = signals.to(device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device), batch[1][2].to(device)), dim=-1)
                
                feats = model(signals)
                outputs = bp_regressor(feats)
                
                vloss = F.mse_loss(outputs, targets) if config['criterion'] == 'MSELoss' else F.smooth_l1_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('pre_train_stage_1/val_loss_epoch', val_losses.avg, epoch)
        print(f"[Stage1] Epoch {epoch+1}/{stage1_epochs} - Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            save_status(None, epoch, model_name + "encoder_ft_stage1", save_name, model, optimizer_stage1, scheduler_stage1, val_losses, checkpoint_path, config)
            save_status(None, epoch, model_name + "regressor_ft_stage1", save_name, bp_regressor, optimizer_stage1, scheduler_stage1, val_losses, checkpoint_path, config)
            
            best_val = val_losses.avg

    print(f"==== Stage 1 completed, best val loss {best_val} ====")

    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name + "encoder_ft_stage1", save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    print(f"[Reptile] Loaded encoder pretrained weights after pre-training stage 1")
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    load_status(None, model_name + "regressor_ft_stage1", save_name, bp_regressor, None, None, checkpoint_path, config)
    bp_regressor = bp_regressor.to(device)
    print(f"[Reptile] Loaded regressor pretrained weights after pre-training stage 1")
    
    # ====== Stage 2: Backbone + Head ======
    lr_backbone = base_lr * backbone_lr_mul
    set_requires_grad_safe(model.encoder, True, name_prefix="encoder")
    set_requires_grad_safe(model.channel_proj, True, name_prefix="channel_proj")

    backbone_params, head_params = get_backbone_and_head_params(model, bp_regressor)
    backbone_params = [p for p in backbone_params if p.requires_grad]
    head_params = [p for p in head_params if p.requires_grad]

    optimizer_stage2 = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': base_lr}
    ], weight_decay=weight_decay)
    scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=stage2_epochs)

    best_val = float("+inf")
    print(f"==== Stage 2: fine-tune backbone, {stage2_epochs} epochs, head_lr={base_lr}, backbone_lr={lr_backbone} ====")
    for epoch in range(stage2_epochs):
        model.train()
        bp_regressor.train()
        
        # Warmup
        for i, pg in enumerate(optimizer_stage2.param_groups):
            target_lr = base_lr if i == 1 else lr_backbone
            pg['lr'] = linear_warmup(epoch, warmup_stage2, target_lr)

        train_losses = AverageMeter(name='pre_train_stage_2/train_loss')
        for batch_idx, batch in enumerate(pre_train_dataloader):
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device), batch[1][2].to(device)), dim=-1)

            optimizer_stage2.zero_grad()
            
            feats = model(signals)
            outputs = bp_regressor(feats)
            
            if config['criterion'] == 'MSELoss':
                loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion")

            loss *= config.get('lambda_supervised', 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            torch.nn.utils.clip_grad_norm_(bp_regressor.parameters(), max_norm=grad_clip)
            optimizer_stage2.step()

            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('pre_train_stage_2/train_loss', loss.item(), epoch * len(pre_train_dataloader) + batch_idx)

        writer.add_scalar('pre_train_stage_2/train_loss_epoch', train_losses.avg, epoch)
        scheduler_stage2.step()

        # Validation
        model.eval()
        bp_regressor.eval()
        val_losses = AverageMeter(name='pre_train_stage_2/val_loss')
        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_valid_dataloader):
                signals, targets = batch
                signals = signals.to(device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device), batch[1][2].to(device)), dim=-1)
                
                feats = model(signals)
                outputs = bp_regressor(feats)
                
                vloss = F.mse_loss(outputs, targets) if config['criterion'] == 'MSELoss' else F.smooth_l1_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('pre_train_stage_2/val_loss_epoch', val_losses.avg, epoch)
        print(f"[Stage2] Epoch {epoch+1}/{stage2_epochs} - Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            save_status(None, epoch, model_name + "encoder_ft_stage2", save_name, model, optimizer_stage2, scheduler_stage2, val_losses, checkpoint_path, config)
            save_status(None, epoch, model_name + "regressor_ft_stage2", save_name, bp_regressor, optimizer_stage2, scheduler_stage2, val_losses, checkpoint_path, config)
            
            best_val = val_losses.avg

        if config['es_enable']:
            early_stopping(val_losses.avg)
            if early_stopping.early_stop:
                print("Early stopping Stage 2")
                break
            
    # ====== Final test ======
    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name + "encoder_ft_stage2", save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    load_status(None, model_name + "regressor_ft_stage2", save_name, bp_regressor, None, None, checkpoint_path, config)
    bp_regressor = bp_regressor.to(device)    
    
    # Meters    
    test_losses = AverageMeter(name='pre_train_test/loss')
    test_sbp_maes = AverageMeter(name='pre_train_test/sbp_mae')
    test_dbp_maes = AverageMeter(name='pre_train_test/dbp_mae')
    test_map_maes = AverageMeter(name='pre_train_test/map_mae')
    test_sbp_mes = AverageMeter(name='pre_train_test/sbp_me')
    test_dbp_mes = AverageMeter(name='pre_train_test/dbp_me')
    test_map_mes = AverageMeter(name='pre_train_test/map_me')
    test_sbp_mae_stds = AverageMeter(name='pre_train_test/sbp_mae_std')
    test_dbp_mae_stds = AverageMeter(name='pre_train_test/dbp_mae_std')
    test_map_mae_stds = AverageMeter(name='pre_train_test/map_mae_std')
    test_sbp_me_stds = AverageMeter(name='pre_train_test/sbp_me_std')
    test_dbp_me_stds = AverageMeter(name='pre_train_test/dbp_me_std')
    test_map_me_stds = AverageMeter(name='pre_train_test/map_me_std')
    
    # Record outputs and targets
    if config['sig2sig']:
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
    else:
        all_test_outputs = np.empty((0, 3), dtype=float) # Assuming SBP/DBP/MAP output shape is 3
        all_test_targets = np.empty((0, 3), dtype=float)
        
    model.eval()
    bp_regressor.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(pre_test_dataloader):
            
            signals, targets = batch
            signals = signals.to(device)
            
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            
            targets = targets.to(device) if config['sig2sig'] else torch.cat((batch[1][0].to(device), batch[1][1].to(device), batch[1][2].to(device)), dim=-1)
            
            feats = model(signals)
            outputs = bp_regressor(feats)
            
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
            test_map_maes.update(metric_values['map_mae'], signals.size(0))
            test_sbp_mes.update(metric_values['sbp_me'], signals.size(0))
            test_dbp_mes.update(metric_values['dbp_me'], signals.size(0))
            test_map_mes.update(metric_values['map_me'], signals.size(0))
            test_sbp_mae_stds.update(metric_values['sbp_mae_std'], signals.size(0))
            test_dbp_mae_stds.update(metric_values['dbp_mae_std'], signals.size(0))
            test_map_mae_stds.update(metric_values['map_mae_std'], signals.size(0))
            test_sbp_me_stds.update(metric_values['sbp_me_std'], signals.size(0))
            test_dbp_me_stds.update(metric_values['dbp_me_std'], signals.size(0))
            test_map_me_stds.update(metric_values['map_me_std'], signals.size(0))


    # Log test metrics
    # Note: `epoch` here is the last epoch of training, not ideal for test summary
    writer.add_scalar('pre_train_test/loss_final', test_losses.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/sbp_mae_final', test_sbp_maes.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/dbp_mae_final', test_dbp_maes.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/map_mae_final', test_map_maes.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/sbp_me_final', test_sbp_mes.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/dbp_me_final', test_dbp_mes.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/map_me_final', test_map_mes.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/sbp_mae_std_final', test_sbp_mae_stds.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/dbp_mae_std_final', test_dbp_mae_stds.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/map_mae_std_final', test_map_mae_stds.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/sbp_me_std_final', test_sbp_me_stds.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/dbp_me_std_final', test_dbp_me_stds.avg, config['ft_stage2_epochs'])
    writer.add_scalar('pre_train_test/map_me_std_final', test_map_me_stds.avg, config['ft_stage2_epochs'])
    
    # Log test metrics (optionally, plot them) and return loss for validation
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'pretraining_supervised_stage'), plot=True)    
    
    print(f"==== Stage 2 completed ====")

def reptile_meta_training(
    save_name,
    checkpoint_path,
    writer,
    model_name,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    early_stopping,
    config,
    device
):
    
    #pre_training(save_name, checkpoint_path, writer, model_name, model, early_stopping, config, device)

    # Load best model after pre-training
    #model = get_model_architecture(config)
    #load_status(None, model_name + "encoder_ft_stage2", save_name, model, None, None, checkpoint_path, config)
    #model = model.to(device)
    #print(f"[Reptile] Loaded encoder pretrained weights after pre-training stage 2")
    #
    #feat_dim = model.embed_dim
    #output_dim = 3  # SBP, DBP, MAP
    #bp_regressor = BPRegressor(feat_dim, output_dim)
    #load_status(None, model_name + "regressor_ft_stage2", save_name, bp_regressor, None, None, checkpoint_path, config)
    #bp_regressor = bp_regressor.to(device)
    #print(f"[Reptile] Loaded regressor pretrained weights after pre-training stage 2")
    
    # Load best model
    model = get_model_architecture(config)
    checkpoint = torch.load('/data/users/mgaspari/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTencoder_ft_stage2', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    checkpoint = torch.load('/data/users/mgaspari/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTregressor_ft_stage2', weights_only=False)
    bp_regressor.load_state_dict(checkpoint['model']) # N.B. look at the save_status keys, using model as key for the checkpoint is correct to restore'
    bp_regressor = bp_regressor.to(device)
    print("Loaded chckpoints")
    
    # ==== Meta-learning (Reptile) Stage ====
    
    # ---- Setup ----
    
    # Hyperparams / defaults
    meta_epochs = config.get('max_training_epochs')
    meta_val_tasks = config.get('meta_val_tasks')
    grad_clip_norm = config['grad_clip']
    
    # Keep meta-params as an explicit dict (include regressor)
    meta_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
    regressor_meta_params = {f"regressor.{n}": p.detach().clone().to(device) for n, p in bp_regressor.named_parameters()}
    meta_params.update(regressor_meta_params)
    
    # Initialize first-order approximation option (faster Reptile variant)
    use_first_order = config.get('first_order_reptile', True)
    
    # Enhanced scheduling parameters (keeping existing functionality)
    meta_lr_schedule = config.get('meta_lr_schedule')
    inner_lr_schedule = config.get('inner_lr_schedule')
    inner_steps_schedule = config.get('inner_steps_schedule')
    
    print(f"==== Stage 3 Starting meta-training with schedules ====")
    print(f"  - Meta LR schedule: {meta_lr_schedule}")
    print(f"  - Inner LR schedule: {inner_lr_schedule}")
    print(f"  - Inner steps schedule: {inner_steps_schedule}")
    print(f"  - First-order approximation: {use_first_order}")
    
    best_val = float("+inf")
    global_step = 0  # define once before training loop

    # Outer loop: epochs
    for epoch in range(meta_epochs):
        model.train()
        bp_regressor.train()
        
        # Get current hyperparameters based on schedules
        current_meta_lr = get_meta_lr(epoch, config)
        current_inner_lr = get_inner_lr(epoch, config)
        current_inner_steps = get_inner_steps(epoch, config)
        
        epoch_query_loss_meter = AverageMeter(name='meta/train_query_loss')
        epoch_support_loss_meter = AverageMeter(name='meta/train_support_loss')
        
        # Log current hyperparameters
        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)
                
        # Iterate over tasks provided by train_dataloader
        for batch_idx, batch in enumerate(train_dataloader):
            (Xs, Ys), (Xq, Yq), pid = batch
            meta_batch = Xs.shape[0]

            # Accumulate parameter deltas across tasks
            acc_deltas = {k: torch.zeros_like(v) for k, v in meta_params.items()}
            task_query_losses = []
            task_support_losses = []
            task_query_pre_losses = []   # NEW: pre-adaptation query loss per task

            for t in range(meta_batch):
                # Extract task-specific support/query
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()

                # Create adapted model copy for this task
                adapted = copy.deepcopy(model).to(device)
                adapted_regressor = copy.deepcopy(bp_regressor).to(device)
                adapted.train(), adapted_regressor.train()

                inner_opt = build_inner_optimizer(adapted, adapted_regressor, current_inner_lr, config)

                support_losses = []

                # --------------------
                # Query loss before adaptation (step 0)
                with torch.no_grad():
                    feats_q0 = adapted(qX)
                    out_q0 = adapted_regressor(feats_q0)
                    loss_q0 = F.smooth_l1_loss(out_q0, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q0, qY)
                task_query_pre_losses.append(loss_q0.item())
                # --------------------

                # Inner loop: adapt to support
                for step in range(current_inner_steps):
                    inner_opt.zero_grad()
                    feats_s = adapted(sX)
                    out_s = adapted_regressor(feats_s)
                    loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)
                    loss_s.backward()

                    # Log gradient norm before clipping
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        list(adapted.parameters()) + list(adapted_regressor.parameters()), grad_clip_norm
                    )
                    writer.add_scalar("inner/grad_norm", total_norm.item(), global_step)

                    # Update params (first-order or standard reptile)
                    if not use_first_order:
                        inner_opt.step()
                    else:
                        with torch.no_grad():
                            for param in list(adapted.parameters()) + list(adapted_regressor.parameters()):
                                if param.grad is not None:
                                    param.data -= current_inner_lr * param.grad.data
                        inner_opt.zero_grad()

                    support_losses.append(loss_s.item())
                    # Log per-inner-step support loss
                    writer.add_scalar(f"inner/support_loss_step{step}", loss_s.item(), global_step)

                # Evaluate adapted model on query (after adaptation)
                adapted.eval(), adapted_regressor.eval()
                with torch.no_grad():
                    feats_q = adapted(qX)
                    out_q = adapted_regressor(feats_q)
                    loss_q = F.smooth_l1_loss(out_q, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q, qY)

                task_query_losses.append(loss_q.item())
                task_support_losses.append(np.mean(support_losses))

                epoch_query_loss_meter.update(loss_q.item(), 1)
                epoch_support_loss_meter.update(np.mean(support_losses), 1)

                # Compute parameter delta norm for this task
                for n, p in adapted.named_parameters():
                    if n in meta_params:
                        delta = (p.detach() - meta_params[n])
                        acc_deltas[n] += delta

                for n, p in adapted_regressor.named_parameters():
                    reg_key = f"regressor.{n}"
                    if reg_key in meta_params:
                        delta = (p.detach() - meta_params[reg_key])
                        acc_deltas[reg_key] += delta
                        
                delta_norm = torch.sqrt(sum((d**2).sum() for d in acc_deltas.values()))
                writer.add_scalar("meta/task_delta_norm", delta_norm, global_step)

                # Free memory
                del adapted, adapted_regressor
                torch.cuda.empty_cache()

            # Average deltas over tasks and meta-update
            for k in meta_params:
                avg_delta = acc_deltas[k] / float(meta_batch)
                meta_params[k] = meta_params[k] + current_meta_lr * avg_delta

            # Load updated params back
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in meta_params:
                        p.copy_(meta_params[n])
                for n, p in bp_regressor.named_parameters():
                    reg_key = f"regressor.{n}"
                    if reg_key in meta_params:
                        p.copy_(meta_params[reg_key])

            global_step += 1
            
            # Logging per meta step
            if (batch_idx + 1) % config.get('meta_log_step', 50) == 0:
                avg_q = float(np.mean(task_query_losses))
                avg_s = float(np.mean(task_support_losses))
                avg_q0 = float(np.mean(task_query_pre_losses))   # NEW: query pre-adaptation avg

                step_idx = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('meta/train_query_loss_step', avg_q, step_idx)
                writer.add_scalar('meta/train_support_loss_step', avg_s, step_idx)
                writer.add_scalar('meta/train_query_pre_loss_step', avg_q0, step_idx)
                writer.add_histogram("meta/task_query_losses", torch.tensor(task_query_losses), step_idx)

                print(f"[Meta] Epoch {epoch+1} Step {batch_idx+1}/{len(train_dataloader)} - "
                    f"support_loss {avg_s:.6f}, query_loss_pre {avg_q0:.6f}, query_loss_post {avg_q:.6f} "
                    f"(meta_lr={current_meta_lr:.6f}, inner_lr={current_inner_lr:.6f}, inner_steps={current_inner_steps})")


        # End epoch: run validation on val tasks
        val_loss = evaluate_meta_with_bp_metrics(
            model, 
            bp_regressor, 
            val_dataloader, 
            device, 
            config
        )
        
        writer.add_scalar('meta/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('meta/train_query_loss_epoch', epoch_query_loss_meter.avg, epoch)
        writer.add_scalar('meta/train_support_loss_epoch', epoch_support_loss_meter.avg, epoch)
        
        print(f"[Meta] Epoch {epoch+1}/{meta_epochs} - "
              f"train_support_loss {epoch_support_loss_meter.avg:.6f}, "
              f"train_query_loss {epoch_query_loss_meter.avg:.6f}, "
              f"val_loss {val_loss:.6f}")

        # Save best model according to validation loss
        if val_loss < best_val:
            # Save both model and regressor
            save_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bp_regressor_state_dict': bp_regressor.state_dict(),
                'config': config,
                'val_loss': val_loss
            }
            best_model_path = os.path.join(checkpoint_path, f"{save_name}_best_meta")
            torch.save(save_ckpt, best_model_path)
            best_val = val_loss
            print(f"[Meta] New best validation loss: {val_loss:.6f}, saved to {best_model_path}")

        # Early stopping if configured
        if config.get('es_enable', False):
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping meta-training")
                break

    # After meta-training, run final test evaluation
    print(f"[Meta] Starting final test evaluation with best model...")
    
    # Load best model and regressor
    best_ckpt = torch.load(os.path.join(checkpoint_path, f"{save_name}_best_meta"), weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    bp_regressor.load_state_dict(best_ckpt['bp_regressor_state_dict'])
    model = model.to(device)
    bp_regressor = bp_regressor.to(device)
    
    all_test_targets, all_test_outputs = evaluate_meta_with_bp_metrics(
        model, 
        bp_regressor, 
        test_dataloader, 
        device, 
        config, 
        test=True, 
        writer=writer
    )
    
    # Log test metrics (optionally, plot them) and return loss for validation
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'pretraining_meta_stage'), plot=True)  
    
    writer.close()
    


class MemoryModule(nn.Module):
    """
    Memory-Augmented Neural Network memory module for physiological signal processing.
    """
    def __init__(self, memory_size, memory_dim, feature_dim):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.feature_dim = feature_dim
        
        # Memory matrix [memory_size, memory_dim]
        # memory_dim = feature_dim + output_dim (256 + 3 for features + SBP/DBP/MAP)
        self.register_buffer('memory', torch.randn(memory_size, memory_dim) * 0.1)
        self.register_buffer('memory_age', torch.zeros(memory_size))  # For memory management
        
        # Controllers for memory operations
        self.read_controller = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, memory_size),
            nn.Softmax(dim=-1)
        )
        
        self.write_controller = nn.Sequential(
            nn.Linear(feature_dim + 3, 128),  # features + BP values
            nn.ReLU(),
            nn.Linear(128, memory_size),
            nn.Softmax(dim=-1)
        )
        
        # Value predictor using memory content
        self.value_predictor = nn.Sequential(
            nn.Linear(feature_dim + memory_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # SBP, DBP, MAP
        )
    
    def read(self, features):
        """
        Read from memory based on input features.
        Returns:
            read_content: [batch_size, memory_dim]
        """
        batch_size = features.size(0)

        # Compute attention weights for each sample in batch
        read_weights = self.read_controller(features)  # [batch_size, memory_size]

        # IMPORTANT: use a detached clone of the memory for read so subsequent in-place writes
        # won't modify the tensor that participated in the forward graph.
        # detach() ensures no grad, clone() ensures new storage.
        memory_for_read = self.memory.unsqueeze(0).expand(batch_size, -1, -1).detach().clone()

        # Read from memory using attention
        read_content = torch.bmm(
            read_weights.unsqueeze(1),          # [batch_size, 1, memory_size]
            memory_for_read                     # [batch_size, memory_size, memory_dim] (detached clone)
        ).squeeze(1)  # [batch_size, memory_dim]

        return read_content, read_weights

    def write(self, features, targets):
        """
        Write new experience to memory.
        features: [batch_size, feature_dim]  - NOTE: caller should pass detached features
        targets: [batch_size, 3]
        """
        batch_size = features.size(0)

        # Create memory content to write
        write_content = torch.cat([features, targets], dim=-1)  # [batch_size, memory_dim]

        # Compute write weights. Because we expect features to be detached before write,
        # write_controller shouldn't be part of the gradient graph. Still, we prevent grad tracking here.
        with torch.no_grad():
            write_weights = self.write_controller(write_content)  # [batch_size, memory_size]

            # Update memory using weighted write — pick the highest weight slot per sample.
            # Use .item() to get a python int for indexing (avoid autograd involvement).
            for i in range(batch_size):
                slot_idx = int(torch.argmax(write_weights[i]).item())
                # assign in no_grad (safe)
                self.memory[slot_idx] = write_content[i]
                self.memory_age[slot_idx] = 0  # Reset age

            # Age all other memory slots (in-place but under no_grad)
            self.memory_age += 1

    def clear_old_memories(self, max_age=1000):
        """
        Clear old memories to prevent stagnation.
        """
        with torch.no_grad():
            old_mask = self.memory_age > max_age
            if old_mask.any():
                self.memory[old_mask] = torch.randn_like(self.memory[old_mask]) * 0.1
                self.memory_age[old_mask] = 0
    
    def predict(self, features):
        """
        Predict BP values using features and memory.
        """
        read_content, _ = self.read(features)
        combined = torch.cat([features, read_content], dim=-1)
        return self.value_predictor(combined)


class MANNBPEstimator(nn.Module):
    """
    Complete MANN-based blood pressure estimator.
    """
    def __init__(self, backbone_model, memory_size=2000, feature_dim=256):
        super(MANNBPEstimator, self).__init__()
        self.backbone = backbone_model
        self.memory = MemoryModule(memory_size, feature_dim + 3, feature_dim)
        
    def forward(self, x, update_memory=False, targets=None):
        """
        Forward pass with optional memory update.
        """
        # Extract features using backbone
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # Predict using memory
        predictions = self.memory.predict(features)
        
        # Update memory if training and targets provided
        if update_memory and targets is not None:
            self.memory.write(features.detach(), targets)
        
        return predictions


def mann_meta_training(
    save_name,
    checkpoint_path,
    writer,
    model_name,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    early_stopping,
    config,
    device
):
    """
    Memory-Augmented Neural Network meta-training replacement for Reptile.
    Maintains the same interface but uses continuous memory-based learning.
    """
    
    # Load pretrained weights (same as your original code)
    model = get_model_architecture(config)
    checkpoint = torch.load('/home/michele/Documents/ContinualBP/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTencoder_ft_stage2', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    feat_dim = model.embed_dim
    #bp_regressor = BPRegressor(feat_dim, 3)  # SBP, DBP, MAP
    #checkpoint = torch.load('/home/michele/Documents/ContinualBP/checkpoints/biot_reptile_w_pre_train/biot_reptile_w_pre_train-BIOT-2025_08_28-13_23_50/biot_reptile_w_pre_train/ckpt/BIOTregressor_ft_stage2', weights_only=False)
    #bp_regressor.load_state_dict(checkpoint['model'])
    #bp_regressor = bp_regressor.to(device)
    print("Loaded checkpoints")
    
    # Create MANN model
    memory_size = config.get('memory_size')
    mann_model = MANNBPEstimator(model, memory_size, feat_dim).to(device)
    
    # Setup optimizer for MANN (simpler than meta-learning)
    base_lr = config.get('mann_lr')
    weight_decay = config.get('weight_decay')
    
    # Separate learning rates for backbone and memory components
    backbone_params = list(mann_model.backbone.parameters())
    memory_params = list(mann_model.memory.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': base_lr * config.get('mann_backbone_lr_mult')},
        {'params': memory_params, 'lr': base_lr}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.get('max_training_epochs')
    )
    
    print(f"==== Starting MANN meta-training ====")
    print(f"  - Memory size: {memory_size}")
    print(f"  - Base learning rate: {base_lr}")
    print(f"  - Backbone LR multiplier: {config.get('mann_backbone_lr_mult')}")
    
    best_val = float("+inf")
    meta_epochs = config.get('max_training_epochs')
    
    # Training loop
    for epoch in range(meta_epochs):
        mann_model.train()
        
        epoch_train_loss = AverageMeter(name='mann/train_loss')
        
        # Training phase
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            (Xs, Ys), (Xq, Yq), pid = batch
            
            # Process support and query sets together for memory learning
            # Support set: learn and update memory
            # Query set: test memory retrieval
            
            support_loss = 0.0
            query_loss = 0.0
            
            meta_batch = Xs.shape[0]
            
            for t in range(meta_batch):
                
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()
                
                # Phase 1: Learn from support set and update memory
                support_pred = mann_model(sX, update_memory=True, targets=sY)
                
                if config['criterion'] == 'MSELoss':
                    s_loss = F.mse_loss(support_pred, sY)
                elif config['criterion'] == 'SmoothL1Loss':
                    s_loss = F.smooth_l1_loss(support_pred, sY)
                else:
                    raise ValueError("Invalid criterion")
                
                support_loss += s_loss
                
                # Phase 2: Test on query set (memory retrieval only, no update)
                with torch.no_grad():
                    query_pred = mann_model(qX, update_memory=False)
                    
                    if config['criterion'] == 'MSELoss':
                        q_loss = F.mse_loss(query_pred, qY)
                    elif config['criterion'] == 'SmoothL1Loss':
                        q_loss = F.smooth_l1_loss(query_pred, qY)
                    else:
                        raise ValueError("Invalid criterion")
                    
                    query_loss += q_loss
            
            # Average losses over meta-batch
            support_loss = support_loss / meta_batch
            query_loss = query_loss / meta_batch
            
            # Backpropagate only support loss (memory updates are part of forward pass)
            total_loss = support_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(mann_model.parameters(), max_norm=config['grad_clip'])
            
            optimizer.step()
            
            epoch_train_loss.update(total_loss.item(), meta_batch)
            
            # Periodic memory cleanup
            if (batch_idx + 1) % 100 == 0:
                mann_model.memory.clear_old_memories(max_age=500)
            
            # Logging
            if (batch_idx + 1) % config.get('meta_log_step', 50) == 0:
                step_idx = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('mann/train_support_loss_step', support_loss.item(), step_idx)
                writer.add_scalar('mann/train_query_loss_step', query_loss.item(), step_idx)
                writer.add_scalar('mann/train_total_loss_step', total_loss.item(), step_idx)
                
                print(f"[MANN] Epoch {epoch+1} Step {batch_idx+1}/{len(train_dataloader)} - "
                      f"support_loss {support_loss.item():.6f}, query_loss {query_loss.item():.6f}, "
                      f"total_loss {total_loss.item():.6f}")
        
        # End of epoch: validation
        val_loss = evaluate_mann_validation(
            mann_model, 
            val_dataloader, 
            device, 
            config,
            n_tasks_eval=config.get('meta_val_tasks')
        )
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        writer.add_scalar('mann/train_loss_epoch', epoch_train_loss.avg, epoch)
        writer.add_scalar('mann/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('mann/learning_rate', current_lr, epoch)
        
        print(f"[MANN] Epoch {epoch+1}/{meta_epochs} - "
              f"train_loss {epoch_train_loss.avg:.6f}, "
              f"val_loss {val_loss:.6f}, lr {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val:
            save_ckpt = {
                'epoch': epoch,
                'model_state_dict': mann_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'memory_state': mann_model.memory.memory.clone(),
                'memory_age': mann_model.memory.memory_age.clone()
            }
            best_model_path = os.path.join(checkpoint_path, f"{save_name}_best_mann")
            torch.save(save_ckpt, best_model_path)
            best_val = val_loss
            print(f"[MANN] New best validation loss: {val_loss:.6f}, saved to {best_model_path}")
        
        # Early stopping
        if config.get('es_enable', False):
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping MANN training")
                break
    
    # Final test evaluation
    print(f"[MANN] Starting final test evaluation with best model...")
    
    # Load best model
    best_ckpt = torch.load(os.path.join(checkpoint_path, f"{save_name}_best_mann"), weights_only=False)
    mann_model.load_state_dict(best_ckpt['model_state_dict'])
    mann_model.memory.memory.copy_(best_ckpt['memory_state'])
    mann_model.memory.memory_age.copy_(best_ckpt['memory_age'])
    mann_model = mann_model.to(device)
    
    all_test_targets, all_test_outputs = evaluate_mann_test(
        mann_model,
        test_dataloader,
        device,
        config,
        n_tasks_eval=config.get('meta_test_tasks'),
        writer=writer
    )
    
    # Final metrics computation
    _ = call_metric(all_test_targets, all_test_outputs, config, 
                   figure_savepath=os.path.join(config['figure_path'], 'mann_meta_stage'), 
                   plot=True)
    
    writer.close()


def evaluate_mann_validation(mann_model, val_dataloader, device, config, n_tasks_eval=None):
    """
    Validation evaluation for MANN model.
    """
    mann_model.eval()
    val_losses = AverageMeter(name='mann/val_loss')
    
    tasks_done = 0
    data_iter = iter(val_dataloader)
    
    # Create temporary memory state for validation (don't pollute training memory)
    original_memory = mann_model.memory.memory.clone()
    original_age = mann_model.memory.memory_age.clone()
    
    with torch.no_grad():
        while True:
            if n_tasks_eval is not None and tasks_done >= n_tasks_eval:
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            (Xs, Ys), (Xq, Yq), pid = batch
            
            meta_batch = Xs.shape[0]
            batch_query_losses = []
            
            for t in range(meta_batch):
                # Extract and normalize data
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()
                
                # Learn from support set (update validation memory)
                _ = mann_model(sX, update_memory=True, targets=sY)
                
                # Test on query set
                query_pred = mann_model(qX, update_memory=False)
                
                if config['criterion'] == 'MSELoss':
                    q_loss = F.mse_loss(query_pred, qY)
                elif config['criterion'] == 'SmoothL1Loss':
                    q_loss = F.smooth_l1_loss(query_pred, qY)
                else:
                    raise ValueError("Invalid criterion")
                
                batch_query_losses.append(q_loss.item())
            
            # Average over tasks in this meta-batch
            avg_batch_loss = np.mean(batch_query_losses)
            val_losses.update(avg_batch_loss, meta_batch)
            tasks_done += 1
    
    # Restore original memory state
    mann_model.memory.memory.copy_(original_memory)
    mann_model.memory.memory_age.copy_(original_age)
    
    return val_losses.avg


def evaluate_mann_test(mann_model, test_dataloader, device, config, n_tasks_eval=None, writer=None):
    """
    Final test evaluation for MANN model with detailed metrics.
    """
    mann_model.eval()
    
    # Metrics
    test_losses = AverageMeter(name='mann_test/loss')
    test_sbp_maes = AverageMeter(name='mann_test/sbp_mae')
    test_dbp_maes = AverageMeter(name='mann_test/dbp_mae')
    test_map_maes = AverageMeter(name='mann_test/map_mae')
    test_sbp_mes = AverageMeter(name='mann_test/sbp_me')
    test_dbp_mes = AverageMeter(name='mann_test/dbp_me')
    test_map_mes = AverageMeter(name='mann_test/map_me')
    test_sbp_mae_stds = AverageMeter(name='mann_test/sbp_mae_std')
    test_dbp_mae_stds = AverageMeter(name='mann_test/dbp_mae_std')
    test_map_mae_stds = AverageMeter(name='mann_test/map_mae_std')
    test_sbp_me_stds = AverageMeter(name='mann_test/sbp_me_std')
    test_dbp_me_stds = AverageMeter(name='mann_test/dbp_me_std')
    test_map_me_stds = AverageMeter(name='mann_test/map_me_std')
    
    # Output collection
    all_test_outputs = np.empty((0, 3), dtype=float)
    all_test_targets = np.empty((0, 3), dtype=float)
    
    tasks_done = 0
    data_iter = iter(test_dataloader)
    
    with torch.no_grad():
        while True:
            if n_tasks_eval is not None and tasks_done >= n_tasks_eval:
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            (Xs, Ys), (Xq, Yq), pid = batch
            meta_batch = Xs.shape[0]
            
            for t in range(meta_batch):
                # Extract and normalize data
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()
                
                # Learn from support set
                _ = mann_model(sX, update_memory=True, targets=sY)
                
                # Test on query set
                query_pred = mann_model(qX, update_memory=False)
                
                # Compute loss
                if config['criterion'] == 'MSELoss':
                    test_loss = F.mse_loss(query_pred, qY)
                elif config['criterion'] == 'SmoothL1Loss':
                    test_loss = F.smooth_l1_loss(query_pred, qY)
                else:
                    raise ValueError("Invalid criterion")
                
                # Record outputs
                all_test_outputs = np.concatenate((all_test_outputs, query_pred.detach().cpu().numpy()), axis=0)
                all_test_targets = np.concatenate((all_test_targets, qY.detach().cpu().numpy()), axis=0)
                
                # Compute metrics
                metric_values = get_metric_values(test_loss, query_pred, qY, config)
                
                test_losses.update(metric_values['loss'], qX.size(0))
                test_sbp_maes.update(metric_values['sbp_mae'], qX.size(0))
                test_dbp_maes.update(metric_values['dbp_mae'], qX.size(0))
                test_map_maes.update(metric_values['map_mae'], qX.size(0))
                test_sbp_mes.update(metric_values['sbp_me'], qX.size(0))
                test_dbp_mes.update(metric_values['dbp_me'], qX.size(0))
                test_map_mes.update(metric_values['map_me'], qX.size(0))
                test_sbp_mae_stds.update(metric_values['sbp_mae_std'], qX.size(0))
                test_dbp_mae_stds.update(metric_values['dbp_mae_std'], qX.size(0))
                test_map_mae_stds.update(metric_values['map_mae_std'], qX.size(0))
                test_sbp_me_stds.update(metric_values['sbp_me_std'], qX.size(0))
                test_dbp_me_stds.update(metric_values['dbp_me_std'], qX.size(0))
                test_map_me_stds.update(metric_values['map_me_std'], qX.size(0))
            
            tasks_done += 1
    
    # Log final test metrics
    if writer:
        writer.add_scalar('mann_test/loss_final', test_losses.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/sbp_mae_final', test_sbp_maes.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/dbp_mae_final', test_dbp_maes.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/map_mae_final', test_map_maes.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/sbp_me_final', test_sbp_mes.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/dbp_me_final', test_dbp_mes.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/map_me_final', test_map_mes.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/sbp_mae_std_final', test_sbp_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/dbp_mae_std_final', test_dbp_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/map_mae_std_final', test_map_mae_stds.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/sbp_me_std_final', test_sbp_me_stds.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/dbp_me_std_final', test_dbp_me_stds.avg, config['max_training_epochs'])
        writer.add_scalar('mann_test/map_me_std_final', test_map_me_stds.avg, config['max_training_epochs'])
    
    return all_test_targets, all_test_outputs


def supcon_loss_old(features, config, labels=None, mask=None):
    r"""
    Supervised Contrastive Loss
    from https://github.com/pulp-platform/fscil/blob/main/code/lib/torch_blocks.py#L114
    and from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#SimCLR-implementation
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

def supcon_loss(features, config, labels=None, mask=None):
    """
    Improved Supervised Contrastive Loss / SimCLR Loss
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
        pos_mask = torch.zeros((batch_size * contrast_count, batch_size * contrast_count), 
                              dtype=torch.float, device=device)
        
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
        model.train() 
        set_trainable_parameters(model=model, tune='all', config=config) 
        
        train_losses = AverageMeter(name='train/loss')
        train_msr_losses = AverageMeter(name='train/msr_loss')
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            # Move data to device
            original_input_signal = batch['input_signal'].to(device) 
            mask_indices = batch['mask_indices'].to(device)             

            optimizer.zero_grad() 

            # 1. Masked Signal Reconstruction (MSR)
            msr_loss = torch.tensor(0.0, device=device)
            if config['lambda_msr'] != 0.:
                # MSR input now uses assigned augmented signal
                # Create the masked input from the chosen augmented signal
                masked_input_for_msr = create_masked_signal(original_input_signal, mask_indices)
                masked_embedding = model.encode(masked_input_for_msr)
                reconstructed_signal = model.reconstruct(masked_embedding)

                # Calculate loss only on masked positions against the chosen augmented target
                if config['criterion'] == 'MSELoss':
                    msr_loss = F.mse_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], original_input_signal[mask_indices.unsqueeze(-1).expand_as(original_input_signal)])
                elif config['criterion'] == 'SmoothL1Loss':
                    msr_loss = F.smooth_l1_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], original_input_signal[mask_indices.unsqueeze(-1).expand_as(original_input_signal)])
                else:
                    raise ValueError("Invalid criterion ...")

            # --- Total Loss ---
            loss = config['lambda_msr'] * msr_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch only for cosine-warmup scheduler
            if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
                scheduler.step()

            # Log training losses
            train_losses.update(loss.item(), original_input_signal.size(0))
            train_msr_losses.update(msr_loss.item(), original_input_signal.size(0))
            
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
                
                original_input_signal = batch['input_signal'].to(device)
                mask_indices = batch['mask_indices'].to(device)

                # --- Forward Passes and Loss Calculations for Each Task (Validation) ---
                val_msr_loss = torch.tensor(0.0, device=device)
                if config['lambda_msr'] != 0.:
                    # MSR input now uses assigned augmented signal in validation
                    val_masked_input_for_msr = create_masked_signal(original_input_signal, mask_indices)
                    masked_embedding = model.encode(val_masked_input_for_msr)
                    reconstructed_signal = model.reconstruct(masked_embedding)
                    
                    if config['criterion'] == 'MSELoss':
                        val_msr_loss = F.mse_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], original_input_signal[mask_indices.unsqueeze(-1).expand_as(original_input_signal)])
                    elif config['criterion'] == 'SmoothL1Loss':
                        val_msr_loss = F.smooth_l1_loss(reconstructed_signal[mask_indices.unsqueeze(-1).expand_as(reconstructed_signal)], original_input_signal[mask_indices.unsqueeze(-1).expand_as(original_input_signal)])
                    else:
                        raise ValueError("Invalid criterion ...")
                
                # Total validation loss
                val_loss = config['lambda_msr'] * val_msr_loss

                # Update validation meters
                val_losses.update(val_loss.item(), original_input_signal.size(0))
                val_msr_losses.update(val_msr_loss.item(), original_input_signal.size(0))

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
            pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
            mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            resp=config['resp'],
            sig2sig=config['sig2sig'],
            min_subject_sample_number=config['min_subject_sample_number']
        )

    # Get train/val/test samplers and build the dataloaders for linear probing
    (lp_train_sampler, lp_val_sampler, lp_test_sampler) = lp_dataset.get_pretraining_samplers()

    lp_train_dataloader = DataLoader(lp_dataset, sampler=lp_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    lp_val_dataloader = DataLoader(lp_dataset, sampler=lp_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    lp_test_dataloader = DataLoader(lp_dataset, sampler=lp_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)

    # Freeze encoder, train remaining parameters (depending on the config['tune'] setting)
    set_trainable_parameters(model=model, tune=config['tune'], config=config)
    
    # Print trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params} / {trainable_params + non_trainable_params} ({trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params} / {trainable_params + non_trainable_params} ({non_trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")

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
        model.train() 
        set_trainable_parameters(model=model, tune=config['tune'], config=config)
        train_lp_losses = AverageMeter(name='lp_train/loss')
        for batch_idx, batch in enumerate(lp_train_dataloader):
            
            # Move data to device
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
        model.eval() 
        set_trainable_parameters(model=model, tune='none', config=config) 
        
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
    
    