import os
import sys
folders_to_add = ['data', 'training_utils', 'models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import numpy as np
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from data.dataset import PhysioDataset
from data.meta_dataloaders import build_meta_splits_and_loaders
from BIOT import MAMLLearner, BPRegressor
from training_utils.helpers import save_status, load_status, set_trainable_parameters, get_model_architecture, get_meta_lr, get_inner_lr, get_inner_steps, build_inner_optimizer, get_backbone_and_head_params, linear_warmup, set_requires_grad_safe, supcon_loss
from training_utils.metrics import AverageMeter, get_metric_values, call_metric


def pretraining_training_validation_testing(save_name, checkpoint_path, tensorboard_path, model_name, config, device):


    # Logging to TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    if config['meta_learning']:
        if config['meta_algorithm'] == 'maml':
            #pre_training(
            #    save_name,
            #    checkpoint_path,
            #    writer,
            #    model_name,
            #    config,
            #    device
            #)
            maml_meta_training(
                save_name,
                checkpoint_path,
                writer,
                model_name,
                config,
                device
            )
        else:
            raise ValueError('Unsupported Meta-Learning algorithm')
    else:
        raise ValueError('Only Meta-Learning is currently supported')


def pre_training(save_name, checkpoint_path, writer, model_name, config, device):
    
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

    # N.B. the field in_channels is common to all models
    example_input_array_graph = torch.rand((config['batch_size'], int(config['input_seq_len_s'] * config['fs']), model.in_channels))
    writer.add_graph(model, example_input_array_graph.to(device))

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
        print(f"[Pretraining] Loaded pretrained weights (all) from {config['pretrained_path']}")

    model = model.to(device)
    
    # ---- Pre-training task ----
    pretrain_ds = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        sig2sig=config['sig2sig'],
        min_subject_sample_number=config['min_subject_sample_number'],
        contrastive=False
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

    # ====== Common config ======
    base_lr = config.get('pre_train_lr')
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
            set_requires_grad_safe(model.encoder, False, original_trainable, name_prefix="encoder")
    if hasattr(model, 'channel_proj'):
        set_requires_grad_safe(model.channel_proj, True, original_trainable, name_prefix="channel_proj")
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad] + [p for p in bp_regressor.parameters() if p.requires_grad]
    optimizer_stage1 = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=stage1_epochs)

    print(f"[Pretraining] ==== Stage 1 ====")
    print(f"\t- Update: initial conv and regressor only")
    print(f"\t- Stage 1 epochs: {stage1_epochs}")
    print(f"\t- Base LR: {base_lr}")
    print(f"\t- Scheduler: CosineAnnealingLR")
    
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
            targets = targets.to(device)

            optimizer_stage1.zero_grad()
            
            feats = model(signals)
            outputs = bp_regressor(feats)

            loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
            
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
                targets = targets.to(device)
                
                feats = model(signals)
                outputs = bp_regressor(feats)
                
                vloss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)

                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('pre_train_stage_1/val_loss_epoch', val_losses.avg, epoch)
        print(f"[Pretraining][Stage 1] Epoch {epoch+1}/{stage1_epochs} - Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            print(f"[Pretraining][Stage 1] New best validation loss ✅: {val_losses.avg:.6f}")
            save_status(None, epoch, model_name + "encoder_ft_stage1", save_name, model, optimizer_stage1, scheduler_stage1, val_losses, checkpoint_path, config)
            save_status(None, epoch, model_name + "regressor_ft_stage1", save_name, bp_regressor, optimizer_stage1, scheduler_stage1, val_losses, checkpoint_path, config)
            
            best_val = val_losses.avg
        
    print(f"[Pretraining] ==== Stage 1 completed ====")
    print(f"\t- Best val loss {best_val}")

    # ====== Stage 2: Backbone + Head ======
    # Instantiate the dataset again, but this time with the contrastive flag
    pretrain_ds = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        sig2sig=config['sig2sig'],
        min_subject_sample_number=config['min_subject_sample_number'],
        contrastive=(config['lambda_contrastive'] > 0.0)
    )
        
    (pre_train_sampler, pre_val_sampler, pre_test_sampler) = pretrain_ds.get_pretraining_samplers()
    
    pre_train_dataloader = DataLoader(pretrain_ds, sampler=pre_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)    
    pre_valid_dataloader = DataLoader(pretrain_ds, sampler=pre_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    pre_test_dataloader = DataLoader(pretrain_ds, sampler=pre_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    
    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name + "encoder_ft_stage1", save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    print(f"[Pretraining] Loaded encoder pretrained weights after pre-training stage 1")
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    load_status(None, model_name + "regressor_ft_stage1", save_name, bp_regressor, None, None, checkpoint_path, config)
    bp_regressor = bp_regressor.to(device)
    print(f"[Pretraining] Loaded regressor pretrained weights after pre-training stage 1")
    
    lr_backbone = base_lr * backbone_lr_mul
    set_requires_grad_safe(model.encoder, True, original_trainable, name_prefix="encoder")
    set_requires_grad_safe(model.channel_proj, True, original_trainable, name_prefix="channel_proj")

    backbone_params, head_params = get_backbone_and_head_params(model, bp_regressor)
    backbone_params = [p for p in backbone_params if p.requires_grad]
    head_params = [p for p in head_params if p.requires_grad]

    optimizer_stage2 = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': base_lr}
    ], weight_decay=weight_decay)
    scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=stage2_epochs)

    print(f"[Pretraining] ==== Stage 2: fine-tune backbone, {stage2_epochs} epochs, head_lr={base_lr}, backbone_lr={lr_backbone} ====")
    print(f"\t- Update: all model parameters and regressor")
    print(f"\t- Stage 1 epochs: {stage2_epochs}")
    print(f"\t- Base LR: {base_lr}")
    print(f"\t- Backbone LR: {lr_backbone}")
    print(f"\t- Scheduler: CosineAnnealingLR")
    
    best_val = float("+inf")    
    sup_loss_weight = config.get('lambda_supervised')
    con_loss_weight = config.get('lambda_contrastive')

    for epoch in range(stage2_epochs):
        model.train()
        bp_regressor.train()
        
        # Warmup LRs
        for i, pg in enumerate(optimizer_stage2.param_groups):
            target_lr = base_lr if i == 1 else lr_backbone
            pg['lr'] = linear_warmup(epoch, warmup_stage2, target_lr)

        train_losses = AverageMeter(name='pre_train_stage_2/train_loss')
        for batch_idx, batch in enumerate(pre_train_dataloader):
            
            if con_loss_weight > 0.0:
                # Since pos_signals/annotation are the signals/annotations of another sample from the same subject, the subject_ids are the same
                (signals, targets, bp_cats_anchor), (pos_signals, pos_targets, bp_cats_pos) = batch
                
                # Move to device
                signals = signals.to(device)
                pos_signals = pos_signals.to(device)
                bp_cats_anchor = bp_cats_anchor.to(device)
                
                targets = targets.to(device)
                pos_targets = pos_targets.to(device)
                
                # Handle dimensions
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                if len(pos_signals.shape) == 2:
                    pos_signals = pos_signals.unsqueeze(-1)
                
            else:
                signals, targets = batch
                signals = signals.to(device)
                targets = targets.to(device)
                
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)

            optimizer_stage2.zero_grad()
            
            if con_loss_weight > 0.0:
                # Forward pass for both anchor and positive samples
                feats_anchor = model(signals)           # [B, D]
                feats_positive = model(pos_signals)     # [B, D]
                
                # Stack features for contrastive learning: [B, 2, D]
                feats_contrastive = torch.stack([feats_anchor, feats_positive], dim=1)
                
                # Use anchor features for regression
                outputs = bp_regressor(feats_anchor)    # [B, 3]
                
                # --- Supervised regression loss ---
                sup_loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
                sup_loss = sup_loss * sup_loss_weight
                
                # --- Supervised contrastive loss ---
                con_loss, _, _ = supcon_loss(feats_contrastive, config, labels=bp_cats_anchor)
                con_loss = con_loss * con_loss_weight
                
            else:
                # Standard forward pass without contrastive learning
                feats = model(signals)                  # [B, D]
                outputs = bp_regressor(feats)           # [B, 3]
                
                # --- Supervised regression loss ---
                sup_loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
                sup_loss = sup_loss * sup_loss_weight
                con_loss = 0.0
            
            # Total loss
            total_loss = sup_loss + con_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            torch.nn.utils.clip_grad_norm_(bp_regressor.parameters(), max_norm=grad_clip)
            optimizer_stage2.step()

            train_losses.update(total_loss.item(), signals.size(0))
            writer.add_scalar('pre_train_stage_2/train_loss', total_loss.item(), epoch * len(pre_train_dataloader) + batch_idx)
            if con_loss_weight > 0.0:
                writer.add_scalar('pre_train_stage_2/supervised_loss', sup_loss.item(), epoch * len(pre_train_dataloader) + batch_idx)
                writer.add_scalar('pre_train_stage_2/contrastive_loss', con_loss.item(), epoch * len(pre_train_dataloader) + batch_idx)
        writer.add_scalar('pre_train_stage_2/train_loss_epoch', train_losses.avg, epoch)
        scheduler_stage2.step()

        # --- Validation (NO contrastive loss, no subject_ids) ---
        model.eval()
        bp_regressor.eval()
        val_losses = AverageMeter(name='pre_train_stage_2/val_loss')
        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_valid_dataloader):
                signals, targets = batch
                signals = signals.to(device)
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                targets = targets.to(device)
                
                feats = model(signals)
                outputs = bp_regressor(feats)
                
                vloss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('pre_train_stage_2/val_loss_epoch', val_losses.avg, epoch)
        print(f"[Pretraining][Stage 2] Epoch {epoch+1}/{stage2_epochs} "
            f"- Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            print(f"[Pretraining][Stage 2] New best validation loss ✅: {val_losses.avg:.6f}")
            save_status(None, epoch, model_name + "encoder_ft_stage2", save_name, model, optimizer_stage2, scheduler_stage2, val_losses, checkpoint_path, config)
            save_status(None, epoch, model_name + "regressor_ft_stage2", save_name, bp_regressor, optimizer_stage2, scheduler_stage2, val_losses, checkpoint_path, config)
            
            best_val = val_losses.avg
    
    print(f"[Pretraining] ==== Stage 2 completed ====")
    print(f"\t- Best val loss {best_val}")
    
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
    
    if config['lambda_contrastive'] > 0.0:
        # Record features and bp categories for silhouette score
        all_feats = np.empty((0, feat_dim), dtype=float)
        all_bp_cats = np.empty((0, 1), dtype=float)
    
    model.eval()
    bp_regressor.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(pre_test_dataloader):
            
            if config['lambda_contrastive'] > 0.0:
                # Since pos_signals/annotation are the signals/annotations of another sample from the same subject, the subject_ids are the same
                signals, targets, bp_cats = batch
                
                bp_cats = bp_cats.to(device)
            else:
                signals, targets = batch 
            
            signals = signals.to(device)
            
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            
            targets = targets.to(device)
            
            feats = model(signals)
            outputs = bp_regressor(feats)
            
            # Supervised loss (for metric logging)
            test_loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets) 
            
            # Record predictions and ground truths
            all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
            all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)
            
            if config['lambda_contrastive'] > 0.0:
                # Record normalized features and bp categories for silhouette score
                all_feats = np.concatenate((all_feats, F.normalize(feats, dim=1).detach().cpu().numpy()), axis=0) # Normalized embeddings
                all_bp_cats = np.concatenate((all_bp_cats, bp_cats.unsqueeze(-1).detach().cpu().numpy()), axis=0)

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
    
    if config['lambda_contrastive'] > 0.0:
        sil_score = silhouette_score(all_feats, all_bp_cats.squeeze())
        print(f"[Pretraining][Stage 2] Embedding Silhouette Score by BP phenotype: {sil_score:.4f}")
    
    # Log test metrics (optionally, plot them) and return loss for validation
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'pretraining_supervised_stage'), plot=True)    
    
    print(f"[Pretraining] ==== Stage 3 completed ====")          
    

def maml_meta_training(save_name, checkpoint_path, writer, model_name, config, device):
    
    # Load best model after pre-training
    model = get_model_architecture(config)
    #load_status(None, model_name + "encoder_ft_stage2", save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    print(f"[MAML] Loaded encoder pretrained weights after pre-training stage 2")
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    #load_status(None, model_name + "regressor_ft_stage2", save_name, bp_regressor, None, None, checkpoint_path, config)
    bp_regressor = bp_regressor.to(device)
    print(f"[MAML] Loaded regressor pretrained weights after pre-training stage 2")
    
    # Create enhanced learner wrapper
    learner = MAMLLearner(model, bp_regressor).to(device)
    
    # Configure functional forward approach based on your preference
    learner.use_pure_functional = config.get('use_pure_functional')  # Set to True for memory efficiency
    
    # ==== Meta-learning (MAML) Stage ====
    
    # ---- Setup ----
    
    _, train_dataloader, val_dataloader, test_dataloader, _ = build_meta_splits_and_loaders(
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        seed=config['seed'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        sig2sig=config['sig2sig'],
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],     # IMPORTANT for meta-learning to test on unseen subjects, must be False
        k_support=config['k_support'],
        k_query=config['k_query'],
        meta_batch_size=config['meta_batch_size'],
        contrastive=False # To avoid loading the positive pair for a subject
    )
    
    # Hyperparams / defaults
    meta_epochs = config.get('max_meta_epochs')
    grad_clip_norm = config.get('grad_clip')  # Increased for MAML
    
    # Enhanced scheduling parameters
    meta_lr_schedule = config.get('meta_lr_schedule')
    inner_lr_schedule = config.get('inner_lr_schedule')
    inner_steps_schedule = config.get('inner_steps_schedule')
    
    # MAML specific parameters
    second_order = config.get('second_order_maml')
    
    print(f"[MAML] ==== Stage 3 Starting MAML meta-training ====")
    print(f"\t- Meta LR schedule: {meta_lr_schedule}")
    print(f"\t- Inner LR schedule: {inner_lr_schedule}")
    print(f"\t- Inner steps schedule: {inner_steps_schedule}")
    print(f"\t- Second-order gradients: {second_order}")
    print(f"\t- Using pure functional: {learner.use_pure_functional}")
    
    # Meta-optimizer for the meta-parameters
    meta_optimizer = torch.optim.Adam(learner.parameters(), lr=get_meta_lr(0, config))
    
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
                
        # End epoch: run validation on val tasks
        val_loss = evaluate_maml_with_bp_metrics(
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
            print(f"[MAML] New best validation loss ✅: {val_loss:.6f}, saved to {best_model_path}")
    
    
    # After meta-training, run final test evaluation
    print(f"[MAML] Starting final test evaluation with best model...")
    
    # Load best model
    best_ckpt = torch.load(os.path.join(checkpoint_path, f"{save_name}_best_maml"), weights_only=False)
    learner.load_state_dict(best_ckpt['learner_state_dict'])
    learner = learner.to(device)
    
    all_test_targets, all_test_outputs = evaluate_maml_with_bp_metrics(
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
    
    
def evaluate_maml_with_bp_metrics(learner, dataloader, device, config, test=False, writer=None):
    """
    Evaluation function adapted for MAML learner
    """
    learner.eval()
    
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
            sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
            qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()

            # --- Build adapted copy of learner ---
            # Deepcopy to avoid polluting outer model
            adapted_model = copy.deepcopy(learner.model).to(device)
            adapted_regressor = copy.deepcopy(learner.regressor).to(device)

            # Build inner optimizer with correct parameter groups & LRs
            inner_opt = build_inner_optimizer(
                adapted_model, adapted_regressor,
                base_lr=config['eval_lr'],
                config=config
            )

            adapted_model.train()
            adapted_regressor.train()

            # --- Inner loop adaptation ---
            for step in range(config['eval_steps']):
                out_s = adapted_regressor(adapted_model(sX))
                loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)

                inner_opt.zero_grad()
                loss_s.backward()

                # Optional gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(adapted_model.parameters()) + list(adapted_regressor.parameters()),
                    config['grad_clip']
                )

                inner_opt.step()

            # --- Query evaluation ---
            adapted_model.eval()
            adapted_regressor.eval()
            with torch.no_grad():
                out_q = adapted_regressor(adapted_model(qX))
                loss_q = F.smooth_l1_loss(out_q, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q, qY)
                
                metric_values = get_metric_values(loss_q, out_q, qY, config)

                if test:
                    all_test_outputs = np.concatenate(
                        (all_test_outputs, out_q.detach().cpu().numpy()), axis=0
                    )
                    all_test_targets = np.concatenate(
                        (all_test_targets, qY.detach().cpu().numpy()), axis=0
                    )

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

                del adapted_model
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
    

