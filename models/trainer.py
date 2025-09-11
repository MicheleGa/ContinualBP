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
from BIOT import MAMLLearner, BIOTPredictionHead
from training_utils.helpers import (
    save_status, load_status, get_model_architecture, get_meta_lr, get_inner_lr, get_inner_steps, 
    build_inner_optimizer, supcon_loss, get_msl_weights, 
    should_use_second_order, compute_embedding_stats
    )
from training_utils.metrics import AverageMeter, get_metric_values, call_metric


def pretraining_training_validation_testing(save_name, checkpoint_path, tensorboard_path, model_name, config, device):


    # Logging to TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    if config['meta_learning']:
        if config['meta_algorithm'] == 'maml':
            pre_training(
                save_name,
                checkpoint_path,
                writer,
                model_name,
                config,
                device
            )
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
        contrastive=True
    )
    
    (pre_train_sampler, pre_val_sampler, pre_test_sampler) = pretrain_ds.get_pretraining_samplers()
    
    pre_train_dataloader = DataLoader(pretrain_ds, sampler=pre_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)    
    pre_valid_dataloader = DataLoader(pretrain_ds, sampler=pre_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    pre_test_dataloader = DataLoader(pretrain_ds, sampler=pre_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    
    # ====== Common config ======
    base_lr = config.get('pre_train_lr')
    weight_decay = config.get('weight_decay')

    stage1_epochs = config.get('ft_stage1_epochs')
    stage2_epochs = config.get('ft_stage2_epochs')

    # ====== Stage 1: Backbone only ======
    # Some models like transformers may have an index for the positional embedding that is not a parameter and may raise an error if set to requires_grad
    backbone_trainable_params = [p for p in model.parameters() if (p.is_floating_point() or p.is_complex())]
    for p in backbone_trainable_params:
        p.requires_grad = True
    
    # Pass only backbone parameters
    optimizer_stage1 = torch.optim.AdamW(backbone_trainable_params, lr=base_lr, weight_decay=weight_decay)
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=max(1, stage1_epochs))

    print(f"[Pretraining] ==== Stage 1 ====")
    print(f"\t- Update: encoder only")
    print(f"\t- Stage 1 epochs: {stage1_epochs}")
    print(f"\t- Base LR: {base_lr}")
    print(f"\t- Scheduler: CosineAnnealingLR")
    
    # Define feat_dim before validation silhouette computation
    feat_dim = model.embed_dim
    train_losses = AverageMeter(name='pre_train_stage_1/train_loss')
    best_val = -1.0 # Silhouette score is the worse when it is close to -1 and good when it is close to +1 
    for epoch in range(stage1_epochs):
        model.train()

        train_losses.reset()
        for batch_idx, batch in enumerate(pre_train_dataloader):
            # Since signals_anchor/signals_pos are the same window fo PPG/ECG, the bp_cats_anchor/bp_cats_pos are the same
            (signals_anchor, bp_cats_anchor), (signals_pos, bp_cats_pos) = batch
            
            # Move to device
            signals_anchor, signals_pos, bp_cats_anchor = signals_anchor.to(device), signals_pos.to(device), bp_cats_anchor.to(device)
            
            # Handle dimensions (may happen that in PPG only mode a dimension is squeezed out)
            if len(signals_anchor.shape) == 2:
                signals_anchor = signals_anchor.unsqueeze(-1)
            if len(signals_pos.shape) == 2:
                signals_pos = signals_pos.unsqueeze(-1)

            # zero gradients
            optimizer_stage1.zero_grad()

            # Forward pass for both anchor and positive samples
            feats_anchor = model(signals_anchor)           # [B, D]
            feats_positive = model(signals_pos)     # [B, D]
            
            # Stack features for contrastive learning: [B, 2, D]
            feats_contrastive = torch.stack([feats_anchor, feats_positive], dim=1)
            
            # Supervised contrastive loss
            loss, _, _ = supcon_loss(feats_contrastive, config, labels=bp_cats_anchor)
            
            loss.backward()

            # Optimizer step
            optimizer_stage1.step()

            train_losses.update(loss.item(), signals_anchor.size(0))
            writer.add_scalar('pre_train_stage_1/train_loss', loss.item(), epoch * len(pre_train_dataloader) + batch_idx)

        # Scheduler step
        scheduler_stage1.step()
        
        # Log loss
        writer.add_scalar('pre_train_stage_1/train_loss_epoch', train_losses.avg, epoch)

        # Validation
        model.eval()
        
        # Record features and bp categories for silhouette score
        all_feats = np.empty((0, feat_dim), dtype=float)
        all_bp_cats = np.empty((0, 1), dtype=float)
        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_valid_dataloader):
                signals, _, bp_cats = batch
                signals, bp_cats = signals.to(device), bp_cats.to(device)
                
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                
                feats = model(signals)

                # Record normalized features and bp categories for silhouette score
                all_feats = np.concatenate((all_feats, F.normalize(feats, dim=1).detach().cpu().numpy()), axis=0) # Normalized embeddings
                all_bp_cats = np.concatenate((all_bp_cats, bp_cats.unsqueeze(-1).detach().cpu().numpy()), axis=0)


        sil_score = silhouette_score(all_feats, all_bp_cats.squeeze()).astype(np.float32)
        writer.add_scalar('pre_train_stage_1/val_sil_score_epoch', sil_score, epoch)
        print(f"[Pretraining][Stage 1] Epoch {epoch+1}/{stage1_epochs} - Train {train_losses.avg:.6f} Val Silhouette Score {sil_score:.6f}")

        if sil_score > best_val:
            print(f"[Pretraining][Stage 1] New best validation silhouette score ✅: {sil_score:.6f}")
            save_status(None, epoch, model_name + "encoder_ft_stage1", save_name, model, optimizer_stage1, scheduler_stage1, sil_score, checkpoint_path, config)
            
            best_val = sil_score
        
    print(f"[Pretraining] ==== Stage 1 completed ====")
    print(f"\t- Best silhouette score {best_val}")

    # ====== Stage 2: Head-only ======
    # Instantiate the dataset again, but this time without the contrastive flag
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
    
    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name + "encoder_ft_stage1", save_name, model, None, None, checkpoint_path, config)
    model.eval()
    model = model.to(device)
    print(f"[Pretraining] Loaded encoder pretrained weights after pre-training stage 1")
    
    # Initialize prediction head
    feat_dim = model.embed_dim
    output_dim = config['input_seq_len_s'] * config['fs'] if config['sig2sig'] else 3  # Full waveform or SBP/DBP/MAP
    bp_prediction_head = BIOTPredictionHead(feat_dim, output_dim, depth=config['num_decoder_layers'], heads=config['num_heads'])
    bp_prediction_head.eval()
    bp_prediction_head = bp_prediction_head.to(device)
    print(f"[Pretraining] Intialized prediction head weights after pre-training stage 1")
    
    # Freeze backbone, unfreeze head
    backbone_trainable_params = [p for p in model.parameters() if (p.is_floating_point() or p.is_complex())]
    for p in backbone_trainable_params:
        p.requires_grad = False
    
    head_trainable_params = [p for p in bp_prediction_head.parameters() if (p.is_floating_point() or p.is_complex())]
    for p in head_trainable_params:
        p.requires_grad = True
        
    # Pass only head parameters
    optimizer_stage2 = torch.optim.AdamW(head_trainable_params, lr=base_lr, weight_decay=weight_decay)
    scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=max(1, stage2_epochs))

    print(f"[Pretraining] ==== Stage 2: fine-tune regressor, {stage2_epochs} epochs, head_lr={base_lr} ====")
    print(f"\t- Update: head only")
    print(f"\t- Stage 1 epochs: {stage2_epochs}")
    print(f"\t- Base LR: {base_lr}")
    print(f"\t- Scheduler: CosineAnnealingLR")
    
    best_val = float("+inf")    
    train_losses = AverageMeter(name='pre_train_stage_2/train_loss')
    val_losses = AverageMeter(name='pre_train_stage_2/val_loss')
    for epoch in range(stage2_epochs):
        bp_prediction_head.train()
        
        train_losses.reset()
        for batch_idx, batch in enumerate(pre_train_dataloader):
            
            signals, targets = batch
            signals, targets = signals.to(device), targets.to(device)
                
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)

            # Zero gradients
            optimizer_stage2.zero_grad()

            # Standard forward pass without contrastive learning
            feats = model(signals)                  # [B, D]
            outputs = bp_prediction_head(feats)     # [B, 3] or [B, 1250]
                
            # Supervised regression loss
            loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
            
            loss.backward()

            # Optimizer step
            optimizer_stage2.step()
            
            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('pre_train_stage_2/train_loss', loss.item(), epoch * len(pre_train_dataloader) + batch_idx)
        
        # Step scheduler once per epoch
        scheduler_stage2.step()
        
        writer.add_scalar('pre_train_stage_2/train_loss_epoch', train_losses.avg, epoch)
        
        # Validation
        bp_prediction_head.eval()
        val_losses.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_valid_dataloader):
                signals, targets = batch
                signals, targets = signals.to(device), targets.to(device)
                
                if len(signals.shape) == 2:
                    signals = signals.unsqueeze(-1)
                
                feats = model(signals)
                outputs = bp_prediction_head(feats)
                
                vloss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('pre_train_stage_2/val_loss_epoch', val_losses.avg, epoch)
        print(f"[Pretraining][Stage 2] Epoch {epoch+1}/{stage2_epochs} "
            f"- Train {train_losses.avg:.6f} Val {val_losses.avg:.6f}")

        if val_losses.avg < best_val:
            print(f"[Pretraining][Stage 2] New best validation loss ✅: {val_losses.avg:.6f}")
            save_status(None, epoch, model_name + "head_ft_stage2", save_name, bp_prediction_head, optimizer_stage2, scheduler_stage2, val_losses, checkpoint_path, config)
            
            best_val = val_losses.avg
    
    print(f"[Pretraining] ==== Stage 2 completed ====")
    print(f"\t- Best val loss {best_val}")
    print(f"[Pretraining] ==== Testing ====")
    
    # ====== Final test ======
    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name + "encoder_ft_stage1", save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    
    # Load best prediction head
    feat_dim = model.embed_dim
    output_dim = config['input_seq_len_s'] * config['fs'] if config['sig2sig'] else 3  # Full waveform or SBP/DBP/MAP
    bp_prediction_head = BIOTPredictionHead(feat_dim, output_dim, depth=config['num_decoder_layers'], heads=config['num_heads'])
    load_status(None, model_name + "head_ft_stage2", save_name, bp_prediction_head, None, None, checkpoint_path, config)
    bp_prediction_head.eval()
    bp_prediction_head = bp_prediction_head.to(device)
    
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
    all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float) if config['sig2sig'] else np.empty((0, 3), dtype=float) # Assuming SBP/DBP/MAP output shape is 3
    all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float) if config['sig2sig'] else np.empty((0, 3), dtype=float)
        
    model.eval()
    bp_prediction_head.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(pre_test_dataloader):
            signals, targets = batch
            signals, targets = signals.to(device), targets.to(device)
            
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)
            
            feats = model(signals)
            outputs = bp_prediction_head(feats)
            
            # Supervised loss (for metric logging)
            test_loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets) 
            
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
    
    print(f"[Pretraining] ==== Testing completed ====")


def maml_meta_training(save_name, checkpoint_path, writer, model_name, config, device):
    
    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name + "encoder_ft_stage1", save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    print(f"[MAML] Loaded encoder pretrained weights after pre-training stage 1")
    
    # Load best prediction head
    feat_dim = model.embed_dim
    output_dim = config['input_seq_len_s'] * config['fs'] if config['sig2sig'] else 3  # Full waveform or SBP/DBP/MAP
    bp_prediction_head = BIOTPredictionHead(feat_dim, output_dim, depth=config['num_decoder_layers'], heads=config['num_heads'])
    load_status(None, model_name + "head_ft_stage2", save_name, bp_prediction_head, None, None, checkpoint_path, config)
    bp_prediction_head.eval()
    bp_prediction_head = bp_prediction_head.to(device)
    print(f"[MAML] Loaded regressor pretrained weights after pre-training stage 2")
    
    # Create enhanced learner wrapper
    learner = MAMLLearner(model, bp_prediction_head).to(device)
    
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
    
    # Saving feature statistics for intiializing the drift detector during the personalization step
    print("[MAML] Computing baseline embedding statistics before meta-training...")
    baseline_mean, baseline_std = compute_embedding_stats(model, train_dataloader, device)

    # Save to file for personalization later on
    stats_ckpt = {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
    }
    np.savez(os.path.join(checkpoint_path, f"{save_name}_embedding_stats.npz"), **stats_ckpt)
    print(f"[MAML] Saved embedding stats to {save_name}_embedding_stats.npz")
    
    # Hyperparams / defaults
    meta_epochs = config.get('max_meta_epochs')
    grad_clip_norm = config.get('grad_clip')  # Increased for MAML
    
    # Enhanced scheduling parameters
    meta_lr_schedule = config.get('meta_lr_schedule')
    inner_lr_schedule = config.get('inner_lr_schedule')
    inner_steps_schedule = config.get('inner_steps_schedule')
    
    # MAML specific parameters
    print(f"[MAML] ==== Stage 3 Starting MAML meta-training ====")
    print(f"\t- Meta LR schedule: {meta_lr_schedule}")
    print(f"\t- Inner LR schedule: {inner_lr_schedule}")
    print(f"\t- Inner steps schedule: {inner_steps_schedule}")
    print(f"\t- Using pure functional: {learner.use_pure_functional}")
    
    # Meta-optimizer for the meta-parameters
    meta_optimizer = torch.optim.Adam(learner.parameters(), lr=get_meta_lr(0, config))
    
    # Meters
    epoch_query_loss_meter = AverageMeter(name='meta/train_query_loss')
    epoch_support_loss_meter = AverageMeter(name='meta/train_support_loss')
    epoch_meta_loss_meter = AverageMeter(name='meta/train_meta_loss')
    
    best_val = float("+inf")
    global_step = 0

    # ---- Outer loop ----
    for epoch in range(meta_epochs):
        learner.train()

        # Decide schedules / hyperparams for this epoch
        current_meta_lr = get_meta_lr(epoch, config)
        current_inner_lr = get_inner_lr(epoch, config)
        current_inner_steps = get_inner_steps(epoch, config)

        # Update meta-optimizer learning rate if using manual schedule
        for param_group in meta_optimizer.param_groups:
            param_group['lr'] = current_meta_lr

        # Decide derivative-order for this epoch (derivative annealing)
        use_second_order = should_use_second_order(epoch, config)

        # Get MSL weights for post-update steps (and optionally pre-update)
        include_pre = bool(config.get('msl_include_pre'))
        msl_weights = get_msl_weights(epoch, config, current_inner_steps, include_pre=include_pre)
    
        # Reset meters
        epoch_query_loss_meter.reset()
        epoch_support_loss_meter.reset()
        epoch_meta_loss_meter.reset()
        grad_norms = []
        
        # Log current hyperparameters
        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)
        writer.add_scalar('meta/use_second_order', float(use_second_order), epoch)

        for batch_idx, batch in enumerate(train_dataloader):
            (Xs, Ys), (Xq, Yq), pid = batch
            meta_batch = Xs.shape[0]

            meta_optimizer.zero_grad()

            task_query_losses = []
            task_support_losses = []
            task_query_pre_losses = []
            meta_loss = 0.0  # accumulated weighted multi-step meta-loss (averaged over tasks)

            for t in range(meta_batch):
                # Extract task-specific support/query
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                qX, qY = Xq[t].to(device).float(), Yq[t].to(device).float()

                # Query loss before adaptation (step 0) — keep for logging and optional MSL include
                with torch.no_grad():
                    out_q0 = learner(qX)  # Use current meta-parameters
                    loss_q0 = F.smooth_l1_loss(out_q0, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q0, qY)
                task_query_pre_losses.append(loss_q0.item())

                # Initialize fast_weights from current learner parameters **without detaching**.
                # IMPORTANT: keep same cloning/behavior as before to avoid changing other parts.
                # We follow your original approach: start from a clone list of parameters.
                fast_weights = [p.clone() for p in learner.parameters()]
                support_losses = []
                per_step_query_losses = []

                # If include_pre is True, include pre-adapt loss as first entry (so weights align)
                if include_pre:
                    per_step_query_losses.append(loss_q0)

                for step in range(current_inner_steps):
                    # Support forward using the current fast_weights
                    out_s = learner(sX, fast_weights)
                    loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)
                    support_losses.append(loss_s.item())

                    # Select differentiable params from fast_weights
                    opt_idx, opt_params = zip(
                        *[(i, w) for i, w in enumerate(fast_weights)
                        if w.requires_grad and torch.is_floating_point(w)]
                    )

                    # Compute grads for the inner-update. create_graph True iff we want second-order
                    grads = torch.autograd.grad(
                        loss_s,
                        opt_params,
                        create_graph=use_second_order,
                        retain_graph=True,
                        allow_unused=True
                    )

                    # Apply inner update (no clipping)
                    new_fast_weights = []
                    g_iter = iter(grads)
                    for i, w in enumerate(fast_weights):
                        if i in opt_idx:
                            g = next(g_iter)
                            if g is not None:
                                new_fast_weights.append(w - current_inner_lr * g)
                            else:
                                new_fast_weights.append(w)
                        else:
                            new_fast_weights.append(w)

                    fast_weights = new_fast_weights

                    # Evaluate query using the updated fast_weights (multi-step loss)
                    out_q_step = learner(qX, fast_weights)
                    loss_q_step = F.smooth_l1_loss(out_q_step, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q_step, qY)
                    per_step_query_losses.append(loss_q_step)

                    # Log per-inner-step support loss
                    writer.add_scalar(f"inner/support_loss_step{step}", loss_s.item(), global_step)

                # --- combine per-step query losses into single meta-loss for this task ---

                # Weighted sum of query losses for this task
                weighted_task_meta = 0.0
                for wgt, loss_tensor in zip(msl_weights, per_step_query_losses):
                    weighted_task_meta = weighted_task_meta + (wgt * loss_tensor)

                # Average task contribution over the meta-batch
                meta_loss = meta_loss + (weighted_task_meta / float(meta_batch))

                # Logging and meters
                # use the final (post-last-step) query loss for per-task statistics
                final_query_loss = per_step_query_losses[-1].item() if isinstance(per_step_query_losses[-1], torch.Tensor) else float(per_step_query_losses[-1])
                task_query_losses.append(final_query_loss)
                task_support_losses.append(np.mean(support_losses) if len(support_losses) > 0 else 0.0)

                epoch_query_loss_meter.update(final_query_loss, 1)
                epoch_support_loss_meter.update(np.mean(support_losses) if len(support_losses) > 0 else 0.0, 1)

            # End tasks in meta-batch: do meta-backward on accumulated meta_loss
            # Note: if using first-order (use_second_order=False), create_graph was False above,
            # so second-order contributions are omitted (Derivative-order annealing behavior).
            meta_loss.backward()

            # Gradient clipping and meta step
            total_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), grad_clip_norm)
            grad_norms.append(total_norm.item())

            meta_optimizer.step()

            epoch_meta_loss_meter.update(meta_loss.item(), 1)
            global_step += 1

        # ---- After training batches (end of epoch) ----
        avg_train_query_loss = epoch_query_loss_meter.avg
        avg_train_support_loss = epoch_support_loss_meter.avg
        avg_train_meta_loss = epoch_meta_loss_meter.avg
        avg_grad_norm = np.mean(grad_norms)

        # Run validation ONCE
        val_loss = evaluate_maml_with_bp_metrics(
            learner, val_dataloader, device, config
        )

        # ---- Logging ----
        writer.add_scalar('meta/train_query_loss_epoch', avg_train_query_loss, epoch)
        writer.add_scalar('meta/train_support_loss_epoch', avg_train_support_loss, epoch)
        writer.add_scalar('meta/train_meta_loss_epoch', avg_train_meta_loss, epoch)
        writer.add_scalar('meta/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('meta/grad_norm_epoch', avg_grad_norm, epoch)

        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)
        writer.add_scalar('meta/use_second_order', float(use_second_order), epoch)

        print(
            f"[MAML] Epoch {epoch+1}/{meta_epochs} - "
            f"train_support_loss {avg_train_support_loss:.6f}, "
            f"train_query_loss {avg_train_query_loss:.6f}, "
            f"train_meta_loss {avg_train_meta_loss:.6f}, "
            f"val_loss {val_loss:.6f}, "
            f"grad_norm {avg_grad_norm:.3f}, "
            f"meta_lr {current_meta_lr:.6f}, "
            f"inner_lr {current_inner_lr:.6f}, "
            f"inner_steps {current_inner_steps}, "
            f"second_order {use_second_order}"
        )

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
    
        
        
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float) if config['sig2sig'] else np.empty((0, 3), dtype=float) # Assuming SBP/DBP/MAP output shape is 3
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float) if config['sig2sig'] else np.empty((0, 3), dtype=float)
        
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
            adapted_head = copy.deepcopy(learner.head).to(device)

            # Build inner optimizer with correct parameter groups & LRs
            inner_opt = build_inner_optimizer(
                adapted_model, adapted_head,
                base_lr=config['eval_lr'],
                config=config
            )

            adapted_model.train()
            adapted_head.train()

            # --- Inner loop adaptation ---
            for step in range(config['eval_steps']):
                out_s = adapted_head(adapted_model(sX))
                loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)

                inner_opt.zero_grad()
                loss_s.backward()
                inner_opt.step()

            # --- Query evaluation ---
            adapted_model.eval()
            adapted_head.eval()
            with torch.no_grad():
                out_q = adapted_head(adapted_model(qX))
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
                del adapted_head
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
    

