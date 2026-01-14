import os
import sys
folders_to_add = ['data', 'training_utils', 'models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
from itertools import chain
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.dataset import PhysioDataset
from data.meta_dataloaders import build_meta_splits_and_loaders
from models.maml import MAML
from models.component_factory import Model
from training_utils.helpers import (
    save_status, load_status, count_parameters,
    get_encoder_architecture, get_prediction_head_architecture, 
    get_meta_lr, get_inner_lr, get_inner_steps, build_inner_optimizer, 
    get_msl_weights, compute_embedding_stats
    )
from training_utils.metrics import AverageMeter, get_metric_values, call_metric


def pre_training(save_name, checkpoint_path, writer, model_name, config, device):
    
    print(f"[Pretraining][Stage 1] ==== Supervised stage start ====")
    
    # Instantiate the dataset again, but this time without the contrastive flag
    pretrain_ds = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        meta_split_ratio=config['meta_train_split_ratio'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        min_subject_sample_number=config['min_subject_sample_number']
    )
        
    (pre_train_sampler, _, pre_val_sampler, pre_test_sampler) = pretrain_ds.get_pretraining_samplers()
    
    pre_train_dataloader = DataLoader(pretrain_ds, sampler=pre_train_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)    
    pre_valid_dataloader = DataLoader(pretrain_ds, sampler=pre_val_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    pre_test_dataloader = DataLoader(pretrain_ds, sampler=pre_test_sampler, batch_size=config['batch_size'], num_workers=config['loader_worker'], pin_memory=True)
    
    print(f"[Pretraining][Stage 1] Dataset initialized ✅")
    
    # Initialize encoder
    encoder = get_encoder_architecture(config)
    encoder = encoder.to(device)
    print(f"[Pretraining][Stage 1] Encoder weights initialized ✅")
    
    # Initialize prediction head
    prediction_head = get_prediction_head_architecture(config)
    prediction_head = prediction_head.to(device)
    print(f"[Pretraining][Stage 1] Prediction head weights initialized ✅")
    
    # Pretraining config
    stage1_epochs = config.get('stage1_epochs')
    stage1_pre_train_lr = config.get('stage1_pre_train_lr')
    stage1_pre_train_scheduler_eta_min = config.get('stage1_pre_train_scheduler_eta_min')
    weight_decay = config.get('weight_decay')
    
    # Train encoder
    for param in encoder.parameters():
        if (param.is_floating_point() or param.is_complex()):
            param.requires_grad = True

    # Train regression head
    for param in prediction_head.parameters():
        if (param.is_floating_point() or param.is_complex()):
            param.requires_grad = True
    
    # Count trainable params
    encoder_trainable, encoder_non_trainable = count_parameters(encoder)
    head_trainable, head_non_trainable = count_parameters(prediction_head)
    total_trainable = encoder_trainable + head_trainable 
    total_non_trainable = encoder_non_trainable + head_non_trainable

    print(f"[Pretraining][Stage 1] Parameter count:")
    print(f"\t- Encoder: {encoder_trainable:,} trainable, {encoder_non_trainable:,} non-trainable")
    print(f"\t- Prediction head: {head_trainable:,} trainable, {head_non_trainable:,} non-trainable")
    print(f"\t- Total: {total_trainable:,} trainable, {total_non_trainable:,} non-trainable")
    
    # Collect trainable params
    trainable_params = list(filter(lambda p: p.requires_grad, chain(encoder.parameters(), prediction_head.parameters())))
    optimizer_stage1 = torch.optim.AdamW(trainable_params, lr=stage1_pre_train_lr, weight_decay=weight_decay)
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=stage1_epochs, eta_min=stage1_pre_train_scheduler_eta_min)

    print(f"[Pretraining][Stage 1] Run configuration")
    print(f"\t- Update: encoder and prediction head")
    print(f"\t- Epochs: {stage1_epochs}")
    print(f"\t- Scheduler: CosineAnnealingLR - base LR {stage1_pre_train_lr} / min LR {stage1_pre_train_scheduler_eta_min}")
    
    # ==== Train/Val ====
    best_val = best_val = float("+inf")    
    train_losses = AverageMeter(name='pre_train_stage1/train_loss')
    val_losses = AverageMeter(name='pre_train_stage1/val_loss')
    for epoch in range(stage1_epochs):
        
        # Training
        train_losses.reset()
        encoder.train(); prediction_head.train()
        for batch_idx, batch in enumerate(pre_train_dataloader):
            signals, targets = batch
            signals, targets = signals.to(device), targets.to(device)
            
            # Zero gradients
            optimizer_stage1.zero_grad()

            feats = encoder(signals) # [B, 256]
            outputs = prediction_head(feats) # [B, 3]
                
            # Supervised regression loss
            loss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
            
            loss.backward()

            # Optimizer step
            optimizer_stage1.step()
            
            # Logg batch loss
            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('pre_train_stage1/train_loss', loss.item(), epoch * len(pre_train_dataloader) + batch_idx)
        
        # Step scheduler once per epoch
        scheduler_stage1.step()
        
        # Log epoch loss
        writer.add_scalar('pre_train_stage1/train_loss_epoch', train_losses.avg, epoch)
        
        # Validation
        val_losses.reset()
        encoder.eval(); prediction_head.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_valid_dataloader):
                signals, targets = batch
                signals, targets = signals.to(device), targets.to(device)
                
                feats = encoder(signals)
                outputs = prediction_head(feats)
                
                vloss = F.smooth_l1_loss(outputs, targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, targets)
                val_losses.update(vloss.item(), signals.size(0))

        writer.add_scalar('pre_train_stage1/val_loss_epoch', val_losses.avg, epoch)
        print(f"[Pretraining][Stage 1] Epoch {epoch+1}/{stage1_epochs} - Train {train_losses.avg:.5f} Val {val_losses.avg:.5f}")
            
        if val_losses.avg < best_val:
            print(f"[Pretraining][Stage 1] New best validation loss ✅: {val_losses.avg:.5f}")
            save_status(
                subject_id=None, epoch=epoch, model_name=model_name + "_encoder", 
                save_name=save_name, model=encoder, optimizer=optimizer_stage1, scheduler=scheduler_stage1, 
                meter=val_losses, checkpoint_path=checkpoint_path, config=config
            )
            save_status(
                subject_id=None, epoch=epoch, model_name=model_name + "_prediction_head", 
                save_name=save_name, model=prediction_head, optimizer=optimizer_stage1, scheduler=scheduler_stage1, 
                meter=val_losses, checkpoint_path=checkpoint_path, config=config
            )
            
            best_val = val_losses.avg
    
    print(f"[Pretraining][Stage 1] Supervised stage training completed ✅")
    print(f"\t- Best val loss {best_val}")
    print(f"[Pretraining][Stage 1] Supervised stage testing ...")
        
    # ==== Test ====
    # Load best encoder
    encoder = get_encoder_architecture(config)
    load_status(
        subject_id=None, model_name=model_name + "_encoder", save_name=save_name, 
        model=encoder, optimizer=None, scheduler=None, 
        checkpoint_path=checkpoint_path, config=config
    )
    encoder.eval()
    encoder = encoder.to(device)
    
    # Load best prediction head
    prediction_head = get_prediction_head_architecture(config)
    load_status(
        subject_id=None, model_name=model_name + "_prediction_head", save_name=save_name, 
        model=prediction_head, optimizer=None, scheduler=None, 
        checkpoint_path=checkpoint_path, config=config
    )
    prediction_head.eval()
    prediction_head = prediction_head.to(device)
    
    # Log graph to tensorboard
    example_input_batch = next(iter(pre_train_dataloader))
    signals, _ = example_input_batch
    signals = signals.to(device)

    full_model = Model(encoder, prediction_head).to(device)
    # Log whole graph
    if model_name != "Proto": # Skip Proto as it has MHA which is not traced with add_graph
        writer.add_graph(full_model, signals)
    
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
    all_test_outputs = np.empty((0, 3), dtype=float) 
    all_test_targets = np.empty((0, 3), dtype=float)
        
    with torch.no_grad():
        for batch_idx, batch in enumerate(pre_test_dataloader):
            signals, targets = batch
            signals, targets = signals.to(device), targets.to(device)
                        
            feats = encoder(signals)
            outputs = prediction_head(feats)
            
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
    writer.add_scalar('pre_train_test/loss_final', test_losses.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/sbp_mae_final', test_sbp_maes.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/dbp_mae_final', test_dbp_maes.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/map_mae_final', test_map_maes.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/sbp_me_final', test_sbp_mes.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/dbp_me_final', test_dbp_mes.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/map_me_final', test_map_mes.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/sbp_mae_std_final', test_sbp_mae_stds.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/dbp_mae_std_final', test_dbp_mae_stds.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/map_mae_std_final', test_map_mae_stds.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/sbp_me_std_final', test_sbp_me_stds.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/dbp_me_std_final', test_dbp_me_stds.avg, config['stage1_epochs'])
    writer.add_scalar('pre_train_test/map_me_std_final', test_map_me_stds.avg, config['stage1_epochs'])

    # Log test metrics (optionally, plot them) and return loss for validation
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'pretraining_supervised_stage'), plot=True)    
    
    print(f"[Pretraining][Stage 1] Supervised stage testing completed ✅")
    print(f"[Pretraining][Stage 1] ==== Stage 1 end ====")


def maml_meta_training(save_name, checkpoint_path, writer, model_name, config, device):
    
    print(f"[Pretraining][MAML] ==== MAML start ====")
    
    # Load best encoder
    encoder = get_encoder_architecture(config)
    load_status(
        subject_id=None, model_name=model_name + "_encoder", save_name=save_name, 
        model=encoder, optimizer=None, scheduler=None, 
        checkpoint_path=checkpoint_path, config=config
    )
    if config['use_lora']:
        encoder.attach_lora(r=config['lora_r'], alpha=config['lora_alpha'])
    encoder.eval()
    print(f"[Pretraining][MAML] Stage-1 encoder weights initialized ✅")
    
    # Load best prediction head
    prediction_head = get_prediction_head_architecture(config)
    load_status(
        subject_id=None, model_name=model_name + "_prediction_head", save_name=save_name, 
        model=prediction_head, optimizer=None, scheduler=None, 
        checkpoint_path=checkpoint_path, config=config
    )
    prediction_head.eval()
    print(f"[Pretraining][MAML] Stage-1 prediction head weights initialized ✅")
    
    # Count trainable params
    model = Model(encoder, prediction_head)
    model_trainable, model_non_trainable = count_parameters(model)
    print(f"[Pretraining][MAML] Parameter count:")
    print(f"\t- Model: {model_trainable:,} trainable, {model_non_trainable:,} non-trainable")
    
    # FOMAML from learn2learn
    model = MAML(
        model, 
        lr=config['inner_lr'], 
        first_order=True, 
        anil=(config['inner_adapt'] == 'head'), 
        lora=config['use_lora']
    ).to(device)

    # MAML meta-dataloader setup
    _, train_dataloader, val_dataloader, test_dataloader, _ = build_meta_splits_and_loaders(
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        seed=config['seed'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        meta_split_ratio=config['meta_train_split_ratio'],
        min_subject_sample_number=config['min_subject_sample_number'],
        k_support=config['k_support'],
        k_query=config['k_query'],
        meta_batch_size=config['meta_batch_size']
    )
    
    # Hyperparams / defaults
    meta_epochs = config.get('max_meta_epochs')
    meta_lr_schedule = config.get('meta_lr_schedule')
    inner_lr_schedule = config.get('inner_lr_schedule')
    inner_steps_schedule = config.get('inner_steps_schedule')
    
    # MAML specific parameters
    print(f"[Pretraining][MAML] Run configuration")
    print(f"\t- Update: {'Almost-No-Inner-Loop' if config['inner_adapt'] == 'head' else 'encoder and prediction head'}{' with LoRA' if config['use_lora'] else ''}")
    print(f"\t- Epochs: {meta_epochs}")
    print(f"\t- Meta LR schedule: {meta_lr_schedule} - base LR {config['meta_lr']} / min LR {config['meta_lr_scheduler_eta_min']}")
    print(f"\t- Inner LR schedule: {inner_lr_schedule} - LR {config['inner_lr']}")
    print(f"\t- Inner steps schedule: {inner_steps_schedule} - Steps {config['inner_steps']}")
    
    # Meta-optimizer for the meta-parameters
    model_param_list = list(model.parameters())
    meta_optimizer = torch.optim.Adam(model_param_list, lr=get_meta_lr(0, config), weight_decay=config['weight_decay'])
    
    # Meters
    best_val = float("+inf")
    epoch_query_loss_meter = AverageMeter(name='meta/train_query_loss')
    epoch_support_loss_meter = AverageMeter(name='meta/train_support_loss')
    epoch_meta_loss_meter = AverageMeter(name='meta/train_meta_loss')
    
    # ==== Train/Val ====
    # Outer loop
    for epoch in range(meta_epochs):
        
        model.train()

        # Set schedules / hyperparams for this epoch
        current_meta_lr = get_meta_lr(epoch, config)
        current_inner_lr = get_inner_lr(epoch, config)
        current_inner_steps = get_inner_steps(epoch, config)

        # Update meta-optimizer learning rate with manual schedule
        for param_group in meta_optimizer.param_groups:
            param_group['lr'] = current_meta_lr

        # Get MSL weights for post-update steps (and optionally pre-update)
        msl_weights = get_msl_weights(epoch, config, current_inner_steps)
    
        # Reset meters
        epoch_query_loss_meter.reset()
        epoch_support_loss_meter.reset()
        epoch_meta_loss_meter.reset()
        
        # Log current hyperparameters
        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)

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
                
                # Copy current meta-parameters to the fast weights
                adapted_model = model.clone()
                adapted_model.train()
                
                # Query loss before adaptation (step 0) — keep for logging and optional MSL include
                with torch.no_grad():
                    # Merge LoRA weights for inference if LoRA is applied
                    # N.B.: using class name instead of isinstance because of clone few lines above
                    if config['use_lora']:
                        adapted_model.module.encoder.merge_lora()
                    
                    out_q0 = adapted_model(qX)  # Use current meta-parameters
                    loss_q0 = F.smooth_l1_loss(out_q0, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q0, qY)
                task_query_pre_losses.append(loss_q0.item())
                
                support_losses = []
                per_step_query_losses = []

                # Inner loop
                for step in range(current_inner_steps):
                    # Unmerge LoRA weights for training
                    if config['use_lora']:
                        adapted_model.module.encoder.unmerge_lora()
                    
                    # --- support forward ---
                    out_s = adapted_model(sX)
                    loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)
                    support_losses.append(loss_s.item())

                    # FOMAML from l2l
                    adapted_model.adapt(loss_s)
                    
                    # Merge LoRA weights for inference
                    if config['use_lora']:
                        adapted_model.module.encoder.merge_lora()
                    
                    # Evaluate query using the updated fast_weights (multi-step loss)
                    out_q_step = adapted_model(qX)
                    loss_q_step = F.smooth_l1_loss(out_q_step, qY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_q_step, qY)
                    per_step_query_losses.append(loss_q_step)

                # --- Combine per-step query losses into single meta-loss for this task ---

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
            meta_loss.backward()

            # Meta step
            meta_optimizer.step()

            # Logging
            epoch_meta_loss_meter.update(meta_loss.item(), 1)

        # ---- After training batches (end of epoch) ----
        avg_train_query_loss = epoch_query_loss_meter.avg
        avg_train_support_loss = epoch_support_loss_meter.avg
        avg_train_meta_loss = epoch_meta_loss_meter.avg
        
        # Run validation
        val_loss = evaluate_maml_with_bp_metrics(model, val_dataloader, device, config)

        # Logging 
        writer.add_scalar('meta/train_query_loss_epoch', avg_train_query_loss, epoch)
        writer.add_scalar('meta/train_support_loss_epoch', avg_train_support_loss, epoch)
        writer.add_scalar('meta/train_meta_loss_epoch', avg_train_meta_loss, epoch)
        writer.add_scalar('meta/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)

        print(
            f"[Pretraining][MAML] Epoch {epoch+1}/{meta_epochs} - "
            f"train_support_loss {avg_train_support_loss:.5f}, "
            f"train_query_loss {avg_train_query_loss:.5f}, "
            f"train_meta_loss {avg_train_meta_loss:.5f}, "
            f"val_loss {val_loss:.5f}, "
            f"meta_lr {current_meta_lr:.5f}, "
            f"inner_lr {current_inner_lr:.5f}, "
            f"inner_steps {current_inner_steps}"
        )

        # Save best model according to validation loss
        if val_loss < best_val:
            save_ckpt = {
                'epoch': epoch,
                'learner_state_dict': model.state_dict(),
                'meta_optimizer_state_dict': meta_optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss
            }
            best_model_path = os.path.join(checkpoint_path, f"{save_name}_best_maml")
            torch.save(save_ckpt, best_model_path)
            best_val = val_loss
            print(f"[Pretraining][MAML] New best validation loss ✅: {val_loss:.5f}, saved to {best_model_path}")
    
    # After meta-training, run final test evaluation
    print(f"[Pretraining][MAML] MAML training completed ✅")
    print(f"\t- Best val loss {best_val}")
    print(f"[Pretraining][MAML] MAML testing ...")
    
    # Load best model
    best_ckpt = torch.load(os.path.join(checkpoint_path, f"{save_name}_best_maml"), weights_only=False)
    model.load_state_dict(best_ckpt['learner_state_dict'])
    
    # Saving feature statistics for intiializing the drift detector during the personalization step
    baseline_mean, baseline_std = compute_embedding_stats(model.encoder, train_dataloader, device)

    # Save to file for personalization later on
    stats_ckpt = {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
    }
    np.savez(os.path.join(checkpoint_path, f"{save_name}_embedding_stats.npz"), **stats_ckpt)
    print(f"[Pretraining][MAML] Embedding stats saved to {save_name}_embedding_stats.npz ✅")
    
    # ==== Test ====
    model = model.to(device)
    all_test_targets, all_test_outputs = evaluate_maml_with_bp_metrics(
        model, 
        test_dataloader, 
        device, 
        config, 
        test=True, 
        writer=writer
    )
    
    # Log test metrics
    _ = call_metric(all_test_targets, all_test_outputs, config, 
                   figure_savepath=os.path.join(config['figure_path'], 'maml_meta_stage'), 
                   plot=True)  
    
    writer.close()
    
    print(f"[Pretraining][Stage 1] MAML testing completed ✅")
    print(f"[Pretraining][MAML] === MAML end ====")
    
    
def evaluate_maml_with_bp_metrics(model, dataloader, device, config, test=False, writer=None):
    """
    Evaluation function for MAML learner
    """
    model.eval()
    
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
    
        all_test_outputs = np.empty((0, 3), dtype=float) 
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
            adapted_encoder = copy.deepcopy(model.encoder).to(device)
            adapted_prediction_head = copy.deepcopy(model.prediction_head).to(device)

            # Build inner optimizer with correct parameter groups & LRs
            inner_opt = build_inner_optimizer(adapted_encoder, adapted_prediction_head, base_lr=config['eval_lr'], config=config)
            
            adapted_encoder.train()
            adapted_prediction_head.train()

            # --- Inner loop adaptation ---
            for step in range(config['eval_steps']):
                out_s = adapted_prediction_head(adapted_encoder(sX))
                loss_s = F.smooth_l1_loss(out_s, sY) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(out_s, sY)

                inner_opt.zero_grad()
                loss_s.backward()
                inner_opt.step()

            # Merge LoRA weights for inference
            if config['use_lora']:
                adapted_encoder.merge_lora()

            # --- Query evaluation ---
            adapted_encoder.eval()
            adapted_prediction_head.eval()
            with torch.no_grad():
                out_q = adapted_prediction_head(adapted_encoder(qX))
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

                del adapted_encoder
                del adapted_prediction_head
                torch.cuda.empty_cache()
    
    if test:
        
        # Log test metrics
        writer.add_scalar('test/loss_final', test_losses.avg, config.get('max_meta_epochs'))
        writer.add_scalar('test/sbp_mae_final', test_sbp_maes.avg, config['max_meta_epochs'])
        writer.add_scalar('test/dbp_mae_final', test_dbp_maes.avg, config['max_meta_epochs'])
        writer.add_scalar('test/map_mae_final', test_map_maes.avg, config['max_meta_epochs'])
        writer.add_scalar('test/sbp_me_final', test_sbp_mes.avg, config['max_meta_epochs'])
        writer.add_scalar('test/dbp_me_final', test_dbp_mes.avg, config['max_meta_epochs'])
        writer.add_scalar('test/map_me_final', test_map_mes.avg, config['max_meta_epochs'])
        writer.add_scalar('test/sbp_mae_std_final', test_sbp_mae_stds.avg, config['max_meta_epochs'])
        writer.add_scalar('test/dbp_mae_std_final', test_dbp_mae_stds.avg, config['max_meta_epochs'])
        writer.add_scalar('test/map_mae_std_final', test_map_mae_stds.avg, config['max_meta_epochs'])
        writer.add_scalar('test/sbp_me_std_final', test_sbp_me_stds.avg, config['max_meta_epochs'])
        writer.add_scalar('test/dbp_me_std_final', test_dbp_me_stds.avg, config['max_meta_epochs'])
        writer.add_scalar('test/map_me_std_final', test_map_me_stds.avg, config['max_meta_epochs'])
        
        return all_test_targets, all_test_outputs

    else:
        return val_losses.avg
    

