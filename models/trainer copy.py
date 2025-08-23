import os
import sys
folders_to_add = ['data', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import math
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

    if not config['ssl']:
        if not config['meta_learning']:
            pass
        else:
            # call Reptile meta-training (this returns the final model after meta-training)
            return reptile_meta_training(
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
        pass



def _compute_supervised_loss(outputs, targets, config):
    """Helper to compute supervised loss consistent with the rest of the repo."""
    if config['criterion'] == 'MSELoss':
        return F.mse_loss(outputs, targets)
    elif config['criterion'] == 'SmoothL1Loss':
        return F.smooth_l1_loss(outputs, targets)
    else:
        raise ValueError("Invalid criterion in config['criterion']")

def _ensure_meta_batch(tensor):
    """
    Ensure tensor has meta-batch dim at axis 0.
    Expected task tensor shapes from MetaTaskDataset:
      - X support: (k_support, C, T)  -> returns (1, k_support, C, T)
      - if outer DataLoader uses batch_size>1: (meta_batch, k_support, C, T)
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor  # already has meta-batch dim (or another unexpected shape)

def evaluate_meta_with_bp_metrics(
    model,
    dataloader,
    device,
    config,
    inner_steps=5,
    lr_inner=1e-2,
    n_tasks_eval=None,
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

    if test:
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
    else:
        val_losses = AverageMeter(name='meta/val_loss')

    tasks_done = 0
    data_iter = iter(dataloader)

    while True:
        if n_tasks_eval is not None and tasks_done >= n_tasks_eval:
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        (Xs, Ys), (Xq, Yq), pid = batch
        if isinstance(Xs, torch.Tensor) and Xs.dim() == 4 and Xs.shape[0] == 1:
            Xs, Ys, Xq, Yq = Xs[0], Ys[0], Xq[0], Yq[0]

        sX = Xs.to(device).float()
        sY = Ys.to(device).float()
        qX = Xq.to(device).float()
        qY = Yq.to(device).float()

        # --- Inner adaptation ---
        adapted = copy.deepcopy(model).to(device)
        adapted.train()
        inner_opt = torch.optim.SGD(adapted.parameters(), lr=lr_inner)
        for _ in range(inner_steps):
            inner_opt.zero_grad()
            out_s = adapted(sX)
            loss_s = (
                F.mse_loss(out_s, sY)
                if config.get("criterion", "MSELoss") == "MSELoss"
                else F.smooth_l1_loss(out_s, sY)
            )
            loss_s.backward()
            inner_opt.step()

        # --- Evaluate on query ---
        adapted.eval()
        with torch.no_grad():
            out_q = adapted(qX)
            loss_q = (
                F.mse_loss(out_q, qY)
                if config.get("criterion", "MSELoss") == "MSELoss"
                else F.smooth_l1_loss(out_q, qY)
            )
            # use your existing metrics function
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
                test_sbp_mes.update(metric_values['sbp_me'], qX.size(0))
                test_dbp_mes.update(metric_values['dbp_me'], qX.size(0))
                test_sbp_mae_stds.update(metric_values['sbp_mae_std'], qX.size(0))
                test_dbp_mae_stds.update(metric_values['dbp_mae_std'], qX.size(0))
                test_sbp_me_stds.update(metric_values['sbp_me_std'], qX.size(0))
                test_dbp_me_stds.update(metric_values['dbp_me_std'], qX.size(0))

            else:
                val_losses.update(loss_q.item(), qX.size(0))

        tasks_done += 1
        del adapted
        torch.cuda.empty_cache()

    if test:
        # Log test metrics
        # Note: `epoch` here is the last epoch of training, not ideal for test summary
        writer.add_scalar('test/loss_final', test_losses.avg, config.get('max_training_epochs', 100))
        writer.add_scalar('test/sbp_mae_final', test_sbp_maes.avg, config.get('max_training_epochs', 100))
        writer.add_scalar('test/dbp_mae_final', test_dbp_maes.avg, config.get('max_training_epochs', 100))
        writer.add_scalar('test/sbp_me_final', test_sbp_mes.avg, config.get('max_training_epochs', 100))
        writer.add_scalar('test/dbp_me_final', test_dbp_mes.avg, config.get('max_training_epochs', 100))
        writer.add_scalar('test/sbp_mae_std_final', test_sbp_mae_stds.avg, config.get('max_training_epochs', 100))
        writer.add_scalar('test/dbp_mae_std_final', test_dbp_mae_stds.avg, config.get('max_training_epochs', 100))

        return all_test_targets, all_test_outputs

    else:
        return val_losses.avg

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
    optimizer,          # unused here, kept for API parity
    scheduler,          # unused here
    config,
    device
):
    """
    Enhanced Reptile meta-training loop with sophisticated scheduling.
    - train_dataloader yields tasks: ((Xs, Ys), (Xq, Yq), pid) where support/query have shapes (k,C,T)/(k,T)
    - meta-batching supported by setting DataLoader(batch_size > 1).
    - Enhanced scheduling options for both outer and inner loops
    """
    
    # Hyperparams / defaults
    meta_epochs = config.get('max_training_epochs', 100)
    base_meta_lr = config.get('meta_lr', 1e-3)
    base_inner_steps = config.get('inner_steps', 5)
    base_lr_inner = config.get('lr_inner', 1e-2)
    meta_val_tasks = config.get('meta_val_tasks', 100)
    
    # Enhanced scheduling parameters
    meta_lr_schedule = config.get('meta_lr_schedule', 'constant')  # 'constant', 'cosine', 'step', 'exponential'
    inner_lr_schedule = config.get('inner_lr_schedule', 'constant')  # 'constant', 'cosine', 'adaptive'
    inner_steps_schedule = config.get('inner_steps_schedule', 'constant')  # 'constant', 'increasing', 'adaptive'
    
    # Schedule-specific parameters
    meta_lr_decay = config.get('meta_lr_decay', 0.95)  # for exponential decay
    meta_lr_steps = config.get('meta_lr_steps', [50, 100, 150])  # for step decay
    meta_lr_gamma = config.get('meta_lr_gamma', 0.5)  # step decay factor
    
    inner_lr_min = config.get('inner_lr_min', 2.5e-3)
    inner_steps_max = config.get('inner_steps_max', 10)
    
    # Adaptive scheduling parameters
    patience_adaptive = config.get('adaptive_patience', 10)
    adaptive_factor = config.get('adaptive_factor', 0.8)
    val_loss_history = []
    no_improvement_count = 0

    def get_meta_lr(epoch):
        """Get meta learning rate based on schedule."""
        if meta_lr_schedule == 'constant':
            return base_meta_lr
        elif meta_lr_schedule == 'cosine':
            return base_meta_lr * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
        elif meta_lr_schedule == 'step':
            lr = base_meta_lr
            for step in meta_lr_steps:
                if epoch >= step:
                    lr *= meta_lr_gamma
            return lr
        elif meta_lr_schedule == 'exponential':
            return base_meta_lr * (meta_lr_decay ** epoch)
        else:
            return base_meta_lr

    def get_inner_lr(epoch, val_loss=None):
        """Get inner learning rate based on schedule."""
        nonlocal no_improvement_count
        
        if inner_lr_schedule == 'constant':
            return base_lr_inner
        elif inner_lr_schedule == 'cosine':
            return inner_lr_min + (base_lr_inner - inner_lr_min) * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
        elif inner_lr_schedule == 'adaptive':
            if val_loss is not None and len(val_loss_history) > 0:
                if val_loss >= min(val_loss_history):
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                if no_improvement_count >= patience_adaptive:
                    return max(base_lr_inner * (adaptive_factor ** (no_improvement_count // patience_adaptive)), inner_lr_min)
            return base_lr_inner
        else:
            return base_lr_inner

    def get_inner_steps(epoch, val_loss=None):
        """Get number of inner steps based on schedule."""
        if inner_steps_schedule == 'constant':
            return base_inner_steps
        elif inner_steps_schedule == 'increasing':
            # Linearly increase inner steps over epochs
            progress = epoch / meta_epochs
            return int(base_inner_steps + progress * (inner_steps_max - base_inner_steps))
        elif inner_steps_schedule == 'adaptive':
            if val_loss is not None and len(val_loss_history) > 1:
                # Increase steps if validation loss is not improving
                recent_improvement = val_loss < min(val_loss_history[-5:]) if len(val_loss_history) >= 5 else True
                if not recent_improvement:
                    return min(base_inner_steps + 2, inner_steps_max)
            return base_inner_steps
        else:
            return base_inner_steps

    # Optionally load a pretrained checkpoint
    if config.get('pretrained_path') is not None:
        ckpt = torch.load(config['pretrained_path'], map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model_state = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and any(k.startswith('encoder') for k in ckpt.keys()):
            model_state = ckpt
        else:
            model_state = ckpt
        
        scope = config.get('pretrained_scope', 'all')
        if scope == 'all':
            model.load_state_dict(model_state, strict=False)
            print(f"[Reptile] Loaded pretrained weights (all) from {config['pretrained_path']}")
        elif scope == 'backbone':
            cur_state = model.state_dict()
            filtered = {k: v for k, v in model_state.items() if k in cur_state and k.startswith('encoder')}
            cur_state.update(filtered)
            model.load_state_dict(cur_state)
            print(f"[Reptile] Loaded pretrained BACKBONE weights from {config['pretrained_path']}")
        else:
            model.load_state_dict(model_state, strict=False)
            print(f"[Reptile] Loaded pretrained weights (fallback) from {config['pretrained_path']}")

    model = model.to(device)
    
    # Keep meta-params as an explicit dict
    meta_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
    
    # Initialize first-order approximation option (faster Reptile variant)
    use_first_order = config.get('first_order_reptile', False)
    
    best_val = float("+inf")
    
    print(f"[Reptile] Starting meta-training with schedules:")
    print(f"  - Meta LR schedule: {meta_lr_schedule}")
    print(f"  - Inner LR schedule: {inner_lr_schedule}")
    print(f"  - Inner steps schedule: {inner_steps_schedule}")
    print(f"  - First-order approximation: {use_first_order}")
    # === Inner optimizer builder with per-parameter LRs and optional ANIL ===
    def build_inner_optimizer(adapted_model, base_lr, cfg):
        anil = cfg.get('anil_head_only', False)
        head_tokens = set(cfg.get('head_name_tokens', ['head','regressor','fc','out']))
        last_tokens = set(cfg.get('last_block_tokens', ['layer4','block4','stage4','encoder.layer4','encoder.blocks.3']))
        groups = []
        for name, p in adapted_model.named_parameters():
            if not p.requires_grad:
                continue
            if anil and not any(tok in name for tok in head_tokens):
                p.requires_grad = False
                continue
            if any(tok in name for tok in head_tokens):
                mult = cfg.get('inner_head_lr_mult', 1.0)
            elif any(tok in name for tok in last_tokens):
                mult = cfg.get('inner_last_block_lr_mult', 0.7)
            else:
                mult = cfg.get('inner_backbone_lr_mult', cfg.get('backbone_lr_multiplier', 0.3))
            groups.append({'params': [p], 'lr': base_lr * float(mult)})
        if not groups:
            groups = [{'params': [p for p in adapted_model.parameters() if p.requires_grad], 'lr': base_lr}]
        return torch.optim.SGD(groups, momentum=0.0)


    # Outer loop: epochs
    for epoch in range(meta_epochs):
        model.train()
        
        # Get current hyperparameters based on schedules
        current_meta_lr = get_meta_lr(epoch)
        current_inner_lr = get_inner_lr(epoch)
        current_inner_steps = get_inner_steps(epoch)
        
        epoch_query_loss_meter = AverageMeter(name='meta/train_query_loss')
        epoch_support_loss_meter = AverageMeter(name='meta/train_support_loss')
        
        # Log current hyperparameters
        writer.add_scalar('meta/meta_lr', current_meta_lr, epoch)
        writer.add_scalar('meta/inner_lr', current_inner_lr, epoch)
        writer.add_scalar('meta/inner_steps', current_inner_steps, epoch)
        
        # Iterate over tasks provided by train_dataloader
        for batch_idx, batch in enumerate(train_dataloader):
            (Xs, Ys), (Xq, Yq), pid = batch

            # Ensure meta-batch dim exists
            Xs = _ensure_meta_batch(Xs).to(device).float()
            Xq = _ensure_meta_batch(Xq).to(device).float()

            if isinstance(Ys, list):
                Ys = [_ensure_meta_batch(y).to(device).float() for y in Ys]
                Yq = [_ensure_meta_batch(y).to(device).float() for y in Yq]
            else:
                Ys = _ensure_meta_batch(Ys).to(device).float()
                Yq = _ensure_meta_batch(Yq).to(device).float()
                
            meta_batch = Xs.shape[0]
            
            # Accumulate parameter deltas across tasks
            acc_deltas = {k: torch.zeros_like(v) for k, v in meta_params.items()}
            task_query_losses = []
            task_support_losses = []

            for t in range(meta_batch):
                # Extract task-specific support/query
                sX = Xs[t]
                qX = Xq[t]
                if isinstance(Ys, list):
                    sY = [y[t] for y in Ys]
                    qY = [y[t] for y in Yq]
                else:
                    sY = Ys[t]
                    qY = Yq[t]

                # Create adapted model copy for this task
                adapted = copy.deepcopy(model).to(device)
                adapted.train()
                
                # Inner optimizer with current learning rate
                inner_opt = build_inner_optimizer(adapted, current_inner_lr, config)

                # Inner loop: adapt to support with current number of steps
                support_losses = []
                for step in range(current_inner_steps):
                    inner_opt.zero_grad()
                    out_s = adapted(sX)
                    loss_s = _compute_supervised_loss(out_s, sY.squeeze(0), config)
                    
                    if not use_first_order:
                        # Standard Reptile: compute gradients and update
                        loss_s.backward()
                        inner_opt.step()
                    else:
                        # First-order approximation: detach gradients
                        loss_s.backward()
                        with torch.no_grad():
                            for param in adapted.parameters():
                                if param.grad is not None:
                                    param.data -= current_inner_lr * param.grad.data
                        inner_opt.zero_grad()
                    
                    support_losses.append(loss_s.item())

                # Evaluate adapted model on query (for logging)
                adapted.eval()
                with torch.no_grad():
                    out_q = adapted(qX)
                    loss_q = _compute_supervised_loss(out_q, qY.squeeze(0), config)
                
                task_query_losses.append(loss_q.item())
                task_support_losses.append(np.mean(support_losses))
                epoch_query_loss_meter.update(loss_q.item(), 1)
                epoch_support_loss_meter.update(np.mean(support_losses), 1)

                # Collect adapted parameters and compute delta
                for n, p in adapted.named_parameters():
                    acc_deltas[n] += (p.detach() - meta_params[n])

                # Free memory
                del adapted
                torch.cuda.empty_cache()

            # Average deltas over tasks in meta-batch and take a meta step
            for k in meta_params:
                avg_delta = acc_deltas[k] / float(meta_batch)
                meta_params[k] = meta_params[k] + current_meta_lr * avg_delta

            # Load updated meta params back into model
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(meta_params[n])

            # Logging per-step
            if (batch_idx + 1) % config.get('meta_log_step', 100) == 0:
                avg_q = float(np.mean(task_query_losses))
                avg_s = float(np.mean(task_support_losses))
                step_idx = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('meta/train_query_loss_step', avg_q, step_idx)
                writer.add_scalar('meta/train_support_loss_step', avg_s, step_idx)
                print(f"[Meta] Epoch {epoch+1} Step {batch_idx+1}/{len(train_dataloader)} - "
                      f"support_loss {avg_s:.6f}, query_loss {avg_q:.6f} "
                      f"(meta_lr={current_meta_lr:.6f}, inner_lr={current_inner_lr:.6f}, inner_steps={current_inner_steps})")

        # End epoch: run validation on val tasks
        val_loss = evaluate_meta_with_bp_metrics(
            model, val_dataloader, device, config, 
            current_inner_steps, current_inner_lr, 
            n_tasks_eval=meta_val_tasks
        )
        
        val_loss_history.append(val_loss)
        
        # Update adaptive schedules based on validation performance
        current_inner_lr = get_inner_lr(epoch, val_loss)
        current_inner_steps = get_inner_steps(epoch, val_loss)
        
        writer.add_scalar('meta/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('meta/train_query_loss_epoch', epoch_query_loss_meter.avg, epoch)
        writer.add_scalar('meta/train_support_loss_epoch', epoch_support_loss_meter.avg, epoch)
        
        print(f"[Meta] Epoch {epoch+1}/{meta_epochs} - "
              f"train_support_loss {epoch_support_loss_meter.avg:.6f}, "
              f"train_query_loss {epoch_query_loss_meter.avg:.6f}, "
              f"val_loss {val_loss:.6f}")

        # Save best model according to validation loss
        if val_loss < best_val:
            save_status(None, epoch, model_name, save_name, model, None, None, 
                       epoch_query_loss_meter, checkpoint_path, config)
            best_val = val_loss
            print(f"[Meta] New best validation loss: {val_loss:.6f}")

        # Early stopping if configured
        if config.get('es_enable', False):
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping meta-training")
                break

    # After meta-training, run final test evaluation
    print(f"[Meta] Starting final test evaluation with best model...")
    
    # Load best model
    model = get_model_architecture(config)
    load_status(None, model_name, save_name, model, None, None, checkpoint_path, config)
    model = model.to(device)
    
    all_test_targets, all_test_outputs = evaluate_meta_with_bp_metrics(
        model, test_dataloader, device, config, 
        base_inner_steps, base_lr_inner,  # Use base values for final test
        n_tasks_eval=config.get('meta_test_tasks', 200), 
        test=True, writer=writer
    )
    
    writer.close()
    
    return all_test_targets, all_test_outputs
