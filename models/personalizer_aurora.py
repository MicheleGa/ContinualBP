import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.maml import MAML
from models.DriftDetectors import MMDDriftOnline, LSDDDriftOnline
from component_factory import ReservoirReplayBuffer, Model
from training_utils.helpers import get_encoder_architecture, get_prediction_head_architecture, build_inner_optimizer
from training_utils.metrics import call_metric, compute_transfer_metrics_from_matrix
from data.online_dataset import OnlineSubjectDataset
from data.online_dataset_aurora import AuroraOnlineSubjectDataset
from data.preprocessing_utils.data_visualization import plot_blockwise_mae, plot_subject_annotation_blocks, plot_update_summary_table, plot_param_updates, plot_drift_calibration_summary


def eval_model_on_sample_set(encoder, prediction_head, dataset, sample_list, device, config):
    r"""
    Evaluates the model on a specific set of subject run samples, calculating Mean Absolute 
    Error (MAE) and standard deviation for SBP and DBP.
    
    Parameters
    ------------
    encoder (torch.nn.Module): 
        The feature extraction network.
        
    prediction_head (torch.nn.Module): 
        The regression head that maps features to blood pressure values.
        
    dataset (PhysioDataset): 
        The dataset instance used to retrieve signals and targets via indices.
        
    sample_list (list): 
        A list of sample indices (IDs) representing the specific set of samples to evaluate.
        
    device (torch.device): 
        The computational device (CPU or CUDA) for tensor operations.
        
    config (dict): 
        Configuration dictionary containing batch size and block split settings.
        
    Returns
    ------------
    output param 1:
        SBP Mean Absolute Error (float).
        
    output param 2:
        SBP Error Standard Deviation (float).
        
    output param 3:
        DBP Mean Absolute Error (float).
        
    output param 4:
        DBP Error Standard Deviation (float).
        
    output param 5:
        Predicted SBP and DBP values as a numpy array of shape (N, 2).
        
    output param 6:
        Ground truth SBP and DBP values as a numpy array of shape (N, 2).   
    """

    SBP_IDX = 0
    DBP_IDX = 1
    B = config['personalization_batch_size']
    all_outputs = []
    all_targets = []

    encoder.eval()
    prediction_head.eval()
    with torch.no_grad():
        for i in range(0, len(sample_list), B):
            batch_ids = sample_list[i:i + B]
            batch = [dataset.__getitem__(sid) for sid in batch_ids]

            signals = torch.stack([s for s, _, _ in batch]).to(device)
            targets = torch.stack([t for _, t, _ in batch]).to(device)

            outputs = prediction_head(encoder(signals))

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    if len(all_outputs) == 0:
        return None, None, None, None, None, None

    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Select SBP & DBP only
    outputs_sbp_dbp = all_outputs[:, [SBP_IDX, DBP_IDX]]
    targets_sbp_dbp = all_targets[:, [SBP_IDX, DBP_IDX]]

    # Absolute errors
    abs_err = np.abs(outputs_sbp_dbp - targets_sbp_dbp)

    sbp_err = abs_err[:, SBP_IDX]
    dbp_err = abs_err[:, DBP_IDX]

    sbp_mae = float(np.mean(sbp_err))
    sbp_std = float(np.std(sbp_err))

    dbp_mae = float(np.mean(dbp_err))
    dbp_std = float(np.std(dbp_err))

    return (
        sbp_mae, sbp_std,
        dbp_mae, dbp_std,
        outputs_sbp_dbp,
        targets_sbp_dbp
    )
    

def personalize_no_adapt(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Performs the 'no adapt' personalization baseline by evaluating a pretrained model across 
    sequential data blocks without updating weights to establish a performance floor.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'no_adapt').
        
    dataset (PhysioDataset): 
        The dataset instance providing access to subject-specific signals and labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific results and plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking metrics during the personalization process.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model inference.
        
    config (dict): 
        Configuration dictionary containing sequence lengths, batch sizes, and block settings.
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and STD calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
    
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    calibration_ids = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            enc.eval(); ph.eval()
            with torch.no_grad():
                calibration_features = enc(calibration_signals)
                
            # Encoder params for logging
            n_updated_calibration = 0
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
                
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
        
        # Log updated parameters
        param_update_log.append({
            "step_idx": step_idx,
            "block_idx": batch_info['block_idx'],
            "batch_idx": batch_info['batch_idx'],
            "update_mode": config['inner_adapt'], # irrelevant for this baseline, maintained for consistency with other baselines
            "n_updated_params": 0,
            "total_params": total_params,
            "fraction_updated": 0
        })
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1
            
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)
    
    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_calibration_only(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Calibration-only' personalization baseline where the model adapts 
    once to the first available data block (or the first block exceeding a drift threshold) 
    and then remains frozen for all subsequent blocks.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'first_batch_ft').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking training and validation loss during the one-time adaptation.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing personalization hyperparameters (learning rate, steps, criterion).
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
    
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)

    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Calibration ids
    calibration_ids = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            if config['inner_adapt'] == 'all':
                
                # Update also encoder: trian first with head for calibration and then predict feats for detector init
                # -> train first with head for calibration
                # -> then predict feats for detector init
                enc.train(); ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(enc(calibration_signals))
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                enc.eval(); ph.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
            elif config['inner_adapt'] == 'head':
                
                # Frozen encoder: predict features for head calibration and detector init
                enc.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
                ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(calibration_features)
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                ph.eval()
                
            else:
                raise ValueError('Inexistent adaptation mode, allowed is all or head!')
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
            
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
        
        # --- No more adaptation for this baseline (calibration only) ---
        # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
        # Log updated parameters
        param_update_log.append({
            "step_idx": step_idx,
            "block_idx": batch_info['block_idx'],
            "batch_idx": batch_info['batch_idx'],
            "update_mode": config['inner_adapt'], # irrelevant for this baseline, maintained for consistency with other baselines
            "n_updated_params": 0 if step_idx > config['calibration_phase_size'] else n_updated_calibration,
            "total_params": total_params,
            "fraction_updated": 0
        })
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)

    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)
        
    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics


def personalize_online(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Online' personalization baseline where the model continuously adapts to new 
    data blocks as they arrive, either at every step or based on a drift detection trigger.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'online_adaptation').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking training and validation loss for each online adaptation step.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing personalization hyperparameters (learning rate, steps, setup type).
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
        
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)
    
    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_ids = []
    n_steps = 0 # number of detector predictions after calibration
    n_detections = 0
    first_detection = None
    detection_timesteps = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            if config['inner_adapt'] == 'all':
                
                # Update also encoder: trian first with head for calibration and then predict feats for detector init
                # -> train first with head for calibration
                # -> then predict feats for detector init
                enc.train(); ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(enc(calibration_signals))
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                enc.eval(); ph.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
            elif config['inner_adapt'] == 'head':
                
                # Frozen encoder: predict features for head calibration and detector init
                enc.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
                ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(calibration_features)
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                ph.eval()
                
            else:
                raise ValueError('Inexistent adaptation mode, allowed is all or head!')
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
            if config['setup_type'] == 'drift':
                # Initialize drfit detector
                # NOTE: only after calibration completion
                reference_data = calibration_features.detach().clone()

                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
             
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
        
        # ----- Feature drift tracking -----
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]

        signals = torch.stack([x for x, _, _ in sample_batch]).to(device)
        targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
        
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals) 
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            for i in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[i].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
                    n_steps += 1
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == config['calibration_phase_size']:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > config['calibration_phase_size']:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": 'head',
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                else:
                    raise ValueError("Step idx should not be less than calibration phase size at this point!")
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updates during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            # Encoder params for logging
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])
            total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in ph.parameters())

            # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
            if step_idx == config['calibration_phase_size']:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > config['calibration_phase_size'] else config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Loss Function
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # Training data/labels are already prepared for the current batch
            for step in range(config["personalization_steps"]):
                
                # Train prediction head
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()
                
            # Set to eval mode after adaptation
            ph.eval()
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1      
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)
    
    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    if config['setup_type'] == 'drift':    
        
        print(f"[Personalization] Total detections with drift-aware updates: {n_detections} out of {n_steps} steps with drift detection after calibration ✓")
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{config['detector_ert']}_w{config['detector_window_size']}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=config['calibration_phase_size'],
            batch_size=config['personalization_batch_size'],
            ert=config['detector_ert'],
            window_size=config['detector_window_size'],
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_online_from_scratch(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Online From Scratch' personalization baseline. Unlike the 'online' 
    baseline, this method begins with a randomly initialized (fresh) model and 
    continually trains it on subject-specific data blocks as they arrive.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'online_from_scratch').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking training and validation loss for each online adaptation step.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing personalization hyperparameters (learning rate, steps, batch size).
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
    
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)

    # Initialize encoder and prediction head (fresh weights)
    encoder_fresh = get_encoder_architecture(config)
    prediction_head_fresh = get_prediction_head_architecture(config)
    fresh_learner = Model(encoder_fresh, prediction_head_fresh)
    print(f"[Personalization] Fresh encoder & head initialized ✓")

    # To device
    enc, ph = fresh_learner.encoder.to(device), fresh_learner.prediction_head.to(device)
    
    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_ids = []
    n_steps = 0 # number of detector predictions after calibration
    n_detections = 0
    first_detection = None
    detection_timesteps = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration with fresh weights, we always update teh full model
                
            # Update also encoder: trian first with head for calibration and then predict feats for detector init
            # -> train first with head for calibration
            # -> then predict feats for detector init
            enc.train(); ph.train()
            for step in range(config["personalization_steps"]):
                out = ph(enc(calibration_signals))
                loss = criterion(out, calibration_targets)

                opt.zero_grad()
                loss.backward()
                opt.step()
            
            enc.eval(); ph.eval()
            with torch.no_grad():
                calibration_features = enc(calibration_signals)
                
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
            if config['setup_type'] == 'drift':
                # Initialize drfit detector
                # NOTE: only after calibration completion
                reference_data = calibration_features.detach().clone()

                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
                
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
           
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
            
        # ----- Feature drift tracking -----
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]

        signals = torch.stack([x for x, _, _ in sample_batch]).to(device)
        targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
        
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals) 
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            for i in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[i].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
                    n_steps += 1    
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == config['calibration_phase_size']:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > config['calibration_phase_size']:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": 'head',
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                else:
                    raise ValueError("Step idx should not be less than calibration phase size at this point!")
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updated during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            # Encoder params for logging
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])
            total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in ph.parameters())

            # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
            if step_idx == config['calibration_phase_size']:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > config['calibration_phase_size'] else config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Loss Function
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # Training data/labels are already prepared for the current batch
            for step in range(config["personalization_steps"]):
                
                # Train prediction head
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()
                
            # Set to eval mode after adaptation
            ph.eval()
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1  
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)
    
    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    if config['setup_type'] == 'drift':    
        
        print(f"[Personalization] Total detections with drift-aware updates: {n_detections} out of {n_steps} steps with drift detection after calibration ✓")
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{config['detector_ert']}_w{config['detector_window_size']}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=config['calibration_phase_size'],
            batch_size=config['personalization_batch_size'],
            ert=config['detector_ert'],
            window_size=config['detector_window_size'],
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 

    
def personalize_feature_replay(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Feature Replay' personalization Cl algorithm that mitigates catastrophic 
    forgetting by storing previously seen latent features in a reservoir buffer and 
    interleaving them with current data during adaptation.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'feature_replay').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking training and validation loss during the adaptation steps.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing hyperparameters such as 'replay_buffer_size' and learning rates.
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
        
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)

    # Feature replay buffer
    replay_buffer = ReservoirReplayBuffer(max_size=config['replay_buffer_size'])
    
    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_ids = []
    n_steps = 0 # number of detector predictions after calibration
    n_detections = 0
    first_detection = None
    detection_timesteps = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            if config['inner_adapt'] == 'all':
                
                # Update also encoder: trian first with head for calibration and then predict feats for detector init
                # -> train first with head for calibration
                # -> then predict feats for detector init
                enc.train(); ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(enc(calibration_signals))
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                enc.eval(); ph.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
            elif config['inner_adapt'] == 'head':
                
                # Frozen encoder: predict features for head calibration and detector init
                enc.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
                ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(calibration_features)
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                ph.eval()
                
            else:
                raise ValueError('Inexistent adaptation mode, allowed is all or head!')
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
            if config['setup_type'] == 'drift':
                # Initialize drfit detector
                # NOTE: only after calibration completion
                reference_data = calibration_features.detach().clone()

                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
                
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
            
        # ----- Feature drift tracking -----
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]

        signals = torch.stack([x for x, _, _ in sample_batch]).to(device)
        targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
        
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals) 
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            for i in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[i].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
                    n_steps += 1     
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == config['calibration_phase_size']:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > config['calibration_phase_size']:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": 'head',
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                else:
                    raise ValueError("Step idx should not be less than calibration phase size at this point!")
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updated during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            # Encoder params for logging
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])
            total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in ph.parameters())

            # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
            if step_idx == config['calibration_phase_size']:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > config['calibration_phase_size'] else config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Loss Function
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # Training data/labels are already prepared for the current batch
            for step in range(config["personalization_steps"]):
                
                # Train prediction head
                ph.train()
                
                # Reservoir sampling for feature replay, same size as train features
                replay_feats, replay_tgts = replay_buffer.sample(train_features.shape[0])
                    
                # Replayed feats can be none on the first update as there are no previous features to replay
                if replay_feats is not None:
                    replay_feats = replay_feats.to(device).detach()
                    replay_tgts = replay_tgts.to(device)
                    feats = torch.cat([train_features, replay_feats], dim=0)
                    tgts = torch.cat([train_targets, replay_tgts], dim=0)
                else:
                    feats, tgts = train_features, train_targets
                
                out = ph(feats)
                loss = criterion(out, tgts)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()
                
            # Set to eval mode after adaptation
            ph.eval()
            
            # Update replay buffer 
            with torch.no_grad():
                replay_buffer.add(features.detach().cpu(), targets.detach().cpu())
            
            # Detector re-init when buffer has enough samples for a new reference set
            if config['setup_type'] == 'drift' and len(replay_buffer) > (config['personalization_batch_size'] * config['calibration_phase_size']):
                
                drift_detector = None
                
                # Re-initialize drfit detector
                # NOTE: only after having updated the replay buffer
                reference_data = torch.stack(replay_buffer.features).numpy().copy()
                
                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data,
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data,
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector re-initialized with {reference_data.shape[0]} samples from the replay_buffer ✓")
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)
    
    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    if config['setup_type'] == 'drift':    
        
        print(f"[Personalization] Total detections with drift-aware updates: {n_detections} out of {n_steps} steps with drift detection after calibration ✓")
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{config['detector_ert']}_w{config['detector_window_size']}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=config['calibration_phase_size'],
            batch_size=config['personalization_batch_size'],
            ert=config['detector_ert'],
            window_size=config['detector_window_size'],
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_lwf(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Learning without Forgetting' (LwF) personalization CL algorithm. This approach 
    combines a standard supervised loss on new data with a distillation loss that 
    constrains the current model to mimic the outputs of its previous iteration, 
    thereby preserving prior knowledge without storing old data.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'lwf_adaptation').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking total loss, distillation loss, and validation performance.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing personalization hyperparameters including 'lwf_lambda'.
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
        
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)
    
    # LwF-specfic loss weight
    lwf_lambda = config.get("lwf_lambda")
    
    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_ids = []
    n_steps = 0 # number of detector predictions after calibration
    n_detections = 0
    first_detection = None
    detection_timesteps = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):

        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            if config['inner_adapt'] == 'all':
                
                # Update also encoder: trian first with head for calibration and then predict feats for detector init
                # -> train first with head for calibration
                # -> then predict feats for detector init
                enc.train(); ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(enc(calibration_signals))
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                enc.eval(); ph.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
            elif config['inner_adapt'] == 'head':
                
                # Frozen encoder: predict features for head calibration and detector init
                enc.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
                ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(calibration_features)
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                ph.eval()
                
            else:
                raise ValueError('Inexistent adaptation mode, allowed is all or head!')
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
            if config['setup_type'] == 'drift':
                # Initialize drfit detector
                # NOTE: only after calibration completion
                reference_data = calibration_features.detach().clone()

                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
        
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
            
        # ----- Feature drift tracking -----
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]

        signals = torch.stack([x for x, _, _ in sample_batch]).to(device)
        targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
        
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals) 
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            for i in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[i].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
                    n_steps += 1  
                
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == config['calibration_phase_size']:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > config['calibration_phase_size']:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": 'head',
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                else:
                    raise ValueError("Step idx should not be less than calibration phase size at this point!")
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')
        
        if do_adapt:
            # ---------- Adaptation ----------
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updated during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            # Encoder params for logging
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])
            total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in ph.parameters())

            # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
            if step_idx == config['calibration_phase_size']:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > config['calibration_phase_size'] else config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Loss Function
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # LwF teacher: it can be either the pretrained head or the previously updated head
            # Here we use the previously updated head as teacher
            teacher_ph = copy.deepcopy(ph)

            for p in teacher_ph.parameters():
                p.requires_grad = False

            teacher_ph.eval()
            with torch.no_grad():
                teacher_out = teacher_ph(train_features)
            
            for step in range(config["personalization_steps"]):

                # Train prediction head
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)

                # L2 distance from teacher and student logits
                distill_loss = F.mse_loss(out, teacher_out)
            
                loss = loss + lwf_lambda * distill_loss
            
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()
            
            # Load best model
            ph.eval()
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)

    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    if config['setup_type'] == 'drift':    
        
        print(f"[Personalization] Total detections with drift-aware updates: {n_detections} out of {n_steps} steps with drift detection after calibration ✓")
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{config['detector_ert']}_w{config['detector_window_size']}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=config['calibration_phase_size'],
            batch_size=config['personalization_batch_size'],
            ert=config['detector_ert'],
            window_size=config['detector_window_size'],
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )
        
    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_ewc(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Elastic Weight Consolidation' (EWC) personalization CL algorithm. This approach slows 
    down learning on weights that are important for previous data blocks by adding a quadratic penalty based on the Fisher Information Matrix.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'ewc_adaptation').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking training loss (including EWC penalty) and validation loss.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing personalization hyperparameters including 'ewc_lambda'.
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
        
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)
    
    # Fisher Information matrix data structures for EWC
    ewc_fisher = {}
    ewc_prev_params = {}
    ewc_lambda = config.get("ewc_lambda")
        
    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_ids = []
    n_steps = 0 # number of detector predictions after calibration
    n_detections = 0
    first_detection = None
    detection_timesteps = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            if config['inner_adapt'] == 'all':
                
                # Update also encoder: trian first with head for calibration and then predict feats for detector init
                # -> train first with head for calibration
                # -> then predict feats for detector init
                enc.train(); ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(enc(calibration_signals))
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                enc.eval(); ph.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
            elif config['inner_adapt'] == 'head':
                
                # Frozen encoder: predict features for head calibration and detector init
                enc.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
                ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(calibration_features)
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                ph.eval()
                
            else:
                raise ValueError('Inexistent adaptation mode, allowed is all or head!')
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
            if config['setup_type'] == 'drift':
                # Initialize drfit detector
                # NOTE: only after calibration completion
                reference_data = calibration_features.detach().clone()

                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
                
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
             
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
            
        # ----- Feature drift tracking -----
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]

        signals = torch.stack([x for x, _, _ in sample_batch]).to(device)
        targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
        
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals) 
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            for i in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[i].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
                    n_steps += 1 
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == config['calibration_phase_size']:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > config['calibration_phase_size']:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": 'head',
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                else:
                    raise ValueError("Step idx should not be less than calibration phase size at this point!")
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updated during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            # Encoder params for logging
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])
            total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in ph.parameters())

            # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
            if step_idx == config['calibration_phase_size']:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > config['calibration_phase_size'] else config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Loss Function
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # Training data/labels are already prepared for the current batch
            for step in range(config["personalization_steps"]):

                # Train prediction head
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)
                
                # EWC loss with Fisher Information Matrix
                ewc_loss = 0.0
                for n, p in list(ph.named_parameters()):
                    if n in ewc_fisher:
                        ewc_loss += (ewc_fisher[n] * (p - ewc_prev_params[n]).pow(2)).sum()
                loss = loss + ewc_lambda * ewc_loss
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()

            # Set to eval mode after adaptation
            ph.eval()
               
            # Fisher Matrix update for EWC
            ewc_fisher = {}

            params = [
                (n, p) for n, p in list(ph.named_parameters())
                if p.requires_grad
            ]

            for n, p in params:
                ewc_fisher[n] = torch.zeros_like(p)

            ph.eval()
            ph.zero_grad()
            out = ph(train_features)
            loss = criterion(out, train_targets)
            loss.backward()

            for n, p in params:
                if p.grad is not None:
                    ewc_fisher[n] += p.grad.data.pow(2)

            for n in ewc_fisher:
                ewc_fisher[n] /= len(train_features)

            ewc_prev_params = {
                n: p.detach().clone()
                for n, p in list(ph.named_parameters())
                if p.requires_grad
            }
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1   
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)

    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    if config['setup_type'] == 'drift':    
        
        print(f"[Personalization] Total detections with drift-aware updates: {n_detections} out of {n_steps} steps with drift detection after calibration ✓")
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{config['detector_ert']}_w{config['detector_window_size']}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=config['calibration_phase_size'],
            batch_size=config['personalization_batch_size'],
            ert=config['detector_ert'],
            window_size=config['detector_window_size'],
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )
        
    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_agem(baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Averaged Gradient Episodic Memory' (A-GEM) personalization CL algorithm. 
    This method ensures that the update gradient for the current data block does not 
    increase the loss on a reference set of previous data (stored in a replay buffer). 
    If a conflict is detected (negative dot product), the gradient is projected onto 
    the normal plane of the reference gradient.
    
    Parameters
    ------------
    baseline (str): 
        Identifier string for the specific baseline experiment (e.g., 'agem_adaptation').
        
    dataset (PhysioDataset): 
        The dataset instance providing subject-specific physiology signals and blood pressure labels.
        
    subject_id (int/str): 
        The unique identifier for the subject being personalized.
        
    subj_dir (str): 
        The root directory for saving subject-specific logs, CSVs, and visualization plots.
        
    writer (SummaryWriter): 
        TensorBoard logger for tracking training loss and gradient projection events.
        
    device (torch.device): 
        The computational device (CPU/CUDA) used for model training and inference.
        
    config (dict): 
        Configuration dictionary containing personalization hyperparameters like 'replay_buffer_size'.
        
    Returns
    ------------
    output param 1:
        A dictionary containing lists of SBP/DBP MAE and standard deviations calculated per block.
        
    output param 2:
        A tuple of concatenated (predictions, targets) as numpy arrays for the entire run.
        
    output param 3:
        A dictionary of SBP Continual Learning metrics (Average Accuracy, Backward Transfer).
        
    output param 4:
        A dictionary of DBP Continual Learning metrics (Average Accuracy, Backward Transfer).   
    """
    # ---- INITIALIZATION ----
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    if config['plot_personalization']:
        # Plot subject blocks and SBP/DBP/MAP drifts
        blocks = dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
        
        plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)
    
    # Feature replay buffer
    replay_buffer = ReservoirReplayBuffer(max_size=config.get('replay_buffer_size'))
    
    # Prepare data structures for evaluation  w/ online TTA
    
    # Track past batches for AE/BWT metrics
    # -> only during Online TTA
    past_batches = []
    
    # AE/BWT data structures
    T = config['num_batches'] * config['num_blocks'] - config['calibration_phase_size']
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per baseline
    param_update_log = []
    total_params = sum(p.numel() for p in Model(enc, ph).parameters())
    
    # Track targets per baseline, to do drfit analysis after perosnalization
    targets_log = []
    predictions_log = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_ids = []
    n_steps = 0 # number of detector predictions after calibration
    n_detections = 0
    first_detection = None
    detection_timesteps = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for batch_info in dataset.get_subject_blocks(
            subject_id, window_length=config['input_seq_len_s'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        ):
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            sample_ids = batch_info["sample_ids"]
            calibration_ids.extend(sample_ids)
            
            sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
            targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
            
            # Targets tracking for drift analysis (post hoc)
            targets_log.append({
                "step_idx": step_idx,
                "sbp_values": targets[:, 0].detach().cpu().numpy().tolist(),
                "dbp_values": targets[:, 1].detach().cpu().numpy().tolist(),
            })
            
            # No parameters are updated
            n_updated = 0

            param_update_log.append({
                "step_idx": step_idx,
                "block_idx": batch_info['block_idx'],
                "batch_idx": batch_info['batch_idx'],
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
    
            # Stack calibration data
            calibration_batch = [dataset.__getitem__(sid) for sid in calibration_ids]

            calibration_signals = torch.stack([x for x, _, _ in calibration_batch]).to(device)
            calibration_targets = torch.stack([y for _, y, _ in calibration_batch]).to(device)

            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )

            # Encoder params for logging
            n_updated_calibration = sum(p.numel() for group in opt.param_groups for p in group['params'])
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            if config['inner_adapt'] == 'all':
                
                # Update also encoder: trian first with head for calibration and then predict feats for detector init
                # -> train first with head for calibration
                # -> then predict feats for detector init
                enc.train(); ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(enc(calibration_signals))
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                enc.eval(); ph.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
            elif config['inner_adapt'] == 'head':
                
                # Frozen encoder: predict features for head calibration and detector init
                enc.eval()
                with torch.no_grad():
                    calibration_features = enc(calibration_signals)
                    
                ph.train()
                for step in range(config["personalization_steps"]):
                    out = ph(calibration_features)
                    loss = criterion(out, calibration_targets)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                
                ph.eval()
                
            else:
                raise ValueError('Inexistent adaptation mode, allowed is all or head!')
            
            # Predict calibration data to get outputs for logging
            previous_steps = 0
            while previous_steps < step_idx:
                with torch.no_grad():
                    calibration_outputs = ph(calibration_features[config['personalization_batch_size'] * previous_steps: config['personalization_batch_size'] * (previous_steps + 1)])
                    predictions_log.append({
                        "step_idx": previous_steps,
                        "sbp_values": calibration_outputs[:, 0].detach().cpu().numpy().tolist(),
                        "dbp_values": calibration_outputs[:, 1].detach().cpu().numpy().tolist(),
                    })
                previous_steps += 1
            
            if config['setup_type'] == 'drift':
                # Initialize drfit detector
                # NOTE: only after calibration completion
                reference_data = calibration_features.detach().clone()

                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=config['detector_ert'],
                        window_size=config['detector_window_size'],
                        n_bootstraps=config['detector_n_bootstraps'],
                        backend='pytorch',
                        verbose=False
                    )
                else:
                    raise ValueError('Inexistent drift detector type, allowed is mmd or lsdd!')

                print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
                
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        sample_ids = batch_info["sample_ids"]

        # Evaluate on CURRENT batch (before adaptation)
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_sample_set(
                enc, ph, dataset,
                sample_ids,   
                device,
                config
            )

        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)

        # Log targets/predictions for AE/BWT
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
        
        # Log targets/predictions for drift analysis
        targets_log.append({
            "step_idx": step_idx,
            "sbp_values": tgts_before[:, 0].tolist(),
            "dbp_values": tgts_before[:, 1].tolist(),
        })
        predictions_log.append({
            "step_idx": step_idx,
            "sbp_values": outs_before[:, 0].tolist(),
            "dbp_values": outs_before[:, 1].tolist(),
        })
            
        # ----- Feature drift tracking -----
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]

        signals = torch.stack([x for x, _, _ in sample_batch]).to(device)
        targets = torch.stack([y for _, y, _ in sample_batch]).to(device)
        
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals) 
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            for i in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[i].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
                    n_steps += 1  
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == config['calibration_phase_size']:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > config['calibration_phase_size']:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
                        "block_idx": batch_info['block_idx'],
                        "batch_idx": batch_info['batch_idx'],
                        "update_mode": 'head',
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                else:
                    raise ValueError("Step idx should not be less than calibration phase size at this point!")
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updated during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            # Encoder params for logging
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])
            total_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in ph.parameters())

            # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
            if step_idx == config['calibration_phase_size']:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > config['calibration_phase_size'] else config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })
            
            # Trainable parameters required by AGEM
            trainable_params = [
                p for p in list(enc.parameters()) + list(ph.parameters())
                if p.requires_grad
            ]

            # Loss Function
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # Training data/labels are already prepared for the current batch
            for step in range(config["personalization_steps"]):

                # Train prediction head
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)
                
                if len(replay_buffer) > 0:
                    # Current batch gradient
                    opt.zero_grad()
                    loss.backward()

                    grad_cur = torch.cat([
                            p.grad.view(-1)
                            for p in trainable_params
                            if p.grad is not None
                        ])

                    # Replay batch gradient
                    replay_feats, replay_tgts = replay_buffer.sample(train_features.shape[0])
                    if replay_feats is not None:
                        opt.zero_grad()
                        replay_out = ph(replay_feats.to(device))
                        replay_loss = criterion(replay_out, replay_tgts.to(device))
                        replay_loss.backward()
                        grad_ref = torch.cat([
                            p.grad.view(-1)
                            for p in trainable_params
                            if p.grad is not None
                        ])

                        dot = torch.dot(grad_cur, grad_ref)
                        if dot < 0:
                            proj = grad_cur - (dot / grad_ref.dot(grad_ref)) * grad_ref

                            # Write projected gradient back
                            idx = 0
                            for p in trainable_params:
                                if p.grad is not None:
                                    numel = p.grad.numel()
                                    p.grad.copy_(proj[idx:idx+numel].view_as(p))
                                    idx += numel

                    opt.step()
                else:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                train_loss = loss.item()
  
            # Set to eval mode after adaptation
            ph.eval()
            
            # Update replay buffer 
            with torch.no_grad():
                replay_buffer.add(features.detach().cpu(), targets.detach().cpu())
        
        # Add sample ids for AE/BWT
        past_batches.append(sample_ids)
        
        # ---------- Evaluate on ALL PREVIOUS batches (for BWT/AE) ----------
        for i, past_ids in enumerate(past_batches):

            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_sample_set(
                    enc, ph, dataset,
                    past_ids,  
                    device,
                    config
                )

            sbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = sbp_mae_i
            dbp_errors_matrix[step_idx - config['calibration_phase_size'], i] = dbp_mae_i
        
        # Increment step idx for the next block
        step_idx += 1
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AE, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✓")

    # Prepare per-block stats
    per_block_stats = {
        'sbp_mae': baseline_block_sbp_mae, 'sbp_std': baseline_block_sbp_std,
        'dbp_mae': baseline_block_dbp_mae, 'dbp_std': baseline_block_dbp_std,
    }

    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
                
    # Save update logs per baseline
    df_updates = pd.DataFrame(param_update_log)
    
    log_csv_path = os.path.join(baseline_path, "param_update_log.csv")
    df_updates.to_csv(log_csv_path, index=False)
    
    plot_path = os.path.join(baseline_path, "param_updates.png")
    plot_param_updates(df_updates, subject_id, plot_path)

    # Save targets/predictions log per baseline
    log_json_path = os.path.join(baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(targets_log, f)
    
    log_json_path = os.path.join(baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(predictions_log, f)

    if config['setup_type'] == 'drift':    
        
        print(f"[Personalization] Total detections with drift-aware updates: {n_detections} out of {n_steps} steps with drift detection after calibration ✓")
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{config['detector_ert']}_w{config['detector_window_size']}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=config['calibration_phase_size'],
            batch_size=config['personalization_batch_size'],
            ert=config['detector_ert'],
            window_size=config['detector_window_size'],
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalization(tensorboard_path, config, device):
    r"""
    Orchestrates the personalization and continual learning evaluation of blood pressure 
    estimation models. This function initializes datasets, loads a meta-pretrained model 
    (MAML), and evaluates multiple adaptation baselines across a stream of subject data.

    The function handles the full lifecycle of personalization:
    1.  **Initialization**: Sets up fresh and meta-trained learners.
    2.  **Drift Analysis**: Establishes thresholds for relative Blood Pressure drift.
    3.  **Cross-Subject Iteration**: For each subject, it runs several baselines (No Adapt, 
        Online, EWC, LwF, AGEM, etc.).
    4.  **Logging & Visualization**: Generates block-wise MAE plots, update summaries, 
        and Bland-Altman/AAMI performance metrics.
    5.  **Aggregation**: Computes Average Accuracy (AA) and Backward Transfer (BWT) across 
        the entire subject cohort.

    Parameters
    ------------
    tensorboard_path (str): 
        Path to save TensorBoard log files for monitoring training and validation loss.
        
    config (dict): 
        Global configuration containing hyperparameters, paths, and model specifications.
        Required keys include 'inner_adapt', 'pretrained_model_ckpt_path', and 'baselines'.
        
    device (torch.device): 
        Hardware accelerator (CPU/CUDA) to be used for model adaptation and inference.

    Returns
    ------------
    This function does not return values but writes CSV logs, metrics, and visualization 
    plots to the directory specified in `config['figure_path']`.
    """
    ## --- Personalization ---

    # Initialize dataset
    if 'aurora' in config['dataset_name'].lower():
        online_physio_dataset = AuroraOnlineSubjectDataset(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
    elif 'vital_db' in config['dataset_name'].lower():
        online_physio_dataset = OnlineSubjectDataset(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks']
        )
    else:
        raise ValueError(f"Dataset {config['dataset_name']} not recognized for personalization!")

    # Get test subjects
    personalization_subjects = online_physio_dataset.subjects_for_personalization

    # Baseline code source: https://github.com/GMvandeVen/continual-learning
    # -> if config['model_name'] == 'BIOT' or config['model_name'] == 'ResGruNet' or config['model_name'] == 'TCN':
    # USe a reduced set of baselines to avoid long runtimes (set this from argparse)
    baselines = config['baselines']
    
    # Include the new baseline key here as well so aggregation handles it
    global_outs_and_tgts = {
        b: [] for b in baselines 
    }
    
    # Create aggregate directory
    agg_dir = os.path.join(config['figure_path'], 'aggregate_metrics')
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)
        
    if config['setup_type'] == 'drift':
        print(f"[Personalization] Personalization performed with feature-based drift detection with percentile threshold {config['drift_threshold']}")
    
    for subject_counter, subject_id in enumerate(personalization_subjects):
        print(f"[Personalization] {subject_counter + 1}/{len(personalization_subjects)} personalizing model on subject {subject_id}")
        
        subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        os.makedirs(subj_dir, exist_ok=True)
        
        # Setup Tensorboard
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, f"subject_{subject_id}"))
        
        sbp_baseline_metrics = {}
        dbp_baseline_metrics = {}
        per_block_stats = {}
        outs_and_tgts = {}
        
        # Personalize each different baseline
        for b in baselines:
            print(f"[Personalization] Personalizing baseline {b} on subject {subject_id}")

            if b == 'no_adapt':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_no_adapt(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'calibration_only':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_calibration_only(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'online':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_online(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'online_from_scratch':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_online_from_scratch(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'feature_replay':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_feature_replay(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'lwf':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_lwf(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'ewc':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_ewc(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'agem':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_agem(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            else:    
                raise ValueError('Invalid baseline name!')
            
            # Collect results
            sbp_baseline_metrics[b] = baseline_sbp_baseline_metrics
            dbp_baseline_metrics[b] = baseline_dbp_baseline_metrics
            per_block_stats[b] = baseline_per_block_stats
            outs_and_tgts[b] = baseline_outs_and_tgts
            
        writer.close()
        
        if outs_and_tgts is None:
            raise ValueError("Outputs and targets for a subject cannot be None ...")

        subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        # --- Save baseline_metrics as CSV (AA and BWT per baseline) ---
        sbp_df_metrics = pd.DataFrame.from_dict(sbp_baseline_metrics, orient='index')
        sbp_csv_path = os.path.join(subj_dir, f'subject_{subject_id}_sbp_baseline_metrics.csv')
        sbp_df_metrics.to_csv(sbp_csv_path, index=True)
        dbp_df_metrics = pd.DataFrame.from_dict(dbp_baseline_metrics, orient='index')
        dbp_csv_path = os.path.join(subj_dir, f'subject_{subject_id}_dbp_baseline_metrics.csv')
        dbp_df_metrics.to_csv(dbp_csv_path, index=True)
        print(f"[Personalization] Saved SBP/DBP baseline metrics CSV to {sbp_csv_path} ✓")

        # Plot and save MAE per block using the standalone function
        plot_blockwise_mae(per_block_stats=per_block_stats, subject_id=subject_id, index_to_plot='sbp', savepath=os.path.join(subj_dir, f'subject_{subject_id}_sbp_blockwise_mae.png'))
        plot_blockwise_mae(per_block_stats=per_block_stats, subject_id=subject_id, index_to_plot='dbp', savepath=os.path.join(subj_dir, f'subject_{subject_id}_dbp_blockwise_mae.png'))

        # ---- Call metric plots for each baseline ----
        for b, (outs, tgts) in outs_and_tgts.items():
            if outs.shape[0] > 0:
                print(f"[Personalization] {subject_counter + 1}/{len(personalization_subjects)} Results for {subject_id} with baseline {b}")
                call_metric(tgts, outs, config, os.path.join(subj_dir, b), plot=True)
                # append to global for aggregated metrics later
                if b in global_outs_and_tgts:
                    global_outs_and_tgts[b].append((outs, tgts))

        # --- Extract and plot number of updates per baseline ---
        # Using only SBP as DBP is the same
        updates_per_baseline = {}
        for b in baselines:
            updates_log = pd.read_csv(os.path.join(subj_dir, b, 'param_update_log.csv'))
            updates_per_baseline[b] = (updates_log['n_updated_params'] > 0).sum()
        update_fig_path = os.path.join(subj_dir, f"subject_{subject_id}_update_summary.png")
        plot_update_summary_table(updates_per_baseline, update_fig_path)
        print(f"[Personalization] Saved update summary figure for subject {subject_id} ✓")

        print(f"[Personalization] Personalization on {subject_counter + 1}/{len(personalization_subjects)} subject completed ✓")

        if config['num_personalization_subjects'] > 0 and subject_counter > config['num_personalization_subjects']:
            break

    # ---- Aggregate AA and BWT across subjects ----
    all_metrics_sbp = {b: {"AE": [], "BWT": []} for b in global_outs_and_tgts.keys()}
    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_sbp_baseline_metrics.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            for b in df.index:
                if "AE" in df.columns and not pd.isna(df.loc[b, "AE"]):
                    all_metrics_sbp[b]["AE"].append(df.loc[b, "AE"])
                if "BWT" in df.columns and not pd.isna(df.loc[b, "BWT"]):
                    all_metrics_sbp[b]["BWT"].append(df.loc[b, "BWT"])
                    
    all_metrics_dbp = {b: {"AE": [], "BWT": []} for b in global_outs_and_tgts.keys()}
    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_dbp_baseline_metrics.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            for b in df.index:
                if "AE" in df.columns and not pd.isna(df.loc[b, "AE"]):
                    all_metrics_dbp[b]["AE"].append(df.loc[b, "AE"])
                if "BWT" in df.columns and not pd.isna(df.loc[b, "BWT"]):
                    all_metrics_dbp[b]["BWT"].append(df.loc[b, "BWT"])
                    
    # Compute mean/std across subjects for each baseline
    sbp_agg_rows = []
    for b, vals in all_metrics_sbp.items():
        if len(vals["AE"]) > 0:
            ae_mean, ae_std = np.mean(vals["AE"]), np.std(vals["AE"])
            bwt_mean, bwt_std = np.mean(vals["BWT"]), np.std(vals["BWT"])
            sbp_agg_rows.append({"Baseline": b, "AE_mean": ae_mean, "AE_std": ae_std,
                             "BWT_mean": bwt_mean, "BWT_std": bwt_std})
            print(f'[Personalization] Aggregated personalization results (AE/BWT) for SBP with baseline {b}')
            print(f'\t- AE: {ae_mean:.4f} ± {ae_std:.4f}')
            print(f'\t- BWT: {bwt_mean:.4f} ± {bwt_std:.4f}')

    sbp_df_agg = pd.DataFrame(sbp_agg_rows)
    sbp_agg_csv_path = os.path.join(agg_dir, 'sbp_aggregate_baseline_metrics.csv')
    sbp_df_agg.to_csv(sbp_agg_csv_path, index=False)
    
    dbp_agg_rows = []
    for b, vals in all_metrics_dbp.items():
        if len(vals["AE"]) > 0:
            ae_mean, ae_std = np.mean(vals["AE"]), np.std(vals["AE"])
            bwt_mean, bwt_std = np.mean(vals["BWT"]), np.std(vals["BWT"])
            dbp_agg_rows.append({"Baseline": b, "AE_mean": ae_mean, "AE_std": ae_std,
                             "BWT_mean": bwt_mean, "BWT_std": bwt_std})
            print(f'[Personalization] Aggregated personalization results (AE/BWT) for DBP with baseline {b}')
            print(f'\t- AE: {ae_mean:.4f} ± {ae_std:.4f}')
            print(f'\t- BWT: {bwt_mean:.4f} ± {bwt_std:.4f}')

    dbp_df_agg = pd.DataFrame(dbp_agg_rows)
    dbp_agg_csv_path = os.path.join(agg_dir, 'dbp_aggregate_baseline_metrics.csv')
    dbp_df_agg.to_csv(dbp_agg_csv_path, index=False)

    # ---- Aggregate call_metric across subjects ----
    for b, data_list in global_outs_and_tgts.items():
        if len(data_list) > 0:
            print(f'[Personalization] Aggregated personalization results (BHS/AAMI/Bland-Altman/R²) for the baseline {b}')
            all_outs = np.concatenate([o for o, _ in data_list], axis=0)
            all_tgts = np.concatenate([t for _, t in data_list], axis=0)
            call_metric(all_tgts, all_outs, config, os.path.join(agg_dir, f"aggregate_{b}_metrics"), plot=True)

    print(f"[Personalization] Saved aggregated metrics to {agg_dir} ✓")

    print(f"[Personalization] Personalization completed ✓")