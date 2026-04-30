import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.maml import MAML
from component_factory import ReservoirReplayBuffer, SBPDriftDetector, FeatureDriftDetector, Model
from training_utils.helpers import get_encoder_architecture, get_prediction_head_architecture, build_inner_optimizer, count_parameters
from training_utils.metrics import call_metric, compute_transfer_metrics_from_matrix
from data.online_dataset import OnlineSubjectDataset
from data.preprocessing_utils.data_visualization import (
    plot_subject_annotation_runs, plot_blockwise_mae, plot_update_summary_table, 
    plot_sbp_drift_over_time, plot_sbp_abs_rel_drift_distributions, plot_param_updates
)


def analyze_annotation_stats(dataset, agg_dir, config):
    r"""
    Analyzes and logs blood pressure distribution statistics and Systolic Blood Pressure (SBP) 
    drift across personalization subjects to determine threshold percentiles.
    
    Parameters
    ------------
    dataset (PhysioDataset): 
        The dataset object containing physiology data and subject run information.
        
    agg_dir (str): 
        The directory path where aggregated drift distribution plots will be saved.
        
    config (dict): 
        Configuration dictionary containing hyperparameters such as window lengths, 
        batch sizes, and plotting flags.
        
    Returns
    ------------
    output param 1:
        The 50th percentile (low/medium threshold) of relative SBP drift across all subjects.
        
    output param 2:
        The 80th percentile (medium/high threshold) of relative SBP drift across all subjects.   
    """
    personalization_subjects = dataset.subjects_for_personalization
    
    for subject_counter, subject_id in enumerate(personalization_subjects):
        # ---- Prepare runs ----
        block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
        runs = dataset.get_subject_runs(
            subject_id,
            window_length=config['input_seq_len_s'],
            adapt_size=config['personalization_batch_size'],
            val_size=config['validation_batch_size'],
            block_length=block_length,
            split_blocks=config['split_blocks']
        )

        assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

        subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        os.makedirs(subj_dir, exist_ok=True)
        
        # Optional visualization
        if config.get('plot_personalization'):
            plot_subject_annotation_runs(
                dataset,
                subject_id,
                runs,
                show_bp_plot=False,
                savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_runs.jpg")
            )
        
        # ---- SBP Drift Detector (annotation-based) ----
        sbp_drift_detector = SBPDriftDetector()
        
        # Target Statistics Logging (for drift analysis)
        target_stats = {
            "mean": [],   # per block: [mean_SBP, mean_DBP]
            "std": []     # per block: [std_SBP, std_DBP]
        }
        
        for step_idx, block in enumerate(runs):
            r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
            
            # Ground truth analysis
            batch = [dataset.__getitem__(sid) for sid in block['train']]
            targets = torch.stack([t for _, t, _ in batch]).cpu().numpy()  # [N, 2]

            # Training stats tracking
            mean_t = np.mean(targets, axis=0)
            std_t = np.std(targets, axis=0)

            target_stats["mean"].append(mean_t)
            target_stats["std"].append(std_t)
            
            # ----- SBP drift tracking with detector (TRAINING only) -----
            # SBP drift only on training data, remove the validation part
            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            train_batch = [dataset.__getitem__(sid) for sid in train_ids]

            targets = torch.stack([t for _, t, _ in train_batch]).cpu().numpy()
            sbp_values = targets[:, 0]  # SBP only

            _, _ = sbp_drift_detector.update(
                sbp_values=sbp_values,
                step_idx=step_idx
            )
            
            # Test stats tracking
            batch = [dataset.__getitem__(sid) for sid in block['test']]
            targets = torch.stack([t for _, t, _ in batch]).cpu().numpy()  # [N, 2]

            mean_t = np.mean(targets, axis=0)
            std_t = np.std(targets, axis=0)

            target_stats["mean"].append(mean_t)
            target_stats["std"].append(std_t)
        
        # ---- Save SBP drift statistics ----
        df_sbp_drift = sbp_drift_detector.to_dataframe()

        drift_csv_path = os.path.join(
            subj_dir,
            f"subject_{subject_id}_sbp_drift_over_time.csv"
        )
        df_sbp_drift.to_csv(drift_csv_path, index=False)
        
        plot_sbp_drift_over_time(
            df_sbp_drift,
            subject_id,
            os.path.join(subj_dir, f"subject_{subject_id}_sbp_drift_over_time.png")
        )
        
        print(f"[Personalization] Saved SBP drift CSV to {drift_csv_path} ✅")

        # Save target statistics
        df_targets = pd.DataFrame({
            "mean_SBP": [m[0] for m in target_stats["mean"]],
            "mean_DBP": [m[1] for m in target_stats["mean"]],
            "std_SBP":  [s[0] for s in target_stats["std"]],
            "std_DBP":  [s[1] for s in target_stats["std"]],
        })
        target_stats_csv_path = os.path.join(
            subj_dir,
            f"subject_{subject_id}_target_stats_over_time.csv"
        )
        
        df_targets.to_csv(
            target_stats_csv_path,
            index=False
        )
        print(f"[Personalization] Saved target statistics CSV to {target_stats_csv_path} ✅")
                
    # ---- Aggregate relative/absolute drift across subjects ----
    all_dfs = []
    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_sbp_drift_over_time.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            all_dfs.append(df)
        else:
            raise RuntimeError(f"SBP drift CSV for subject {subject_id} not found at {subj_csv}.")
        
    if len(all_dfs) == 0:
        raise RuntimeError("No personalization CSVs found.")
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    plot_sbp_abs_rel_drift_distributions(
        master_df,
        savepath=agg_dir,
        filename_prefix="sbp_drift"
    )
    
    rel_drift = master_df["rel_drift_sbp"].values
    
    # Low/medium and medium/high thresholds
    low_thr  = np.percentile(rel_drift, 50)
    high_thr = np.percentile(rel_drift, 80)
    
    return low_thr, high_thr


def eval_model_on_runset(encoder, prediction_head, dataset, run_samples, device, config):
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
        
    run_samples (list): 
        A list of sample indices (IDs) representing the specific run to evaluate.
        
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

    all_outputs = []
    all_targets = []

    encoder.eval()
    prediction_head.eval()

    with torch.no_grad():
        B = config['validation_batch_size'] // config['split_blocks']
        for i in range(0, len(run_samples), B):
            batch_ids = run_samples[i:i + B]
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

def personalize_no_adapt(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Performs the 'no adapt' personalization baseline by evaluating a pretrained model across 
    sequential data blocks without updating weights to establish a performance floor.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial model containing the encoder and prediction head weights.
        
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

    # Savepath
    baseline_path = os.path.join(subj_dir, baseline)
    os.makedirs(baseline_path, exist_ok=True)
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = copy.deepcopy(pretrained_learner)
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    
    # Prepare data structures for evaluation 
    T = len(runs)
    sbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    dbp_errors_matrix = np.full((T, T), np.nan, dtype=float)
    baseline_block_sbp_mae = []
    baseline_block_sbp_std = []
    baseline_block_dbp_mae = []
    baseline_block_dbp_std = []
    baseline_outputs = []
    baseline_targets = []

    # Track updates per no_adapt (maintained for consistency with other baselines)
    param_update_log = []
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate on training data ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        param_update_log.append({
            "step_idx": step_idx,
            "r_idx": r_idx,
            "b_idx": b_idx,
            "s_idx": s_idx,
            "update_mode": config['inner_adapt'], # irrelevant for this baseline, maintained for consistency with other baselines
            "n_updated_params": 0,
            "total_params": 0,
            "fraction_updated": 0
        })
        
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
            
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_first_batch_finetune(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'First-Batch Fine-Tuning' personalization baseline where the model adapts 
    once to the first available data block (or the first block exceeding a drift threshold) 
    and then remains frozen for all subsequent blocks.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial meta-trained model containing the encoder and prediction head.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = pretrained_learner
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )

    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # First batch finetuned flag
    first_finetuned = False
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)

        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        # ---------- Adaptation ----------
        if not first_finetuned and do_adapt:
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
                    
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]

            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None
            
            # Encoder is always frozen, only head is trained
            enc.eval()
            ph.train()

            train_features = []
            train_targets = []

            val_features = []
            val_targets = []

            with torch.no_grad():
                # ---- Train features ----
                for sid in train_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)   # [1, D]
                    train_features.append(feat.squeeze(0))
                    train_targets.append(target)

                # ---- Val features ----
                for sid in val_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)
                    val_features.append(feat.squeeze(0))
                    val_targets.append(target)

            train_features = torch.stack(train_features)
            train_targets  = torch.stack(train_targets).to(device)

            val_features = torch.stack(val_features)
            val_targets  = torch.stack(val_targets).to(device)
            
            for step in range(config["personalization_steps"]):

                # Train
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                ph.eval()

                with torch.no_grad():
                    
                    out = ph(val_features)
                    loss = criterion(out, val_targets)

                    val_loss = loss.item()    
                    
                    writer.add_scalar(
                        tag=f"val_loss/{baseline}/block_{step_idx}",
                        scalar_value=val_loss,
                        global_step=step
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_enc_state = copy.deepcopy(enc.state_dict())
                        best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
                
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
            
            # Once the First-Batch-Fine-Tune baseline is adapted, it is frozen and only used in inference
            for p in list(enc.parameters()) + list(ph.parameters()):
                p.requires_grad = False
            enc.eval(); ph.eval()
            first_finetuned = True 
        
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics


def personalize_online(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Online' personalization baseline where the model continuously adapts to new 
    data blocks as they arrive, either at every step or based on a drift detection trigger.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial meta-trained model containing the encoder and prediction head.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = pretrained_learner
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)

    # Feature-based Drift Detector 
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )
    
    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        # Feature-based Drift Detection 
        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
                
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]

            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None
            
            # Encoder is always frozen, only head is trained
            enc.eval()
            ph.train()

            train_features = []
            train_targets = []

            val_features = []
            val_targets = []

            with torch.no_grad():
                # ---- Train features ----
                for sid in train_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)   # [1, D]
                    train_features.append(feat.squeeze(0))
                    train_targets.append(target)

                # ---- Val features ----
                for sid in val_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)
                    val_features.append(feat.squeeze(0))
                    val_targets.append(target)

            train_features = torch.stack(train_features)
            train_targets  = torch.stack(train_targets).to(device)

            val_features = torch.stack(val_features)
            val_targets  = torch.stack(val_targets).to(device)
            
            for step in range(config["personalization_steps"]):

                # Train
                ph.train()
                
                out = ph(train_features)
                loss = criterion(out, train_targets)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = loss.item()

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                ph.eval()

                with torch.no_grad():
                    
                    out = ph(val_features)
                    loss = criterion(out, val_targets)

                    val_loss = loss.item()    
                    
                    writer.add_scalar(
                        tag=f"val_loss/{baseline}/block_{step_idx}",
                        scalar_value=val_loss,
                        global_step=step
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_enc_state = copy.deepcopy(enc.state_dict())
                        best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
                
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
        
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_online_from_scratch(fresh_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Online From Scratch' personalization baseline. Unlike the 'online' 
    baseline, this method begins with a randomly initialized (fresh) model and 
    continually trains it on subject-specific data blocks as they arrive.
    
    Parameters
    ------------
    fresh_learner (MAML or Model): 
        A non-pretrained, randomly initialized model instance used as the starting point.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = copy.deepcopy(fresh_learner)
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    
    # Feature-based Drift Detector
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )

    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)

        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )    
            
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]

            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None
            
            for step in range(config["personalization_steps"]):

                # Train
                enc.train()
                ph.train()
                
                train_loss_accum = 0.0
                train_batches = 0

                for i in range(0, len(train_ids), config["personalization_batch_size"] // config['split_blocks']):
                    batch_ids = train_ids[i:i + config["personalization_batch_size"] // config['split_blocks']]
                    batch = [dataset.__getitem__(sid) for sid in batch_ids]

                    signals = torch.stack([s for s, _, _ in batch]).to(device)
                    targets = torch.stack([t for _, t, _ in batch]).to(device)

                    out = ph(enc(signals))
                    tgts = targets

                    loss = criterion(out, tgts)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    train_loss_accum += loss.item()
                    train_batches += 1

                train_loss = train_loss_accum / max(train_batches, 1)

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                enc.eval()
                ph.eval()

                val_loss_accum = 0.0
                val_batches = 0

                with torch.no_grad():
                    for i in range(0, len(val_ids), config["personalization_batch_size"] // config['split_blocks']):
                        batch_ids = val_ids[i:i + config["personalization_batch_size"] // config['split_blocks']]
                        batch = [dataset.__getitem__(sid) for sid in batch_ids]

                        signals = torch.stack([s for s, _, _ in batch]).to(device)
                        targets = torch.stack([t for _, t, _ in batch]).to(device)

                        out = ph(enc(signals))
                        loss = criterion(out, targets)

                        val_loss_accum += loss.item()
                        val_batches += 1

                val_loss = val_loss_accum / max(val_batches, 1)

                writer.add_scalar(
                    tag=f"val_loss/{baseline}/block_{step_idx}",
                    scalar_value=val_loss,
                    global_step=step
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_enc_state = copy.deepcopy(enc.state_dict())
                    best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
            
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
        
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_feature_replay(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Feature Replay' personalization Cl algorithm that mitigates catastrophic 
    forgetting by storing previously seen latent features in a reservoir buffer and 
    interleaving them with current data during adaptation.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial meta-trained model containing the encoder and prediction head.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = pretrained_learner
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    
    # Feature replay buffer
    replay_buffer = ReservoirReplayBuffer(max_size=config.get('replay_buffer_size'))
    
    # Feature-based Drift Detector
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )

    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)

        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
                
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]

            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None
                    
            # Encoder is always frozen, only head is trained
            enc.eval()
            ph.train()

            train_features = []
            train_targets = []

            val_features = []
            val_targets = []

            with torch.no_grad():
                # ---- Train features ----
                for sid in train_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)   # [1, D]
                    train_features.append(feat.squeeze(0))
                    train_targets.append(target)

                # ---- Val features ----
                for sid in val_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)
                    val_features.append(feat.squeeze(0))
                    val_targets.append(target)

            train_features = torch.stack(train_features)
            train_targets  = torch.stack(train_targets).to(device)

            val_features = torch.stack(val_features)
            val_targets  = torch.stack(val_targets).to(device)
            
            for step in range(config["personalization_steps"]):

                # Train
                ph.train()
                
                # Reservoir sampling for feature replay
                replay_feats, replay_tgts = replay_buffer.sample(len(train_features))
                    
                # Replayed feats can be none on the first update as there are no rpevious features to replay
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

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                ph.eval()

                with torch.no_grad():
                    
                    out = ph(val_features)
                    loss = criterion(out, val_targets)

                    val_loss = loss.item()    
                    
                    writer.add_scalar(
                        tag=f"val_loss/{baseline}/block_{step_idx}",
                        scalar_value=val_loss,
                        global_step=step
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_enc_state = copy.deepcopy(enc.state_dict())
                        best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
                
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
            
            # Update replay buffer
            with torch.no_grad():
                replay_buffer.add(train_features.detach().cpu(), train_targets.detach().cpu())
    
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_lwf(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Learning without Forgetting' (LwF) personalization CL algorithm. This approach 
    combines a standard supervised loss on new data with a distillation loss that 
    constrains the current model to mimic the outputs of its previous iteration, 
    thereby preserving prior knowledge without storing old data.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial meta-trained model containing the encoder and prediction head.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = pretrained_learner
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    lwf_lambda = config.get("lwf_lambda")
    
    # Feature-based Drift Detector
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )

    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)

        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        # ---------- Adaptation ----------
        if do_adapt:
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
                    
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None

            # Encoder is always frozen, only head is trained
            enc.eval()
            ph.train()

            train_features = []
            train_targets = []

            val_features = []
            val_targets = []

            with torch.no_grad():
                # ---- Train features ----
                for sid in train_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)   # [1, D]
                    train_features.append(feat.squeeze(0))
                    train_targets.append(target)

                # ---- Val features ----
                for sid in val_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)
                    val_features.append(feat.squeeze(0))
                    val_targets.append(target)

            train_features = torch.stack(train_features)
            train_targets  = torch.stack(train_targets).to(device)

            val_features = torch.stack(val_features)
            val_targets  = torch.stack(val_targets).to(device)
            
            # LwF teacher: it can be either the pretrained head or the previously updated head
            # Here we use the previously updated head as teacher
            teacher_ph  = copy.deepcopy(ph)

            for p in teacher_ph.parameters():
                p.requires_grad = False

            teacher_ph.eval()
            with torch.no_grad():
                teacher_out = teacher_ph(train_features)
            
            for step in range(config["personalization_steps"]):

                # Train
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

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                ph.eval()

                with torch.no_grad():
                    
                    out = ph(val_features)
                    loss = criterion(out, val_targets)

                    val_loss = loss.item()    
                    
                    writer.add_scalar(
                        tag=f"val_loss/{baseline}/block_{step_idx}",
                        scalar_value=val_loss,
                        global_step=step
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_enc_state = copy.deepcopy(enc.state_dict())
                        best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
                
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
        
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_ewc(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Elastic Weight Consolidation' (EWC) personalization CL algorithm. This approach slows 
    down learning on weights that are important for previous data blocks by adding a quadratic penalty based on the Fisher Information Matrix.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial meta-trained model containing the encoder and prediction head.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = pretrained_learner
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    ewc_fisher = {}
    ewc_prev_params = {}
    ewc_lambda = config.get("ewc_lambda")
        
    # Feature-based Drift Detector
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )

    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)

        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
                
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]

            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None
            
            # Encoder is always frozen, only head is trained
            enc.eval()
            ph.train()

            train_features = []
            train_targets = []

            val_features = []
            val_targets = []

            with torch.no_grad():
                # ---- Train features ----
                for sid in train_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)   # [1, D]
                    train_features.append(feat.squeeze(0))
                    train_targets.append(target)

                # ---- Val features ----
                for sid in val_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)
                    val_features.append(feat.squeeze(0))
                    val_targets.append(target)

            train_features = torch.stack(train_features)
            train_targets  = torch.stack(train_targets).to(device)

            val_features = torch.stack(val_features)
            val_targets  = torch.stack(val_targets).to(device)
            
            for step in range(config["personalization_steps"]):

                # Train
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

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                ph.eval()

                with torch.no_grad():
                    
                    out = ph(val_features)
                    loss = criterion(out, val_targets)

                    val_loss = loss.item()    
                    
                    writer.add_scalar(
                        tag=f"val_loss/{baseline}/block_{step_idx}",
                        scalar_value=val_loss,
                        global_step=step
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_enc_state = copy.deepcopy(enc.state_dict())
                        best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
            
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
                
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
        
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    return per_block_stats, outs_and_tgts, sbp_baseline_metrics, dbp_baseline_metrics 


def personalize_agem(pretrained_learner, baseline, dataset, subject_id, subj_dir, writer, device, config):
    r"""
    Executes the 'Averaged Gradient Episodic Memory' (A-GEM) personalization CL algorithm. 
    This method ensures that the update gradient for the current data block does not 
    increase the loss on a reference set of previous data (stored in a replay buffer). 
    If a conflict is detected (negative dot product), the gradient is projected onto 
    the normal plane of the reference gradient.
    
    Parameters
    ------------
    pretrained_learner (MAML or Model): 
        The initial meta-trained model containing the encoder and prediction head.
        
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
    
    # Prepare runs
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length,
        split_blocks=config['split_blocks']
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    # Initialize learner for a specific baseline
    learner = pretrained_learner
    enc, ph = learner.encoder.to(device), learner.prediction_head.to(device)
    replay_buffer = ReservoirReplayBuffer(max_size=config.get('replay_buffer_size'))
    
    # Feature-based Drift Detector
    feature_drift_detector = FeatureDriftDetector(
        stats_path=config['pretraining_feats_stats'],
        threshold=config['drift_threshold'],
        device=device
    )

    # Prepare data structures for evaluation 
    T = len(runs)
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
    model = Model(enc, ph)
    total_params = sum(p.numel() for p in model.parameters())
    
    # ---- PERSONALIZATION ----
    for step_idx, block in enumerate(runs):
        r_idx, b_idx, s_idx = block['r_idx'], block['b_idx'], block['s_idx']
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        sbp_mae_before, sbp_std_before, dbp_mae_before, dbp_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
        
        baseline_block_sbp_mae.append(sbp_mae_before)
        baseline_block_sbp_std.append(sbp_std_before)
        baseline_block_dbp_mae.append(dbp_mae_before)
        baseline_block_dbp_std.append(dbp_std_before)            
        baseline_outputs.append(outs_before)
        baseline_targets.append(tgts_before)
            
        # ----- Feature drift tracking (TRAINING only) -----
        # Remove validation indexes for drift calculation, ensuring consistency with the adaptation loop later
        num_ids = len(block["train"])
        split_idx = int(0.75 * num_ids)

        train_ids = block["train"][:split_idx]
        train_batch = [dataset.__getitem__(sid) for sid in train_ids]

        enc.eval()

        signals = torch.stack([x for x, _, _ in train_batch]).to(device)

        with torch.no_grad():
            features = enc(signals)   # [N, 128]

        score, drift = feature_drift_detector.update(features=features)

        # ---------- Decide adaptation ----------
        do_adapt = False
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if drift:
                do_adapt = True
            else:
                # No parameters are updated
                n_updated = 0

                param_update_log.append({
                    "step_idx": step_idx,
                    "r_idx": r_idx,
                    "b_idx": b_idx,
                    "s_idx": s_idx,
                    "update_mode": config['inner_adapt'],
                    "n_updated_params": n_updated,
                    "total_params": total_params,
                    "fraction_updated": n_updated / total_params
                })
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')

        if do_adapt:
            # ---------- Adaptation ----------
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
                
            n_updated = sum(p.numel() for group in opt.param_groups for p in group['params'])

            param_update_log.append({
                "step_idx": step_idx,
                "r_idx": r_idx,
                "b_idx": b_idx,
                "s_idx": s_idx,
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })
            
            # Trainable parameters required by AGEM
            trainable_params = [
                p for p in list(enc.parameters()) + list(ph.parameters())
                if p.requires_grad
            ]

            # Chronological train/val split
            num_ids = len(block["train"])
            split_idx = int(0.75 * num_ids)

            train_ids = block["train"][:split_idx]
            val_ids = block["train"][split_idx:]

            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )

            best_val_loss = float("inf")
            best_enc_state = None
            best_ph_state = None
            
            # Encoder is always frozen, only head is trained
            enc.eval()
            ph.train()

            train_features = []
            train_targets = []

            val_features = []
            val_targets = []

            with torch.no_grad():
                # ---- Train features ----
                for sid in train_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)   # [1, D]
                    train_features.append(feat.squeeze(0))
                    train_targets.append(target)

                # ---- Val features ----
                for sid in val_ids:
                    signal, target, _ = dataset[sid]
                    signal = signal.unsqueeze(0).to(device)
                    feat = enc(signal)
                    val_features.append(feat.squeeze(0))
                    val_targets.append(target)

            train_features = torch.stack(train_features)
            train_targets  = torch.stack(train_targets).to(device)

            val_features = torch.stack(val_features)
            val_targets  = torch.stack(val_targets).to(device)
            
            for step in range(config["personalization_steps"]):

                # Train
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
                    replay_feats, replay_tgts = replay_buffer.sample(len(train_features))
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

                writer.add_scalar(
                    tag=f"train_loss/{baseline}/r{r_idx}_b{b_idx}_s{s_idx}",
                    scalar_value=train_loss,
                    global_step=step
                )

                # Eval
                ph.eval()

                with torch.no_grad():
                    
                    out = ph(val_features)
                    loss = criterion(out, val_targets)

                    val_loss = loss.item()    
                    
                    writer.add_scalar(
                        tag=f"val_loss/{baseline}/block_{step_idx}",
                        scalar_value=val_loss,
                        global_step=step
                    )

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_enc_state = copy.deepcopy(enc.state_dict())
                        best_ph_state = copy.deepcopy(ph.state_dict()) 

            # Load best model
            if best_enc_state is not None:
                enc.load_state_dict(best_enc_state)
                ph.load_state_dict(best_ph_state)
                
            # Update detector statistics only after successful adaptation
            feature_drift_detector.update_stats(features)
                
            # Update replay buffer (pretrained version)
            with torch.no_grad():
                replay_buffer.add(train_features.detach().cpu(), train_targets.detach().cpu())
    
        # ---------- Evaluate on test data ----------
        # This evaluation is to log AA/BWT and to ensure the system always produce an output given the stream of data
        enc.eval(); ph.eval()
        
        # Previous blocks (for AA, BWT metrics)
        for i in range(step_idx+1):
            sbp_mae_i, sbp_std_i, dbp_mae_i, dbp_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
            sbp_errors_matrix[step_idx, i] = sbp_mae_i
            dbp_errors_matrix[step_idx, i] = dbp_mae_i
        
        # Current block (for predicting the current test batch)
        sbp_mae_after, sbp_std_after, dbp_mae_after, dbp_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
        baseline_block_sbp_mae.append(sbp_mae_after)
        baseline_block_sbp_std.append(sbp_std_after)
        baseline_block_dbp_mae.append(dbp_mae_after)
        baseline_block_dbp_std.append(dbp_std_after)            
        baseline_outputs.append(outs_after)
        baseline_targets.append(tgts_after)
        
                    
    # ---- AGGREGATED METRICS & LOGGING ----
    # Compute CL metrics (AA, BWT)
    sbp_baseline_metrics = compute_transfer_metrics_from_matrix(sbp_errors_matrix)
    dbp_baseline_metrics = compute_transfer_metrics_from_matrix(dbp_errors_matrix)
    
    # Save error matrices as CSV
    sbp_df_err = pd.DataFrame(sbp_errors_matrix)
    sbp_df_err.to_csv(os.path.join(baseline_path, "sbp_error_matrix.csv"), index=False)
    dbp_df_err = pd.DataFrame(dbp_errors_matrix)
    dbp_df_err.to_csv(os.path.join(baseline_path, "dbp_error_matrix.csv"), index=False)
        
    print(f"[Personalization] Saved error matrices for subject {subject_id}, baseline {baseline} ✅")

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

    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])

    # Initialize dataset
    online_physio_dataset = OnlineSubjectDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        block_length=block_length,
        valid_runs_number=config['valid_runs_number']
    )

    # Initialize encoder and prediction head (fresh weights)
    encoder = get_encoder_architecture(config)
    prediction_head = get_prediction_head_architecture(config)
    fresh_learner = Model(copy.deepcopy(encoder), copy.deepcopy(prediction_head))
    print(f"[Personalization] Fresh encoder & head initialized ✅")

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
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✅")

    # Count trainable parameters
    learner_trainable, learner_non_trainable = count_parameters(pretrained_learner)
    print(f"[Personalization] Parameter count:")
    print(f"\t- MAML learner total parameters: {(learner_trainable + learner_non_trainable):,}")

    # Get test subjects
    personalization_subjects = online_physio_dataset.subjects_for_personalization

    # Baselines
    # -> NOTE: code source https://github.com/GMvandeVen/continual-learning
    if config['model_name'] == 'BIOT' or config['model_name'] == 'ResGruNet' or config['model_name'] == 'TCN':
        # Reduced set of baselines for models from the literature to avoid long runtimes 
        baselines = [
            'no_adapt',
            'first_batch_finetune',
            'online',
            'online_from_scratch',
            'feature_replay'
        ]
    else:
        # Proto baseline with all the classical CL baselines
        baselines = [
            #'no_adapt',
            #'first_batch_finetune',
            #'online',
            #'online_from_scratch',
            'feature_replay',
            #'lwf',
            #'ewc',
            #'agem'
        ]
    
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
    
    # Ensure only head adaptation for now
    assert config['inner_adapt'] == 'head', "Full backbone updates during personalization are not supported yet!"

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
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    subj_dir=subj_dir, 
                    writer=writer,
                    device=device, 
                    config=config
                )
            elif b == 'first_batch_finetune':
                baseline_per_block_stats, baseline_outs_and_tgts, baseline_sbp_baseline_metrics, baseline_dbp_baseline_metrics = personalize_first_batch_finetune(
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
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
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
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
                    fresh_learner=copy.deepcopy(fresh_learner), 
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
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
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
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
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
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
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
                    pretrained_learner=copy.deepcopy(pretrained_learner), 
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
        print(f"[Personalization] Saved SBP/DBP baseline metrics CSV to {sbp_csv_path} ✅")

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
        print(f"[Personalization] Saved update summary figure for subject {subject_id} ✅")

        print(f"[Personalization] Personalization on {subject_counter + 1}/{len(personalization_subjects)} subject completed ✅")

        if config['num_personalization_subjects'] > 0 and subject_counter > config['num_personalization_subjects']:
            break

    # ---- Aggregate AA and BWT across subjects ----
    all_metrics_sbp = {b: {"AA": [], "BWT": []} for b in global_outs_and_tgts.keys()}
    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_sbp_baseline_metrics.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            for b in df.index:
                if "AA" in df.columns and not pd.isna(df.loc[b, "AA"]):
                    all_metrics_sbp[b]["AA"].append(df.loc[b, "AA"])
                if "BWT" in df.columns and not pd.isna(df.loc[b, "BWT"]):
                    all_metrics_sbp[b]["BWT"].append(df.loc[b, "BWT"])
                    
    all_metrics_dbp = {b: {"AA": [], "BWT": []} for b in global_outs_and_tgts.keys()}
    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_dbp_baseline_metrics.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            for b in df.index:
                if "AA" in df.columns and not pd.isna(df.loc[b, "AA"]):
                    all_metrics_dbp[b]["AA"].append(df.loc[b, "AA"])
                if "BWT" in df.columns and not pd.isna(df.loc[b, "BWT"]):
                    all_metrics_dbp[b]["BWT"].append(df.loc[b, "BWT"])
                    
    # Compute mean/std across subjects for each baseline
    sbp_agg_rows = []
    for b, vals in all_metrics_sbp.items():
        if len(vals["AA"]) > 0:
            aa_mean, aa_std = np.mean(vals["AA"]), np.std(vals["AA"])
            bwt_mean, bwt_std = np.mean(vals["BWT"]), np.std(vals["BWT"])
            sbp_agg_rows.append({"Baseline": b, "AA_mean": aa_mean, "AA_std": aa_std,
                             "BWT_mean": bwt_mean, "BWT_std": bwt_std})
            print(f'[Personalization] Aggregated personalization results (AA/BWT) for SBP with baseline {b}')
            print(f'\t- AA: {aa_mean:.4f} ± {aa_std:.4f}')
            print(f'\t- BWT: {bwt_mean:.4f} ± {bwt_std:.4f}')

    sbp_df_agg = pd.DataFrame(sbp_agg_rows)
    sbp_agg_csv_path = os.path.join(agg_dir, 'sbp_aggregate_baseline_metrics.csv')
    sbp_df_agg.to_csv(sbp_agg_csv_path, index=False)
    
    dbp_agg_rows = []
    for b, vals in all_metrics_dbp.items():
        if len(vals["AA"]) > 0:
            aa_mean, aa_std = np.mean(vals["AA"]), np.std(vals["AA"])
            bwt_mean, bwt_std = np.mean(vals["BWT"]), np.std(vals["BWT"])
            dbp_agg_rows.append({"Baseline": b, "AA_mean": aa_mean, "AA_std": aa_std,
                             "BWT_mean": bwt_mean, "BWT_std": bwt_std})
            print(f'[Personalization] Aggregated personalization results (AA/BWT) for DBP with baseline {b}')
            print(f'\t- AA: {aa_mean:.4f} ± {aa_std:.4f}')
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
            call_metric(all_tgts, all_outs, config, os.path.join(agg_dir, f"aggregate_{b}_metrics.png"), plot=True)

    print(f"[Personalization] Saved aggregated metrics to {agg_dir} ✅")

    print(f"[Personalization] Personalization completed ✅")
