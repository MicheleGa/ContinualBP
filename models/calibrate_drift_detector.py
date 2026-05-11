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
from models.personalizer import eval_model_on_sample_set
from component_factory import ReservoirReplayBuffer, Model
from training_utils.helpers import get_encoder_architecture, get_prediction_head_architecture, build_inner_optimizer
from training_utils.metrics import compute_transfer_metrics_from_matrix
from data.dataset import PhysioDataset
from data.online_dataset import OnlineSubjectDataset
from data.preprocessing_utils.data_visualization import plot_subject_annotation_blocks, plot_drift_calibration_summary, plot_param_updates
    

def subject_calibration_with_feature_replay(baseline, dataset, subject_id, subj_dir, calibration_phase_size, ert, window_size, n_bootstraps, writer, device, config):
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
    T = config['num_batches'] * config['num_blocks'] - calibration_phase_size
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
        if calibration_phase_size > 0 and step_idx < calibration_phase_size:
            
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
                "update_mode": config['inner_adapt'],
                "n_updated_params": n_updated,
                "total_params": total_params,
                "fraction_updated": n_updated / total_params
            })

            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if calibration_phase_size > 0 and step_idx == calibration_phase_size:
    
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
                        ert=ert,
                        window_size=window_size,
                        n_bootstraps=n_bootstraps,
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data.cpu().numpy(),
                        ert=ert,
                        window_size=window_size,
                        n_bootstraps=n_bootstraps,
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
                
                # Increment n_steps here to count the detector predictions (also after the detector is reinit. after an update)
                n_steps += 1    
                
                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= window_size:
                    if drift_flag == 1:
                        n_detections += 1
                        if first_detection is None:
                            first_detection = n_steps
                            
                        global_window_idx = step_idx * config['personalization_batch_size'] + i
                        detection_timesteps.append(global_window_idx)
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts 
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True
        elif config.get("setup_type") == "drift":
            # Use detector for each adaptive baseline
            if not do_adapt:
                # We reach this part of code only when step_idx is equal to or greater than calibration phase size or 
                if step_idx == calibration_phase_size:
                    # No parameters are updated exceet for the calibraiton ones
                    n_updated = n_updated_calibration

                    param_update_log.append({
                        "step_idx": step_idx,
                        "update_mode": config['inner_adapt'],
                        "n_updated_params": n_updated,
                        "total_params": total_params,
                        "fraction_updated": n_updated / total_params
                    })
                elif step_idx > calibration_phase_size:
                    # No parameters are updated
                    n_updated = 0

                    param_update_log.append({
                        "step_idx": step_idx,
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
            if step_idx == calibration_phase_size:
                n_updated += n_updated_calibration
                
            param_update_log.append({
                "step_idx": step_idx,
                "update_mode": 'head' if step_idx > calibration_phase_size else config['inner_adapt'],
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
            if config['setup_type'] == 'drift' and len(replay_buffer) > (config['personalization_batch_size'] * calibration_phase_size):
                
                drift_detector = None
                
                # Re-initialize drfit detector
                # NOTE: only after having updated the replay buffer
                reference_data = torch.stack(replay_buffer.features).numpy().copy()
                
                if config['drift_detector_type'] == 'mmd':
                    drift_detector = MMDDriftOnline(
                        x_ref=reference_data,
                        ert=ert,
                        window_size=window_size,
                        n_bootstraps=n_bootstraps,
                        backend='pytorch',
                        verbose=False
                    )
                elif config['drift_detector_type'] == 'lsdd':
                    drift_detector = LSDDDriftOnline(
                        x_ref=reference_data,
                        ert=ert,
                        window_size=window_size,
                        n_bootstraps=n_bootstraps,
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

            sbp_errors_matrix[step_idx - calibration_phase_size, i] = sbp_mae_i
            dbp_errors_matrix[step_idx - calibration_phase_size, i] = dbp_mae_i
        
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
        
        print(f"[Personalization] Total detections with drift-aware updates: {len(detection_timesteps)} out of {(step_idx - config['calibration_phase_size']) * config['personalization_batch_size']} steps with drift detection after calibration ✓")
        
        # Calibration summary plot
        sbp_ae, sbp_bwt = sbp_baseline_metrics['AE'], sbp_baseline_metrics['BWT']
        dbp_ae, dbp_bwt = dbp_baseline_metrics['AE'], dbp_baseline_metrics['BWT']
        
        plot_fname = f"ert{ert}_w{window_size}_detector_summary.png"
        plot_path  = os.path.join(baseline_path, plot_fname)
    
        plot_drift_calibration_summary(
            targets_log=targets_log,
            predictions_log=predictions_log,
            detection_timesteps=detection_timesteps,
            calibration_phase_size=calibration_phase_size,
            batch_size=config['personalization_batch_size'],
            ert=ert,
            window_size=window_size,
            sbp_ae=sbp_ae,
            dbp_ae=dbp_ae,
            sbp_bwt=sbp_bwt,
            dbp_bwt=dbp_bwt,
            save_path=plot_path,
        )

    return {
        "n_detections": n_detections,
        "first_detection": first_detection,
        "n_steps": n_steps,
        "sbp_ae":  sbp_ae,
        "dbp_ae":  dbp_ae,
        "sbp_bwt": sbp_bwt,
        "dbp_bwt": dbp_bwt,
    }


def calibrate_drift_detector(config, device):
    r"""
    Drift detector calibration with pretraining data
    
    Parameters
    ------------
    config (dict): 
        Global configuration containing hyperparameters, paths, and model specifications.
        Required keys include 'inner_adapt', 'pretrained_model_ckpt_path', and 'baselines'.
        
    device (torch.device): 
        Hardware accelerator (CPU/CUDA) to be used for model adaptation and inference.
    """
    
    # Initialize dataset
    if 'mimic_iii' in config['dataset_name'].lower():
        dataset = PhysioDataset(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
            pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
            meta_split_ratio=config['meta_train_split_ratio'],
            drift_aware=config['drift_aware'],
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            min_subject_sample_number=config['min_subject_sample_number']
        )

        # Fill in subject lists per split
        _, _, _, _ = dataset.get_pretraining_samplers()
        
        # Copy subjects for personalization before deleting dataset to save memory
        test_subjects = copy.deepcopy(dataset.pretraining_test_subjects)
        
        # Delete dataset to remove pointer to LMDB dataset
        del dataset
        
        online_physio_dataset = OnlineSubjectDataset(
            seed=config['seed'],
            lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
            fs=config['fs'],
            input_seq_len_s=config['input_seq_len_s'],
            ecg=config['ecg'],
            batch_size=config['personalization_batch_size'],
            num_batches=config['num_batches'],
            num_blocks=config['num_blocks'],
            subject_list=test_subjects
        )
    else:
        raise ValueError(f"Dataset {config['dataset_name']} not recognized for personalization!")

    # Get test subjects
    test_subjects = online_physio_dataset.subjects_for_personalization
    print(f"[Drift Detector Calibration] Calibrating drift detector with {len(test_subjects)} subjects from the pretraining test set of {config['dataset_name']}")
    
    # Parameters to tune
    param_grid = {
        "window_size": [2, 3, 4, 5],
        "ert": [16, 32, 64],
        "n_bootstraps": [500, 1000],
        "calibration_phase_size": [config['calibration_phase_size']],
    }
    
    # Logging
    results = []
    
    for calibration_phase_size in param_grid["calibration_phase_size"]:
        for window_size in param_grid["window_size"]:
            for ert in param_grid["ert"]:
                if window_size * 4 > ert:
                    print(f"Skipping config with W={window_size} and ERT={ert} since ERT should be > 4*W to allow for at least 4 windows between false alarms")
                    continue 
                for n_bootstraps in param_grid["n_bootstraps"]:

                    print(f"\n[Calibration] Testing config: W={window_size}, ERT={ert}, N_bootstraps={n_bootstraps}, Calibration Phase Size={calibration_phase_size}")

                    total_detections = 0
                    total_steps = 0
                    first_detection_steps = []
                    sbp_ae_list  = []
                    dbp_ae_list  = []
                    sbp_bwt_list = []
                    dbp_bwt_list = []

                    for subject_counter, subject_id in enumerate(test_subjects):
                        
                        try:
                            subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
                            os.makedirs(subj_dir, exist_ok=True)
                            
                            stats = subject_calibration_with_feature_replay(
                                baseline='feature_replay',
                                dataset=online_physio_dataset,
                                subject_id=subject_id,
                                subj_dir=subj_dir,
                                calibration_phase_size=calibration_phase_size,
                                ert=ert,
                                window_size=window_size,
                                n_bootstraps=n_bootstraps,
                                device=device,
                                config=config,
                                writer=None
                            )
                            print(stats, flush=True)
 
                            # Detector stats
                            total_detections += stats["n_detections"]
                            total_steps += stats["n_steps"]
                            if stats["first_detection"] is not None:
                                first_detection_steps.append(stats["first_detection"])
 
                            # Model-performance stats (guard against None)
                            if stats["sbp_ae"]  is not None: sbp_ae_list.append(stats["sbp_ae"])
                            if stats["dbp_ae"]  is not None: dbp_ae_list.append(stats["dbp_ae"])
                            if stats["sbp_bwt"] is not None: sbp_bwt_list.append(stats["sbp_bwt"])
                            if stats["dbp_bwt"] is not None: dbp_bwt_list.append(stats["dbp_bwt"])
                                
                        except Exception as e:
                            print(f"Error during calibration of subject {subject_id}: {e}")
                            continue
                        
                    false_alarm_rate = total_detections / max(total_steps, 1)
                    empirical_ert    = 1.0 / max(false_alarm_rate, 1e-8)
 
                    results.append({
                        # Grid parameters
                        "n_bootstraps":          n_bootstraps,
                        "calibration_phase_size":calibration_phase_size,
                        "window_size":           window_size,
                        "ert_target":            ert,
                        # Detector behaviour
                        "empirical_ert":         empirical_ert,
                        "false_alarm_rate":      false_alarm_rate,
                        "avg_first_detection":   np.mean(first_detection_steps) if first_detection_steps else None,
                        "pct_subjects_detected": 100.0 * len(first_detection_steps) / max(len(test_subjects), 1),
                        # Model performance (AE = average per-step MAE on current batch)
                        "avg_sbp_ae":  np.mean(sbp_ae_list)  if sbp_ae_list  else None,
                        "avg_dbp_ae":  np.mean(dbp_ae_list)  if dbp_ae_list  else None,
                        # Continual-learning stability (positive BWT = forgetting)
                        "avg_sbp_bwt": np.mean(sbp_bwt_list) if sbp_bwt_list else None,
                        "avg_dbp_bwt": np.mean(dbp_bwt_list) if dbp_bwt_list else None,
                    })
    
    # Save results as CSV
    df = pd.DataFrame(results)
    df["ert_relative_error"] = (
        (df["empirical_ert"] - df["ert_target"]).abs() / df["ert_target"]
    )

    df.sort_values("ert_relative_error", inplace=True, ascending=True)
    save_path = os.path.join(config['detector_calibration_csv_path'], f"{config['drift_detector_type']}_calibration_results.csv")
    df.to_csv(save_path, index=False)