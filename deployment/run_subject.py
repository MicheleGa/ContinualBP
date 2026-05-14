import os
import sys
import copy
import json
import pickle
import time
import resource   
import tracemalloc
import platform
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from deployment.deployment_helpers import build_inner_optimizer
from deployment.deployment_maml import MAML
from deployment.deployment_component_factory import BPRegressor, ReservoirReplayBuffer, Model
from deployment.Proto import Proto
from deployment.DriftDetectors import MMDDriftOnline, LSDDDriftOnline

def _peak_memory_mb() -> float:
    """Return peak RSS memory in MB.
 
    On Linux (Raspberry Pi) resource.getrusage gives the true peak since
    process start, including PyTorch tensor allocations.
    On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
    Falls back to tracemalloc (Python heap only) on Windows.
    NOTE: if you run other processes along the main one, they will be recorded too by the resource.getresourceusage function
    """
    if platform.system() in ("Linux", "Darwin"):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Linux":
            return usage / 1024.0          # KB → MB
        else:
            return usage / (1024.0 * 1024) # bytes → MB
    else:
        # Windows / unknown – best-effort via tracemalloc
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024.0 * 1024)
    

def run_subject(data_path, model_path, config_path, verbose=False):
    
    # Start memory tracking
    if platform.system() not in ("Linux", "Darwin"):
        tracemalloc.start()
    
    # Initialize profiling data structure
    prof = {
        # calibration phase (one-shot)
        'calibration_time_s': 0.0,
 
        # per TTA step (appended in loop)
        'per_step': {
            'prediction_s':         [],   # pred(feats) latency
            'feature_extraction_s': [],   # enc(signals) latency
            'drift_detection_s':    [],   # drift_detector.predict() loop latency
            'adaptation_s':         [],   # entire do_adapt block latency
            'total_step_s':         [],   # wall time for the full TTA step
            'did_adapt':            [],   # bool
            'n_drift_detections':   [],   # int – how many samples triggered drift
        },
 
        # filled at the end
        'peak_memory_mb':                       0.0,
        'n_steps':                              0,
        'n_adaptations':                        0,
        'adaptation_rate':                      0.0,
        'total_tta_time_s':                     0.0,
        'total_prediction_time_s':              0.0,
        'total_feature_extraction_time_s':      0.0,
        'total_drift_detection_time_s':         0.0,
        'total_adaptation_time_s':              0.0,
        'estimated_time_if_always_adapted_s':   0.0,
        'time_saved_by_drift_detection_s':      0.0,
    }
    
    device = torch.device("cpu")
    
    # Load data
    with open(data_path, "rb") as f:
        blocks_list = pickle.load(f)
    
    # Load config
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = Proto(
        ecg=config['ecg'], 
        fs=config['fs'], 
        input_seq_len_s=config['input_seq_len_s'],
        embed_dim=config['embed_dim']
    )
    prediction_head_pre = BPRegressor(input_dim=2 * config['embed_dim'] if config['ecg'] else config['embed_dim'], output_dim=config['output_dim'])
    pretrained_learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    pretrained_learner = MAML(
        pretrained_learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    # Load model
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    pretrained_learner.load_state_dict(ckpt['learner_state_dict'])
    pretrained_learner = pretrained_learner.to(device)
    pretrained_learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✓")

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)

    # Feature replay buffer
    replay_buffer = ReservoirReplayBuffer(seed=config['seed'], max_size=config['replay_buffer_size'])
    
    # Prepare data structures for evaluation 
    baseline_outputs = []
    baseline_targets = []
    
    # Drift detector placeholders
    drift_detector = None
    calibration_signals = []
    calibration_targets = []
    
    # Important for logging
    step_idx = 0
    
    # ---- PERSONALIZATION ----
    for block_idx in range(len(blocks_list)):
        block = blocks_list[block_idx]
        signals = torch.from_numpy(block["signals"]).float()
        targets = torch.from_numpy(block["targets"]).float()
        
        # CALIBRATION PHASE 
        if config['calibration_phase_size'] > 0 and step_idx < config['calibration_phase_size']:
            
            # Accumulate calibration batches
            calibration_signals.append(signals)
            calibration_targets.append(targets)
            
            # Increment step idx for the next block
            step_idx += 1
            continue
        
        if config['calibration_phase_size'] > 0 and step_idx == config['calibration_phase_size']:
            
            # Calibration starts
            t_cal_start = time.perf_counter()
            
            # Stack calibration data
            calibration_signals = torch.cat(calibration_signals, dim=0).to(device)
            calibration_targets = torch.cat(calibration_targets, dim=0).to(device)
            
            # Model adaptation
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode=config['inner_adapt'],
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
            criterion = (
                F.smooth_l1_loss
                if config["criterion"] == "SmoothL1Loss"
                else F.mse_loss
            )
            
            # During calibration, depending on the device resources, either all model parameters 
            # or only the head parameters can be updated 
            calibration_features = None
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
            
            # Calibration end
            prof['calibration_time_s'] = time.perf_counter() - t_cal_start
               
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        # TTA start
        t_step_start = time.perf_counter() 
        
        # ---------- Evaluate before adaptation ----------
        # -> ensure the system always produce an output given the stream of data
        # -> follow test-time-adaptation evaluation
        
        # ----- Feature extraction -----      
        
        # Profile feats. extraction before updates
        t_feat_start = time.perf_counter()
          
        # Extract features with the frozen encoder
        with torch.no_grad():
            features = enc(signals.to(device)) 
            
        # Feats. extraction end
        t_feat_end = time.perf_counter()
        feat_latency = t_feat_end - t_feat_start
        
        # Profile prediction inference before updates
        t_pred_start = time.perf_counter()
        
        # Evaluate on CURRENT batch (before adaptation)
        with torch.no_grad():
            outputs = ph(features)

        # Inference end
        t_pred_end = time.perf_counter()
        pred_latency = t_pred_end - t_pred_start

        # Log targets/predictions: SBP/DBP are on the frist and second position of out/tgt
        baseline_outputs.append(outputs.detach().cpu().numpy()[:, [0,1]])
        baseline_targets.append(targets.detach().cpu().numpy()[:, [0,1]])
        
        # ---------- Decide adaptation ----------
        do_adapt = False
        
        drift_latency = 0.0 # When we do not use the detector, this is 0
        
        if config['setup_type'] == 'drift':
            # Detect data drifts with detector
            
            # Drift detection start
            t_drift_start = time.perf_counter()
            for feat_idx in range(features.shape[0]):
                # Add batch dimension before prediction
                detection_report = drift_detector.predict(features[feat_idx].cpu().numpy())
                drift_flag = detection_report["data"]["is_drift"]

                # Log only after the minimum number of test samples have been seen (i.e. after the first window is filled)
                if drift_detector.t >= config['detector_window_size']:
                    if drift_flag == 1:
                        do_adapt = True # Update when a single sample is considered out of distribution to react quickly to drifts
        
            # Drift detection end
            t_drift_end = time.perf_counter()
            drift_latency = t_drift_end - t_drift_start 
        
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") == "fixed":
            do_adapt = True

        # Adaptatin latency
        adapt_latency = 0.0 # When we do not adapt this is 0
        
        if do_adapt:
            # ---------- Adaptation ----------
            
            # Adaptation starts
            t_adapt_start = time.perf_counter()
            
            # Use features and targets for adaptation
            train_features = features
            train_targets = targets.to(device)
            
            # Optimizer
            opt = build_inner_optimizer(
                adapted_encoder=enc, 
                adapted_head=ph, 
                base_lr=config['personalization_lr'], 
                mode='head', # only head is updated during online TTA
                opt_type=config['inner_opt'].lower(), 
                config=config
            )
            
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
            
            # Adaptation ends
            t_adapt_end = time.perf_counter()
            adapt_latency = t_adapt_end - t_adapt_start
                                
        t_step_end = time.perf_counter()
        step_latency = t_step_end - t_step_start
        
        # Record per-step profiling 
        prof['per_step']['prediction_s'].append(pred_latency)
        prof['per_step']['feature_extraction_s'].append(feat_latency)
        prof['per_step']['drift_detection_s'].append(drift_latency)
        prof['per_step']['adaptation_s'].append(adapt_latency)
        prof['per_step']['total_step_s'].append(step_latency)
        prof['per_step']['did_adapt'].append(do_adapt)
        
        # Increment step idx for the next block
        step_idx += 1
    
    # Aggregate profiling report
    prof['peak_memory_mb'] = _peak_memory_mb()
 
    n_steps = len(prof['per_step']['total_step_s'])
    n_adaptations = sum(prof['per_step']['did_adapt'])
 
    total_pred = sum(prof['per_step']['prediction_s'])
    total_feat = sum(prof['per_step']['feature_extraction_s'])
    total_drift = sum(prof['per_step']['drift_detection_s'])
    total_adapt = sum(prof['per_step']['adaptation_s'])
    total_tta = sum(prof['per_step']['total_step_s'])
 
    prof['n_steps']                          = n_steps
    prof['n_adaptations']                    = n_adaptations
    prof['adaptation_rate']                  = n_adaptations / n_steps if n_steps else 0.0
    prof['total_tta_time_s']                 = total_tta
    prof['total_prediction_time_s']          = total_pred
    prof['total_feature_extraction_time_s']  = total_feat
    prof['total_drift_detection_time_s']     = total_drift
    prof['total_adaptation_time_s']          = total_adapt
    
    # Counterfactual: how long would TTA have taken if we always adapted?
    # Use mean observed adaptation time as the per-step cost estimate.
    adapt_times_when_done = [
        t for t, did in zip(
            prof['per_step']['adaptation_s'],
            prof['per_step']['did_adapt']
        ) if did
    ]
    mean_adapt_time = (
        float(np.mean(adapt_times_when_done)) if adapt_times_when_done else 0.0
    )
    skipped_steps = n_steps - n_adaptations
    estimated_if_always = total_tta + skipped_steps * mean_adapt_time
 
    prof['estimated_time_if_always_adapted_s'] = estimated_if_always
    prof['time_saved_by_drift_detection_s'] = estimated_if_always - total_tta
    prof['adaptation_compute_fraction'] = (
        total_adapt / total_tta
        if total_tta > 0 else 0.0
    )
    prof['mean_step_latency_ms'] = (
        np.mean(prof['per_step']['total_step_s']) * 1e3
    )
    
    if verbose:
        # ── summary print ──────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("  PROFILING REPORT")
        print("=" * 72)

        print(f"  Calibration update          : {prof['calibration_time_s']*1e3:>8.1f} ms")
        print(f"  TTA steps                   : {n_steps:>8d}")
        print(f"  Adaptations triggered       : {n_adaptations:>8d}  "
            f"({prof['adaptation_rate']*100:.1f}%)")

        print(f"")

        print(f"  ── Per-step mean latency ────────────────────────────")
        print(f"  Prediction (before update)  : "
            f"{np.mean(prof['per_step']['prediction_s'])*1e3:>8.2f} ms")

        print(f"  Feature extraction          : "
            f"{np.mean(prof['per_step']['feature_extraction_s'])*1e3:>8.2f} ms")

        if config.get('setup_type') == 'drift':
            print(f"  Drift detection             : "
                f"{np.mean(prof['per_step']['drift_detection_s'])*1e3:>8.2f} ms")

            print(
                f"  Adaptation (when triggered) : "
                f"{mean_adapt_time*1e3:>8.2f} ms"
                if adapt_times_when_done else
                "  Adaptation (when triggered) :      n/a"
            )

            print(f"  Full TTA step               : "
                f"{np.mean(prof['per_step']['total_step_s'])*1e3:>8.2f} ms")

            print(f"")

            print(f"  ── Runtime Composition ──────────────────────────────")

            print(f"  Adaptation compute fraction : "
                f"{prof['adaptation_compute_fraction']*100:>8.2f} %")

            print(f"  Mean online step latency    : "
                f"{prof['mean_step_latency_ms']:>8.2f} ms")

            print(f"")

            print(f"  ── Totals ───────────────────────────────────────────")

            print(f"  Total TTA time              : {total_tta:>8.3f} s")

            print(f"  Total prediction time       : "
                f"{total_pred:>8.3f} s")

            print(f"  Total feature extraction    : "
                f"{total_feat:>8.3f} s")

            if config.get('setup_type') == 'drift':
                print(f"  Total drift detection       : "
                    f"{total_drift:>8.3f} s")

            print(f"  Total adaptation time       : "
                f"{total_adapt:>8.3f} s")

            if config.get('setup_type') == 'drift':
                print(f"  Est. time if always adapted : "
                    f"{estimated_if_always:>8.3f} s")

                print(f"  Time saved by drift detect  : "
                    f"{prof['time_saved_by_drift_detection_s']:>8.3f} s")

            print(f"  Peak memory (RSS)           : "
                f"{prof['peak_memory_mb']:>8.1f} MB")

            print("=" * 72 + "\n")
 
    if platform.system() not in ("Linux", "Darwin"):
        tracemalloc.stop()
        
    return (
        baseline_outputs,
        baseline_targets,
        prof,                 
    )