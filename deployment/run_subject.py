import os
import sys
# Allow imports relative to the project root (adjust if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psutil
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
from deployment_maml import MAML
from deployment_component_factory import BPRegressor, DeeperBPRegressor, ReservoirReplayBuffer, Model
from Proto import Proto
from DriftDetectors import MMDDriftOnline, LSDDDriftOnline


def build_inner_optimizer(
    adapted_encoder,
    adapted_head,
    base_lr,
    mode,       
    opt_type,
    config
):
    r"""
    Build inner optimizer for meta-learning algorithms evaluation and deployment (not pretraining, learn2learn library with autograd is employed for pretraining).
    Behaviors:
    - inner_adapt='all'  : adapt backbone + regressor
    - inner_adapt='head' : freeze backbone, adapt only regressor
    
    Parameters
    ------------
        adapted_encoder (torch.nn.Module): 
            The model encoder to be adapted.
        adapted_head (torch.nn.Module): 
            The model prediction head to be adapted.
        base_lr (float): 
            The base learning rate for the inner optimizer.
        mode (str): 
            Inner adaptation mode, 'all' or 'head'.
        opt_type (str): 
            Inner optimizer type, either 'sgd' or 'adam'.
        config (dict): 
            Configuration dictionary containing inner adaptation parameters.
            
    Returns
    ------------
        inner_opt (torch.optim.Optimizer): 
            The constructed inner optimizer for meta-learning evaluation and deployment.    
    """ 

    # ----------------------------
    # 1) Make everything trainable
    # ----------------------------
    # BIOT has longtensor parameters for index (positional embedding), when setting requires_grad for them an error is raised
    if config['model_name'] == 'BIOT':
        for p in adapted_encoder.parameters():
            if p.dtype.is_floating_point or p.is_complex():
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for p in adapted_encoder.parameters():
            p.requires_grad = True
    
    for p in adapted_head.parameters():
        p.requires_grad = True

    # -------------------------------
    # 2) Apply mode-specific freezing
    # -------------------------------
    if mode == "head":
        # Freeze entire encoder
        for p in adapted_encoder.parameters():
            p.requires_grad = False

    elif mode == "all":
        # Nothing frozen
        pass

    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")

    # -------------------------------
    # 3) Collect trainable parameters
    # -------------------------------
    params = [
        p for p in list(adapted_encoder.parameters()) + list(adapted_head.parameters())
        if p.requires_grad
    ]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters selected for inner optimizer.")

    # ------------------
    # 4) Build optimizer
    # ------------------
    if opt_type == "sgd":
        inner_opt = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=float(config.get("sgd_momentum"))
        )
    else:
        inner_opt = torch.optim.Adam(params, lr=base_lr)

    return inner_opt

# ------------------------
#  MEMORY TRACKING HELPERS
# ------------------------

def _current_rss_mb() -> float:
    """
    Return the current Resident Set Size (RSS) of this process in MB.

    RSS = physical RAM currently mapped to the process, including shared
    library pages (PyTorch, libc, etc.).  Instantaneous — not a peak.

    Used only for large aggregate snapshots (model load, TTA loop baseline,
    per-step peak tracking) where changes are expected to be several MB.
    Per-section RSS deltas are not tracked because PyTorch's caching allocator
    reuses its internal pool without releasing OS pages, making those deltas
    structurally zero for tensor-only sections.
    """
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024.0 * 1024.0)


def _peak_memory_mb() -> float:
    """
    Return the true peak RSS since process start, in MB.

    On Linux (including Raspberry Pi): reads VmHWM from /proc/self/status —
    the kernel-maintained high-water mark, with no kB/bytes unit ambiguity.
    macOS: current RSS (no VmHWM equivalent available).
    Windows: peak Working Set via psutil.

    This is the process-wide peak including interpreter startup, imports,
    and model loading — not just the TTA loop. Use _current_rss_mb()
    snapshots to isolate specific phases.
    """
    if platform.system() == "Linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0   # kB → MB

    proc = psutil.Process(os.getpid())
    mi = proc.memory_info()
    if platform.system() == "Darwin":
        return mi.rss / (1024.0 * 1024.0)
    else:
        return getattr(mi, "peak_wset", mi.rss) / (1024.0 * 1024.0)


def _tm_current_kb() -> float:
    """
    Return the current Python-heap allocated memory in kB via tracemalloc.

    Captures numpy arrays, Python objects, optimizer state dicts, and any
    other CPython-heap allocation.  Does NOT capture PyTorch tensor memory
    (managed by PyTorch's C++ allocator outside the CPython heap).

    This is the complement to RSS: RSS is blind to PyTorch's pool reuse;
    tracemalloc is blind to C++ allocations but faithfully tracks everything
    on the Python heap, which is where optimizer state and numpy reference
    arrays live.

    tracemalloc.start() must be called before using this function.
    """
    current, _ = tracemalloc.get_traced_memory()
    return current / 1024.0


def _tm_peak_kb() -> float:
    """
    Return the peak Python-heap memory since tracemalloc.start(), in kB.
    The peak is monotonically increasing and is not reset between steps.
    Call this once at the end of the run to get the process-wide Python-heap
    high-water mark.
    """
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024.0


def _mono():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def run_subject(data_path, model_path, config_path, device=None, verbose=False):
    
    # Start tracemalloc
    # Tracks Python-heap allocations (numpy, optimizer state, detector internals).
    # Started before model load so tm_peak_run_kb covers the full runtime.
    tracemalloc.start()

    # Baseline RSS before any model work
    rss_before_model_load_mb = _current_rss_mb()
    
    # Initialize profiling data structure
    prof = {
        # per TTA step (appended in loop)
        'per_step': {
            # Latency (seconds)
            'feature_extraction_s':     [],   # enc(signals) latency
            'prediction_s':             [],   # pred(feats) latency
            'drift_detection_s':        [],   # drift_detector.predict() loop latency
            'adaptation_s':             [],   # entire do_adapt block latency
            'head_adapt_s':             [],   # prediction head adapt latency
            'drift_detector_reinit_s':  [],   # drift detector re-init latency
            'total_step_s':             [],   # wall time for the full TTA step
            'did_adapt':                [],   # bool
            
            # Tracemalloc per step (kB)
            # tm_current_kb: absolute Python-heap usage at the end of each step.
            # Plot over steps to detect monotonic growth (leak) vs plateau (healthy).
            # tm_head_adapt_kb: delta during head adaptation.
            #   Non-zero because optimizer state dict entries live on CPython heap.
            # tm_detector_reinit_kb: delta during detector reinit.
            #   Non-zero because reference_data = numpy.copy() is a CPython allocation.
            # All other sections (feature extraction, prediction, drift detection)
            # show ~0 even here because they only touch PyTorch tensors (C++ heap).
            'tm_current_kb':                [],
            'tm_head_adapt_kb':             [],
            'tm_detector_reinit_kb':        [],
            
            # Energy (J)
            't_step_start_mono':             [],  # monotonic wall time at the start of each step
            't_step_end_mono':               [],  # monotonic wall time at the end of each step
        },

        # Total run time (filled at end of run_subject)
        'cumulative_latency_s': 0.0,
        
        # Cumulative monotonic wall time at the start/end of the TTA loop
        't_run_start_mono': 0.0,
        't_run_end_mono': 0.0,

        # Aggregate memory (filled at end of run_subject)

        # RSS delta from before model construction to after load_state_dict.
        # Fixed RAM cost of keeping the model in memory.
        'model_memory_mb':                  0.0,

        # Absolute RSS snapshot just before the TTA loop begins.
        # Fixed cost that must fit in RAM regardless of the data stream.
        'rss_before_tta_mb':                0.0,

        # Highest RSS observed during the TTA loop (absolute, not incremental).
        'peak_tta_rss_mb':                  0.0,

        # peak_tta_rss_mb minus rss_before_tta_mb.
        # Memory the TTA algorithm itself requires on top of the loaded model.
        'peak_incremental_tta_mb':          0.0,

        # Process-wide RSS high-water mark (interpreter + imports + model + TTA).
        # Upper-bound RAM budget figure.
        'peak_process_rss_mb':              0.0,

        # Peak Python-heap usage over the entire run (kB).
        # Monotonically increasing high-water mark from tracemalloc.
        # Captures the worst-case CPython-heap pressure across all phases.
        'tm_peak_run_kb':                   0.0,
    }
    
    if device is None:
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
    prediction_head_pre = BPRegressor(input_dim=config['embed_dim'], output_dim=config['output_dim']) if not config['ecg'] else DeeperBPRegressor(input_dim=2 * config['embed_dim'], output_dim=config['output_dim'])
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

    # Snapshot RSS after model load
    rss_after_model_load_mb = _current_rss_mb()
    prof['model_memory_mb'] = rss_after_model_load_mb - rss_before_model_load_mb

    # To device
    enc, ph = pretrained_learner.encoder.to(device), pretrained_learner.prediction_head.to(device)

    # Feature replay buffer
    replay_buffer = ReservoirReplayBuffer(seed=config['seed'], max_size=config['replay_buffer_size'])
    
    # Prepare data structures for evaluation 
    baseline_outputs = []
    baseline_targets = []
    
    # Drift detector placeholders
    drift_detector = None
    detector_initialized = False
    
    # Important for logging
    step_idx = 0
    
    # Snapshot cumulative latency/RSS immediately before TTA loop 
    t_cumulative_start = time.perf_counter()
    rss_before_tta_mb = _current_rss_mb()
    prof['rss_before_tta_mb'] = rss_before_tta_mb
    peak_tta_rss_mb = rss_before_tta_mb   # updated every step
    
    # Record monotonic wall time at the start of the TTA loop
    prof['t_run_start_mono'] = _mono()
    
    # ---- PERSONALIZATION ----
    for block_idx in range(len(blocks_list)):
        
        # ONLINE TEST-TIME ADAPTATION EVALUATION 
        
        block = blocks_list[block_idx]
        signals = torch.from_numpy(block["signals"]).float()
        targets = torch.from_numpy(block["targets"]).float()
        
        # TTA start
        t_step_start = time.perf_counter()
        
        # Record monotonic wall time at the start of the TTA step
        prof['per_step']['t_step_start_mono'].append(_mono())
        
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
            
        if config['setup_type'] == 'fixed':
            do_adapt = True

        elif config['setup_type'] == 'drift':
            # NOTE: Before the detector is initialized, always adapt.
            if not detector_initialized:
                do_adapt = True
            else:
                # Drift detection start
                t_drift_start = time.perf_counter()
                
                # Use drift detector to decide
                for i in range(features.shape[0]):
                    detection_report = drift_detector.predict(features[i].cpu().numpy())
                    drift_flag = detection_report["data"]["is_drift"]
                    
                    if drift_detector.t >= config['detector_window_size']:
                        if drift_flag == 1:
                            do_adapt = True
                
                # Drift detection end
                t_drift_end = time.perf_counter()
                
                drift_latency = t_drift_end - t_drift_start
                
        else:
            raise ValueError('Inexistent adaptation type, allowed is fixed or drift!')
        
        # Adaptation latency
        # When we do not adapt these are 0
        adapt_latency = 0.0 
        head_adapt_latency = 0.0
        drift_detector_reinit_latency = 0.0
        tm_head_adapt_delta_kb = 0.0
        tm_detector_reinit_delta_kb = 0.0
        
        if do_adapt:
            # ---------- Adaptation (head only) ----------
            
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
            
            # Start prediction head adaptation profiling
            tm_before_head_adapt = _tm_current_kb()
            t_head_adapt_start = time.perf_counter()
            
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
            
            # End head adaptation profiling
            t_head_adapt_end = time.perf_counter()
            tm_after_head_adapt = _tm_current_kb()
            
            head_adapt_latency = t_head_adapt_end - t_head_adapt_start            
            tm_head_adapt_delta_kb = tm_after_head_adapt - tm_before_head_adapt
            
        # Update replay buffer 
        with torch.no_grad():
            replay_buffer.add(features.detach().cpu(), targets.detach().cpu())
                
        # Detector init/re-init when buffer has enough samples for the reference set
        if config['setup_type'] == 'drift':
            if len(replay_buffer) >= config['calibration_phase_size'] and (not detector_initialized or do_adapt):
                
                # Start drift reinit profiling
                tm_before_reinit = _tm_current_kb()
                drift_detector_reinit_start = time.perf_counter()
                
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

                if not detector_initialized:
                    detector_initialized = True
                    print(f"[Personalization] Detector initialized with {reference_data.shape[0]} samples ✓")
                else:
                    print(f"[Personalization] Detector re-initialized with {reference_data.shape[0]} samples ✓")  
        
                # End drift reinit profiling
                drift_detector_reinit_end = time.perf_counter()
                tm_after_reinit =_tm_current_kb()
                
                drift_detector_reinit_latency = drift_detector_reinit_end - drift_detector_reinit_start
                tm_detector_reinit_delta_kb = tm_after_reinit - tm_before_reinit

                # Adaptation ends
                # NOTE: we always enter both head adapt. and detector reinit.
                # whenb do_adapt true, thereby it is correct to end the adaptation counter here
                t_adapt_end = time.perf_counter()
                adapt_latency = t_adapt_end - t_adapt_start
                
        # Step end: timing, memory, and record monotonic wall time                   
        t_step_end = time.perf_counter()
        rss_step_end_mb = _current_rss_mb()
        
        step_latency = t_step_end - t_step_start
        peak_tta_rss_mb = max(peak_tta_rss_mb, rss_step_end_mb)
        
        prof['per_step']['t_step_end_mono'].append(_mono())
        
        # Record per-step profiling 
        prof['per_step']['prediction_s'].append(pred_latency)
        prof['per_step']['feature_extraction_s'].append(feat_latency)
        prof['per_step']['drift_detection_s'].append(drift_latency)
        prof['per_step']['adaptation_s'].append(adapt_latency)
        prof['per_step']['head_adapt_s'].append(head_adapt_latency)
        prof['per_step']['drift_detector_reinit_s'].append(drift_detector_reinit_latency)
        prof['per_step']['total_step_s'].append(step_latency)
        
        prof['per_step']['did_adapt'].append(do_adapt)
        
        prof['per_step']['tm_current_kb'].append(_tm_current_kb())
        prof['per_step']['tm_head_adapt_kb'].append(tm_head_adapt_delta_kb)
        prof['per_step']['tm_detector_reinit_kb'].append(tm_detector_reinit_delta_kb)
        
        # Increment step idx for the next block
        step_idx += 1
    
    # Aggregate memory results
    prof['cumulative_latency_s'] = time.perf_counter() - t_cumulative_start
    prof['peak_tta_rss_mb'] = peak_tta_rss_mb
    prof['peak_incremental_tta_mb'] = peak_tta_rss_mb - rss_before_tta_mb
    prof['peak_process_rss_mb'] = _peak_memory_mb()
    prof['tm_peak_run_kb'] = _tm_peak_kb()
 
    # Always stop tracemalloc — started unconditionally at the top
    tracemalloc.stop()
    
    # Record monotonic wall time at the end of the TTA loop
    prof['t_run_end_mono'] = _mono()
        
    return (
        baseline_outputs,
        baseline_targets,
        prof,                 
    )