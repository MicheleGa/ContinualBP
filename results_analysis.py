
import os
from pathlib import Path
import sys
folders_to_add = ['data', 'models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import json
import yaml
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import torch
from thop import profile
from models.Proto import Proto
from models.component_factory import BPRegressor, DeeperBPRegressor
from data.preprocessing_utils.data_visualization import plot_drift_aware_updates_per_subject, plot_latency_breakdown, plot_memory_breakdown

# ---------
# Constants
# ---------

_PER_STEP_LATENCY_KEYS = [
    'feature_extraction_s',     # enc(signals) latency
    'prediction_s',             # pred(feats) latency
    'drift_detection_s',        # drift_detector.predict() loop latency
    'adaptation_s',             # entire do_adapt block latency
    'head_adapt_s',             # prediction head adapt latency
    'drift_detector_reinit_s',  # drift detector re-init latency
    'total_step_s',             # wall time for the full TTA step
]

_AGGREGATE_LATENCY_KEYS = [
    'cumulative_latency_s',     # total wall time for the entire TTA loop
]

_AGGREGATE_MEMORY_KEYS = [
    'model_memory_mb',          # fixed RAM cost of keeping the model in memory
    'rss_before_tta_mb',        # fixed cost that must fit in RAM regardless of stream
    'peak_tta_rss_mb',          # highest RSS observed during the TTA loop
    'peak_incremental_tta_mb',  # memory the TTA algorithm itself requires on top of model
    'peak_process_rss_mb',      # process-wide RSS: upper-bound RAM budget
    'tm_peak_run_kb',           # peak Python-heap usage over the entire run (kB)
]

_PER_STEP_MEMORY_KEYS = [
    'tm_current_kb',            # absolute Python-heap usage at the end of each step
    'tm_head_adapt_kb',         # delta during head adaptation
    'tm_detector_reinit_kb',    # delta during detector reinit
]

_AGGREGATE_MEMORY_LABELS = {
    'model_memory_mb':          'Model footprint (load cost)',
    'rss_before_tta_mb':        'RSS before TTA loop (fixed cost)',
    'peak_tta_rss_mb':          'Peak RSS during TTA loop (absolute)',
    'peak_incremental_tta_mb':  'Peak incremental TTA memory',
    'peak_process_rss_mb':      'Process-wide peak RSS (incl. imports)',
    'tm_peak_run_kb':           'Peak Python-heap usage over the entire run',
}

_PER_STEP_MEMORY_LABELS = {
    'tm_current_kb':            'Absolute Python-heap usage at the end of each step',
    'tm_head_adapt_kb':         'Delta during head adaptation',
    'tm_detector_reinit_kb':    'Delta during detector reinit',
}

_PER_STEP_ANNOTATED_SAMPLES_NUMBER = 4

# -----------------------------------
# MACs/Communication Costs Estimation
# -----------------------------------

def estimate_communication_costs(setting, config_file_path):
    def fmt(value, decimals=1):
        return rf"${value:.{decimals}f}$"
    
    # Get encoder/head parameters
    with open(config_file_path, "r") as f:
        setup = yaml.safe_load(f)

    adapt_bs = 4
    number_of_updates = 72
    input_data_length_s = int(setup.get('input_seq_len_s'))
    input_data_freq = setup.get('fs')
    input_channels = 2 if setup.get('ecg') else 1
    feature_embed_dim = int(setup.get('embed_dim'))
    
    # ---- Resource Usage Estimation for the Model ----
    print("[Resource Usage Profile] Instantiating the encoder and the head ...")

    # Instantiate encoder
    encoder = Proto(setup.get('ecg'), input_data_freq, input_data_length_s, feature_embed_dim)
    x = torch.rand(adapt_bs, input_data_length_s * input_data_freq, input_channels) # N.B. batch size 1 for profiling
    _, encoder_params = profile(encoder, inputs=(x,))
    
    # Instantiate head
    head = BPRegressor(setup.get('embed_dim'), 3) if not setup.get('ecg') else DeeperBPRegressor(2 * setup.get('embed_dim'), 3)
    y = torch.rand((adapt_bs, setup.get('embed_dim') if not setup.get('ecg') else 2 * setup.get('embed_dim'))) # N.B. batch size 1 for profiling    
    _, head_params = profile(head, inputs=(y,))

    # One sample memory
    sample_memory = input_data_length_s * input_data_freq * input_channels
    
    batch_memory = sample_memory * adapt_bs
  
    memory_transmitted_for_all_updates = batch_memory * number_of_updates
    
    # FP 32 precision
    feature_extractor_fp_params = int(encoder_params)
    prediction_head_fp_params = int(head_params)
    
    # --- Calculation Logic ---
    # Total parameters in the model
    total_params = feature_extractor_fp_params + prediction_head_fp_params
    
    # 1 FP32 value = 4 bytes. 
    # Convert total bytes to kilobytes (kB) using 1 kB = 1,024 bytes
    model_memory_bytes = total_params * 4
    model_memory_kb = model_memory_bytes / (1024)
    
    # Input data values are typically integers or floats. Assuming standard 4-byte (FP32) float 
    # values for the transmitted sensor/input data:
    data_memory_bytes = memory_transmitted_for_all_updates * 4
    data_memory_kb = data_memory_bytes / (1024)
    
    if setting == "A":
        # Return the MB of the model (feat.ext. + head)
        return fmt(model_memory_kb, decimals=1)
    elif setting == 'B':  # Added missing colon
        # Return the MB of the model (feat.ext. + head) + memory transmitted for all updates 
        return fmt((model_memory_kb + data_memory_kb)/1024, decimals=1)
    else:
        raise ValueError("Incorrect setting was passed as input argument")
    
    
# ---------------------
# MACs/Bytes Estimation
# ---------------------

def bytes_from_params(n_params, precision_bits):
    return int(n_params * (precision_bits // 8))


def calculate_activation_memory(model, dummy_input):
    r"""
    Calculate total activation memory during forward pass.
    Source: https://huggingface.co/blog/train_memory
    """
    activation_sizes = []

    def forward_hook(model, input, output):
        """
        Hook to calculate activation size for each module.
        The .element_size() method returns the size in bytes of each element in the tensor.
        """
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            for tensor in output:
                if isinstance(tensor, torch.Tensor):
                    activation_sizes.append(tensor.numel() * tensor.element_size())
        
    # Register hooks for each submodule
    hooks = []
    for submodule in model.modules():
        hooks.append(submodule.register_forward_hook(forward_hook))

    # Perform a forward pass with a dummy input
    model.eval()  # No gradients needed for memory measurement
    with torch.no_grad():
        model(dummy_input)

    # Clean up hooks
    for hook in hooks:
        hook.remove()
        
    return sum(activation_sizes)


def resource_usage_profile_estimation(config_file_path):
        
    # Get configuration parameters
    with open(config_file_path, "r") as f:
        setup = yaml.safe_load(f)

    # Setup Configuration
    precision_bits = int(setup.get('precision_bits', 32)) # Deafult to float32 if not specified
    # Hardcoded because the YAML for pretraining is nt configured with personalization settings
    adapt_bs = 4
    test_bs = 4
    
    # NOTE: during an update the best model is selected helding out 0.25 of the personalization batch as validation, 
    # therefore the personliazaiton batch size and the replayed samples are personalization_bs * 0.75
    # -> overestimation with the full adapt_bs
    num_replay_samples_per_update = adapt_bs
    
    input_seq_len_s = int(setup.get('input_seq_len_s'))
    sampling_frequency = setup.get('fs')
    input_channels = 2 if setup.get('ecg') else 1
    feature_embed_dim = setup.get('embed_dim') if not setup.get('ecg') else 2 * setup.get('embed_dim')
    replay_buffer_size = 16
    
    steps_per_update = 10
    alpha_backward = 2 # fwd+bwd MACs as 2 times the fwd MACs
    
    # ---- Resource Usage Estimation for the Model ----
    print("[Resource Usage Profile] Instantiating the encoder and the head ...")

    # Instantiate encoder
    encoder = Proto(setup.get('ecg'), sampling_frequency, input_seq_len_s, setup.get('embed_dim'))
    x = torch.rand(adapt_bs, input_seq_len_s * sampling_frequency, input_channels) # N.B. batch size 1 for profiling
    
    encoder_forward_macs_sample, encoder_params = profile(encoder, inputs=(x,))
    encoder_forward_m_macs_sample = (encoder_forward_macs_sample) / 1e6
    encoder_params_mb = bytes_from_params(encoder_params, precision_bits=precision_bits) / (1024**2)
    
    encoder_act_bytes = calculate_activation_memory(encoder, x)
    encoder_act_bytes_mb = encoder_act_bytes / (1024**2)
        
    print(f'[Resource Usage Profile] Proto Encoder has:') 
    print(f'\t- {encoder_params} params ({encoder_params_mb:.2f} MB)')
    print(f'\t- {encoder_forward_m_macs_sample:.2f} M MACs per {adapt_bs}-sample batch size')
    print(f'\t- {encoder_act_bytes_mb:.2f} MB forward peak activation bytes')
    
    # Instantiate head
    head = BPRegressor(setup.get('embed_dim'), 3) if not setup.get('ecg') else DeeperBPRegressor(2 * setup.get('embed_dim'), 3)
    y = torch.rand((adapt_bs, setup.get('embed_dim') if not setup.get('ecg') else 2 * setup.get('embed_dim'))) # N.B. batch size 1 for profiling    
        
    head_forward_macs_sample, head_params = profile(head, inputs=(y,))
    head_forward_k_macs_sample = (head_forward_macs_sample) / 1e3
    head_params_kb = bytes_from_params(head_params, precision_bits=precision_bits) / 1024
    
    head_act_bytes = calculate_activation_memory(head, y)
    head_act_bytes_kb = head_act_bytes / 1024

    print(f'[Resource Usage Profile] Proto Head has:') 
    print(f'\t- {head_params} params ({head_params_kb:.2f} kB)')
    print(f'\t- {head_forward_k_macs_sample:.2f} k MACs per {adapt_bs}-sample batch size')
    print(f'\t- {head_act_bytes_kb:.2f} kB forward peak activation bytes')
    
    # ---- Resource Usage Estimation for the Feature Replay Algorithm ----
    
    # NOTE: On update, the head has also the feature replay buffer batch
    # NOTE: Assume that personalization, evaluation, and replayed feature batch sizes are equal
    # NOTE: Keep in mind to check the batch size used in thop, it should be 1
    # NOTE: An update is w/ frozen encoder and head-only updates
    # NOTE: Excluding the replay buffer update operations (e.g. reservoir operations for buffer update)
    
    # ---- MACs for adaptation ----    
    
    # The CL algorithm always predicts the training batch + validation batch to ensure a prediction for all samples (necessary in a real system)
    total_adapt_prediction_macs = (encoder_forward_macs_sample + head_forward_macs_sample) * adapt_bs
    
    # MACs for head when encoder is frozen and there are the replay buffer features
    macs_head_forward_per_update_step = head_forward_macs_sample * (adapt_bs + num_replay_samples_per_update)
    macs_head_forward_backward_per_update_step = alpha_backward * macs_head_forward_per_update_step
    
    # Multiply by steps to get the MACs related to the updated head when updating only the head
    macs_head_total =  steps_per_update * macs_head_forward_backward_per_update_step
    
    # Encoder MACs per update are simply the encoder batch forward by the number of steps
    macs_encoder_total = encoder_forward_macs_sample * adapt_bs
    
    # Total MACs per update with frozen encoder (input = one personalization batch)
    total_adapt_macs = macs_encoder_total + macs_head_total 
     
    # Optimizer operations, assuming Adam formula is the following:
    # Adam Optimizer - Operation Count Per Parameter
    # ================================================
    # Formula:
    #   m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
    #   v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
    #   m̂_t = m_t / (1 - β₁^t)
    #   v̂_t = v_t / (1 - β₂^t)
    #   θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
    #
    # Operation Count:
    # ┌─────────────────┬───────┬──────────────────────────────────────────┐
    # │ Operation Type  │ Count │ Where Used                               │
    # ├─────────────────┼───────┼──────────────────────────────────────────┤
    # │ Multiplication  │   5   │ β₁·m_{t-1}, (1-β₁)·g_t, β₂·v_{t-1},    │
    # │                 │       │ (1-β₂)·g_t², α·[...]                    │
    # │ Addition        │   3   │ m_t sum, v_t sum, √v̂_t + ε              │
    # │ Subtraction     │   1   │ θ_{t-1} - [...]                         │
    # │ Division        │   3   │ m_t/(1-β₁^t), v_t/(1-β₂^t), m̂_t/[...] │
    # │ Square          │   1   │ g_t²                                     │
    # │ Square Root     │   1   │ √v̂_t                                     │
    # ├─────────────────┼───────┼──────────────────────────────────────────┤
    # │ TOTAL           │  14   │                                          │
    # └─────────────────┴───────┴──────────────────────────────────────────┘
    #optimizer_ops = 14
    # For N updated parameters: 14N operations per optimizer step
    # -> these are FLOPs and not MACs
    #optimizer_ops_per_update = steps_per_update * head_params * optimizer_ops
    #
    # NOTE: also reservoir sampling requires ops that are neglected here
    
    # ---- MACs for testing ----
    # Encoder forward + head forward (input = one validation batch)
    # with a single step since this is inference and without replay buffer, buffer is only for training
    total_test_macs = (encoder_forward_macs_sample + head_forward_macs_sample) * test_bs
    
    
    # ---- CL algorithm occupation in memory as number of bytes ----
    model_params = bytes_from_params(encoder_params + head_params, precision_bits)
    
    # Input size
    # During adaptation the fact that we have also validation inference and also the prediction inferece on the training data (necessary in a real system) 
    # does not matter from a storage point of view as adaptation costs dominate the memory usage
    # -> we use adapt_bs instead of splitting into adapt and val sizes
    adapt_input_batch_bytes = bytes_from_params(adapt_bs * input_seq_len_s * sampling_frequency * input_channels, precision_bits)
    adapt_feature_batch_size = bytes_from_params((adapt_bs + num_replay_samples_per_update) * feature_embed_dim, precision_bits)
    
    test_input_batch_bytes = bytes_from_params(test_bs * input_seq_len_s * sampling_frequency * input_channels, precision_bits)
    test_feature_batch_size = bytes_from_params(test_bs * feature_embed_dim, precision_bits)
    
    # Feature replay memory occupation
    replay_buffer_total_bytes = bytes_from_params(replay_buffer_size * feature_embed_dim, precision_bits)
    
    # Gradients bytes
    adapt_gradients_bytes = bytes_from_params(head_params, precision_bits)
    
    # Optimizer states in bytes
    # -> we use Adam, that stores and update momentum and variance, hence two values for each update parameters
    optimizer_states_bytes = bytes_from_params(2 * head_params, precision_bits)
    
    # Activation memory during adaptation (defined as any intermediate output of the encoder/head layers)
    # -> total memory for activations (upper bound)
    head_forward_backward_activations_bytes = head_act_bytes * alpha_backward
    adapt_activation_forward_backward_bytes = encoder_act_bytes * adapt_bs # personalization + val batch size 
    adapt_activation_forward_backward_bytes += head_forward_backward_activations_bytes * (adapt_bs + num_replay_samples_per_update) # adapt batch size
    
    # Total algorithm occupation in memory during update
    # -> encoder is never updated, so it never does a backward pass
    total_adapt_memory = adapt_input_batch_bytes + adapt_feature_batch_size + model_params + adapt_activation_forward_backward_bytes + adapt_gradients_bytes + optimizer_states_bytes + replay_buffer_total_bytes
    
    # Activation memory during testing (defined as any intermediate output of the encoder/head layers)
    # -> total memory for activations (upper bound)
    test_activation_forward_bytes = (encoder_act_bytes + head_act_bytes) * test_bs
    
    # Total algorithm occupation in memory during inference 
    # replay buffer is in memory even if it is not used during inference, gradients/optimziers may even be deallocated
    total_test_memory = test_input_batch_bytes + test_feature_batch_size + model_params + test_activation_forward_bytes + replay_buffer_total_bytes  
    
    print(f"[Resource Usage Profile] Feature Replay Algorithm Resource Usage:")
    print(f"\t- Total model params (encoder + head): {model_params / (1024**2):.2f} MB")
    print(f"\t- Total adaptation MACs per update: {total_adapt_macs / 1e6:.2f} M MACs")
    print(f"\t- Total adaptation prediction MACs per update: {total_adapt_prediction_macs / 1e6:.2f} M MACs")
    print(f"\t- Total testing MACs per inference: {total_test_macs / 1e6:.2f} M MACs")
    print(f"\t- Sample memory: {bytes_from_params(input_seq_len_s * sampling_frequency * input_channels, precision_bits) / 1024:.2f} kB")
    print(f"\t- Total replay buffer memory: {replay_buffer_total_bytes / 1024:.2f} kB")
    print(f"\t- Total adaptation feature memory: {adapt_feature_batch_size / 1024:.2f} kB")
    print(f"\t- Total adaptation memory: {total_adapt_memory / (1024**2):.2f} MB")
    print(f"\t- Total adaptation forward/backward bytes: {adapt_activation_forward_backward_bytes / (1024**2):.2f} MB")
    print(f"\t- Total adaptation head forward/backward bytes: {(head_forward_backward_activations_bytes * (adapt_bs + num_replay_samples_per_update)) / 1024:.2f} kB")
    print(f"\t- Total adaptation gradient bytes: {adapt_gradients_bytes / 1024:.2f} kB")
    print(f"\t- Total adaptation optimizer state bytes: {optimizer_states_bytes / 1024:.2f} kB")
    print(f"\t- Total replay buffer memory: {replay_buffer_total_bytes / 1024:.2f} kB")
    print(f"\t- Total testing batch memory: {test_input_batch_bytes / 1024:.2f} kB")
    print(f"\t- Total testing activation memory: {test_activation_forward_bytes / (1024**2):.2f} MB")
    print(f"\t- Total testing encoder memory: {(encoder_act_bytes * test_bs) / (1024**2):.2f} MB")
    print(f"\t- Total testing feature memory: {test_feature_batch_size / 1024:.2f} kB")
    print(f"\t- Total testing prediction head activation memory: {(head_act_bytes * test_bs) / 1024:.2f} kB")
    print(f"\t- Total testing memory: {total_test_memory / (1024**2):.2f} MB")
    print(f"\t- Total forward MACs encoder batch: {(encoder_forward_macs_sample * test_bs) / 1e6:.2f} M MACs")
    print(f"\t- Total forward MACs head batch: {(head_forward_macs_sample * test_bs) / 1e3:.2f} k MACs")
    print(f"\t- Total forward/backward MACs head batch: {macs_head_total / 1e6:.2f} M MACs")


# -------------------
# Metrics Aggregation
# -------------------

def resource_profiling(baseline, baselines, deployment_device):
    """
    Collect and report profiling results for each detector, averaging
    quantities across random seeds.

    Returns
    -------
    dict[str, dict]
        {detector_name: profiling_results} where each results dict is the
        value returned by _run_profiling_report.
    """
    baseline_paths = baselines[baseline]

    # Each detector maps to an ordered dict of {seed_label: Path}.
    # Add or remove entries here as new detectors / seeds are introduced.
    profiling_paths = {
        "MMD": {
            "seed_42": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path"]),
            "seed_41": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path_seed_41"]),
            "seed_40": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path_seed_40"])
        },
        "Always-on": {
            "seed_42": Path(baseline_paths[f"{deployment_device}_always_on_path"]),
            "seed_41": Path(baseline_paths[f"{deployment_device}_always_on_path_seed_41"]),
            "seed_40": Path(baseline_paths[f"{deployment_device}_always_on_path_seed_40"])
        }
    }
    
    results = {}
    for detector_name, seed_paths in profiling_paths.items():
        results[detector_name] = _run_profiling_report(
            baseline, seed_paths, detector_name, deployment_device
        )

    return results


def resource_profiling_ppg_ecg(baseline, baselines, deployment_device):
    """
    Collect and report profiling results for each detector, averaging
    quantities across random seeds.

    Returns
    -------
    dict[str, dict]
        {detector_name: profiling_results} where each results dict is the
        value returned by _run_profiling_report.
    """
    baseline_paths = baselines[f'{baseline}_ppg_ecg']

    # Each detector maps to an ordered dict of {seed_label: Path}.
    # Add or remove entries here as new detectors / seeds are introduced.
    profiling_paths = {
        "MMD": {
            "seed_42": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path"]),
        },
        "Always-on": {
            "seed_42": Path(baseline_paths[f"{deployment_device}_always_on_path"]),
        }
    }

    results = {}
    for detector_name, seed_paths in profiling_paths.items():
        results[detector_name] = _run_profiling_report(
            baseline, seed_paths, detector_name, deployment_device
        )

    return results


def _collect_seed_data(baseline, profiling_path, detector_name):
    """
    Collect profiling data for a *single* seed path.

    Iterates over subject directories, reads JSON profiling reports, and
    accumulates per-subject aggregates.

    Returns
    -------
    dict or None
        None if ``profiling_path`` does not exist.
        Otherwise a dict with keys:
            per_step_latency   : {key: [mean_per_subject, ...]}
            aggregate_memory   : {key: [value_per_subject, ...]}
            per_step_memory    : {key: [peak_per_subject, ...]}
            frequency_savings  : [saving_per_subject, ...]
            updates_drift      : {subject_id: int}
            total_opportunities: {subject_id: int}
            clinical_metrics   : pd.DataFrame
    """
    if not profiling_path.exists():
        raise ValueError(f"[{detector_name}] Error: path {profiling_path} does not exist — skipping.")

    per_step_per_subject_means = {k: [] for k in _PER_STEP_LATENCY_KEYS}
    aggregate_latency_per_subject = {k: [] for k in _AGGREGATE_LATENCY_KEYS}
    aggregate_memory_per_subject = {k: [] for k in _AGGREGATE_MEMORY_KEYS}
    per_step_memory_peak_per_subject = {k: [] for k in _PER_STEP_MEMORY_KEYS}
    subject_frequency_savings = []
    subject_annotation_savings = []
    subject_updates_drift = {}
    subject_annotations_drift = {}
    subject_total_update_opportunities = {}
    subject_total_annotations = {}

    for subject_dir in profiling_path.iterdir():
        if not (subject_dir.is_dir() and subject_dir.name.startswith("subject_")):
            continue

        subject_id = subject_dir.name.split('_')[-1]

        profiling_json_path = subject_dir / "profiling" / f"{baseline}_profiling_report.json"
        with open(profiling_json_path, "r", encoding="utf-8") as f:
            prof = json.load(f)

        # ── Update frequency reduction ────────────────────────────
        did_adapt = prof['per_step']['did_adapt']
        total_possible = len(did_adapt)
        actual_updates = did_adapt.count(True)
        saved = total_possible - actual_updates
        
        total_annotations = total_possible * _PER_STEP_ANNOTATED_SAMPLES_NUMBER
        actual_annotations = actual_updates * _PER_STEP_ANNOTATED_SAMPLES_NUMBER
        saved_annotations = total_annotations - actual_annotations
        
        subject_frequency_savings.append(
            saved / total_possible if total_possible > 0 else 0.0
        )
        subject_annotation_savings.append(
            saved_annotations / total_annotations if total_annotations > 0 else 0.0
        )
        subject_updates_drift[subject_id] = actual_updates
        subject_annotations_drift[subject_id] = actual_annotations
        subject_total_update_opportunities[subject_id] = total_possible
        subject_total_annotations[subject_id] = total_annotations

        # ── Latency (mean per subject) ────────────────────────────
        for key in _PER_STEP_LATENCY_KEYS:
            values = prof['per_step'][key]
            if values:
                per_step_per_subject_means[key].append(np.mean(values))
        
        # ── Latency (total per subject) ────────────────────────────
        for key in _AGGREGATE_LATENCY_KEYS:
            aggregate_latency_per_subject[key].append(prof[key])
            
        # ── Scalar memory (one value per subject) ─────────────────
        for key in _AGGREGATE_MEMORY_KEYS:
            aggregate_memory_per_subject[key].append(prof[key])

        # ── Per-step memory (peak per subject) ───────────────────
        # We take max() across steps: worst-case allocation spike
        # determines whether the device runs out of RAM.
        for key in _PER_STEP_MEMORY_KEYS:
            values = prof['per_step'].get(key, [])
            if values:
                per_step_memory_peak_per_subject[key].append(max(values))

    # ── Aggregate clinical metrics CSV (one per path) ─────────────
    clinical_metrics_csv = pd.read_csv(
        os.path.join(
            profiling_path,
            "aggregate_metrics",
            f"aggregate_{baseline}_metrics",
            "evaluation_metrics.csv",
        )
    )

    return {
        "per_step_latency":    per_step_per_subject_means,
        "aggregate_latency":   aggregate_latency_per_subject,
        "aggregate_memory":    aggregate_memory_per_subject,
        "per_step_memory":     per_step_memory_peak_per_subject,
        "frequency_savings":   subject_frequency_savings,
        "annotation_savings":  subject_annotation_savings,
        "updates_drift":       subject_updates_drift,
        "annotations_drift":   subject_annotations_drift,
        "total_opportunities": subject_total_update_opportunities,
        "total_annnotations":  subject_total_annotations,
        "clinical_metrics":    clinical_metrics_csv,
    }


def _average_seed_results(seed_data_list):
    """
    Average profiling quantities across seeds.

    Averaging strategy
    ------------------
    For each numeric container (latency / memory) the function first reduces
    each seed to a single scalar (the mean over subjects for that seed).
    The returned arrays therefore have length == n_seeds, so that downstream
    ``np.mean`` / ``np.std`` reflect *cross-seed* variability — which is the
    quantity of interest when reporting seed-averaged results.

    Parameters
    ----------
    seed_data_list : list[dict]
        One element per valid seed, as returned by ``_collect_seed_data``.

    Returns
    -------
    dict with keys:
        latency_seed_means    : {key: np.ndarray of shape (n_seeds,)}
        agg_mem_seed_means    : {key: np.ndarray of shape (n_seeds,)}
        step_mem_seed_means   : {key: np.ndarray of shape (n_seeds,)}
        freq_seed_means       : np.ndarray of shape (n_seeds,)
        avg_updates_drift     : {subject_id: float}   seed-averaged update counts
        avg_total_opportunities: {subject_id: int}    (identical across seeds)
        avg_clinical_metrics  : pd.DataFrame          column-wise seed average
        n_seeds               : int
    """
    def _pool_subjects(data_list, container_key, metric_keys):
        """Concatenate per-subject values across all seeds into a single array."""
        out = {k: [] for k in metric_keys}
        for data in data_list:
            for key in metric_keys:
                vals = data[container_key][key]
                if vals:
                    out[key].extend(vals)          # <-- extend, not append(mean)

        return {k: np.array(v) for k, v in out.items()}

    latency_pooled = _pool_subjects(seed_data_list, "per_step_latency", _PER_STEP_LATENCY_KEYS)
    agg_latency_pooled = _pool_subjects(seed_data_list, "aggregate_latency", _AGGREGATE_LATENCY_KEYS)
    agg_mem_pooled = _pool_subjects(seed_data_list, "aggregate_memory",  _AGGREGATE_MEMORY_KEYS)
    step_mem_pooled = _pool_subjects(seed_data_list, "per_step_memory",   _PER_STEP_MEMORY_KEYS)

    # ── Frequency savings: pool raw per-subject values across seeds ──────────
    freq_pooled = np.array([
        v for d in seed_data_list for v in d["frequency_savings"]
    ])
    
    # ── Annotation savings: pool raw per-subject values across seeds ──────────
    ann_pooled = np.array([
        v for d in seed_data_list for v in d["annotation_savings"]
    ])
    
    # ── Per-subject update counts: average across seeds (subject-keyed) ──────
    # NOTE: kept as seed-average intentionally — these are counts tied to a
    # specific subject identity, not a performance metric to pool over rows.
    all_subject_ids = sorted(
        set().union(*[d["updates_drift"].keys() for d in seed_data_list])
    )
    avg_updates_drift = {
        sid: float(np.mean([d["updates_drift"].get(sid, np.nan) for d in seed_data_list]))
        for sid in all_subject_ids
    }
    
    # ── Per-subject annotation counts: average across seeds (subject-keyed) ──────
    all_subject_ids = sorted(
            set().union(*[d["annotations_drift"].keys() for d in seed_data_list])
    )
    avg_annotations_drift = {
        sid: float(np.mean([d["annotations_drift"].get(sid, np.nan) for d in seed_data_list]))
        for sid in all_subject_ids
    }
       
    # total_opportunities is determined by the data stream, not the seed
    avg_total_opportunities = seed_data_list[0]["total_opportunities"]
    
    # total_annotations is determined by the data stream, not the seed
    avg_total_annotations = seed_data_list[0]["total_annnotations"]

    # ── Clinical metrics: concatenate all subjects across seeds ──────────────
    pooled_clinical = pd.concat(
        [d["clinical_metrics"] for d in seed_data_list],
        ignore_index=True,
    )

    return {
        "latency_pooled":          latency_pooled,
        "agg_latency_pooled":      agg_latency_pooled,
        "agg_mem_pooled":          agg_mem_pooled,
        "step_mem_pooled":         step_mem_pooled,
        "freq_pooled":             freq_pooled,
        "ann_pooled":              ann_pooled,
        "avg_updates_drift":       avg_updates_drift,
        "avg_annotations_drift":   avg_annotations_drift,
        "avg_total_opportunities": avg_total_opportunities,
        "avg_total_annotations":   avg_total_annotations,
        "pooled_clinical_metrics": pooled_clinical,
        "n_seeds":                 len(seed_data_list),
    }


def _run_profiling_report(baseline, seed_paths, detector_name, deployment_device):
    """
    Run the full profiling report for one detector averaged across all seeds.

    Parameters
    ----------
    baseline : str
    seed_paths : dict[str, Path]
        {seed_label: profiling_path} — one entry per random seed.
    detector_name : str
    deployment_device : str

    Returns
    -------
    dict
        Averaged profiling results:
            clinical_metrics        : pd.DataFrame
            frequency_saving_mean   : float   (%)
            frequency_saving_std    : float   (%)
            latency_ms              : {key: (mean_ms, std_ms)}
            aggregate_latency_ms     : {key: (mean_s, std_s)}
            aggregate_memory_mb     : {key: (mean_mb, std_mb)}
            step_memory_kb          : {key: (mean_kb, std_kb)}
            avg_updates_drift       : {subject_id: float}
            avg_total_opportunities : {subject_id: int}
            n_seeds                 : int
        Returns an empty dict if no valid seed paths are found.
    """
    print(f"\n{'='*60}")
    print(f"  Profiling Report  —  Detector: {detector_name}, Device: {deployment_device}")
    print(f"  Seeds ({len(seed_paths)}): {list(seed_paths.keys())}")
    print(f"{'='*60}")

    # ── Collect data per seed ─────────────────────────────────────
    seed_data_list = []
    for seed_label, path in seed_paths.items():
        data = _collect_seed_data(baseline, path, detector_name)
        if data is not None:
            seed_data_list.append(data)

    if not seed_data_list:
        print(f"[{detector_name}] No valid seed data found. Skipping.")
        return {}

    # ── Average across seeds ──────────────────────────────────────
    avg = _average_seed_results(seed_data_list)
    n_seeds = avg["n_seeds"]
    # Suffix updated: pooled population is the unit, not seeds
    suffix  = f"(mean ± std over subjects, {n_seeds} seed(s) pooled)"

    # ── Clinical metrics ──────────────────────────────────────────
    # Pooled DataFrame has n_seeds * n_subjects rows — summarise column-wise
    pooled_clinical = avg["pooled_clinical_metrics"]
    
    # Select numeric columns along with the 'Type' column for grouping
    numeric_cols = pooled_clinical.select_dtypes(include=[np.number]).columns
    grouped_clinical = pooled_clinical[["Type"] + list(numeric_cols)]

    # Group by SBP/DBP and compute mean and std
    clinical_summary = grouped_clinical.groupby("Type").agg(["mean", "std"])

    print(f"\n[{detector_name}] Deployment ~ Clinical Metrics {suffix}:")
    print(clinical_summary)
    
    # ── Update frequency reduction ────────────────────────────────
    freq_arr  = avg["freq_pooled"] * 100          # pooled over all subjects
    mean_freq = float(freq_arr.mean())
    std_freq  = float(freq_arr.std())
    print(
        f"\n[{detector_name}] Deployment ~ Update Frequency Reduction {suffix}:"
        f" {mean_freq:.1f}% ± {std_freq:.1f}%"
    )
    
    # ── Annotation requirement reduction ────────────────────────────────
    ann_arr = avg["ann_pooled"] * 100
    mean_ann = float(ann_arr.mean())
    std_ann  = float(ann_arr.std())
    print(
        f"\n[{detector_name}] Deployment ~ Annotation Frequency Reduction {suffix}:"
        f" {mean_ann:.1f}% ± {std_ann:.1f}%"
    )
    
    # ── Required annotated samples (mean ± std OVER SUBJECTS) ─────
    # avg["avg_annotations_drift"] is {subject_id: seed-averaged annotation
    # count}. Seeds were already averaged per-subject inside
    # _average_seed_results, which is the correct first stage (seeds are
    # repeated noisy measurements of the SAME subject, not independent
    # units). Here we take the second stage: mean/std across the 88
    # independent subjects. This gives a whole-number "samples" unit,
    # not a fraction/percentage.
    subject_ids_sorted = sorted(avg["avg_annotations_drift"].keys())
    annotations_per_subject = np.array(
        [avg["avg_annotations_drift"][sid] for sid in subject_ids_sorted]
    )
    n_subjects = annotations_per_subject.size
 
    mean_annotations = float(annotations_per_subject.mean())
    std_annotations = float(annotations_per_subject.std())
 
    print(
        f"\n[{detector_name}] Deployment ~ Required Annotated Samples "
        f"(mean ± std over {n_subjects} subjects, {n_seeds} seed(s) "
        f"averaged per subject): {mean_annotations:.1f} ± {std_annotations:.1f} samples"
    )

    # ── Per-step latency ──────────────────────────────────────────
    print(f"\n[{detector_name}] Per-step Latency Summary {suffix}:")
    latency_stats = {}
    for key in _PER_STEP_LATENCY_KEYS:
        arr = avg["latency_pooled"][key]          # pooled over all subjects
        m, s = float(arr.mean()) * 1e3, float(arr.std()) * 1e3
        latency_stats[key] = (m, s)
        print(f"  {key:45s}  {m:7.2f} ± {s:6.2f} ms")

    # ── Aggregate latency ──────────────────────────────────────────
    print(f"\n[{detector_name}] Latency Summary — Aggregate (seconds for processing all subjects, over {n_seeds} seed(s) pooled):")
    agg_latency_stats = {}
    for key in _AGGREGATE_LATENCY_KEYS:
        # arr contains the total latency for processing one subject
        # more precisely it contains the latency collected for each seed for a subject, 
        # therby it is num subject by num seeds elements
        arr = np.array(avg["agg_latency_pooled"][key])
        
        # Reshape array to (3 seeds, 88 subjects) and sum per seed (in seconds)
        totals_per_seed = arr.reshape(n_seeds, len(arr) // n_seeds).sum(axis=1)
        
        # Seconds
        m_sec, s_sec = float(totals_per_seed.mean()), float(totals_per_seed.std())
        
        # Minutes
        m_min, s_min = m_sec / 60.0, s_sec / 60.0
        
        agg_latency_stats[key] = {
            "sec": (m_sec, s_sec),
            "min": (m_min, s_min)
        }
        
        print(f"  {key:45s}  {m_sec:7.2f} ± {s_sec:5.2f} s  ({m_min:5.2f} ± {s_min:4.2f} min) / run")
        
    # ── Aggregate memory ──────────────────────────────────────────
    print(f"\n[{detector_name}] Memory Summary — Aggregate {suffix}:")
    agg_mem_stats = {}
    for key in _AGGREGATE_MEMORY_KEYS:
        arr = avg["agg_mem_pooled"][key]          # pooled over all subjects
        if key == 'tm_peak_run_kb':
            arr /= 1024.0
        m, s = float(arr.mean()), float(arr.std())
        agg_mem_stats[key] = (m, s)
        print(f"  {_AGGREGATE_MEMORY_LABELS[key]:50s}  {m:7.2f} ± {s:5.2f} MB")

    # ── Per-step memory ───────────────────────────────────────────
    print(f"\n[{detector_name}] Memory Summary — Per-step MAX delta {suffix}:")
    print("    NOTE: each subject contributes its worst-case step delta;")
    print("    small values reflect allocator caching, not true zero cost.\n")
    step_mem_stats = {}
    for key in _PER_STEP_MEMORY_KEYS:
        arr = avg["step_mem_pooled"][key]         # pooled over all subjects
        if arr.size > 0:
            m, s = float(arr.mean()), float(arr.std())
            step_mem_stats[key] = (m, s)
            print(f"  {_PER_STEP_MEMORY_LABELS[key]:45s}  {m:7.2f} ± {s:5.2f} kB")

    # ── Plots ─────────────────────────────────────────────────────
    #plot_drift_aware_updates_per_subject(
    #    subject_updates_drift=avg["avg_updates_drift"],
    #    subject_total_update_opportunities=avg["avg_total_opportunities"],
    #    mean_frequency_saving=mean_freq,
    #    std_frequency_saving=std_freq,
    #    deployment_device=deployment_device,
    #    detector_name=detector_name,
    #    n_seeds=n_seeds,
    #)
    
    # ── Return results ────────────────────────────────────────────
    return {
        "clinical_metrics":             clinical_summary,        # column-wise summary
        "frequency_saving_mean":        mean_freq,
        "frequency_saving_std":         std_freq,
        "latency_ms":                   latency_stats,
        "aggregate_latency_ms":         agg_latency_stats,
        "aggregate_memory_mb":          agg_mem_stats,
        "update_freq_reduction":        (mean_freq, std_freq),
        "annotation_samples_reduction": (mean_annotations, std_annotations),
        "n_seeds":                 n_seeds,
    }
    
    
def aggregate_seed_dataframes(dataframes, bhs_columns=None):
    """
    Aggregate metrics across multiple seeds.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        List of dataframes having identical structure.
    bhs_columns : list[str]
        Columns containing BHS grades (A/B/C/D).
        These are aggregated using the mode.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe.
    """

    if bhs_columns is None:
        bhs_columns = []

    agg_df = dataframes[0].copy()

    for col in agg_df.columns:

        # -----------------------
        # BHS columns -> use mode
        # -----------------------
        if col in bhs_columns:

            modes = []

            for idx in range(len(agg_df)):
                values = [df.loc[idx, col] for df in dataframes]

                # mode across seeds
                mode_value = Counter(values).most_common(1)[0][0]
                modes.append(mode_value)

            agg_df[col] = modes

        # ---------------------------
        # Numeric columns -> use mean
        # ---------------------------
        else:

            try:
                stacked = np.stack([df[col].astype(float).values for df in dataframes])
                agg_df[f"{col}_mean"] = stacked.mean(axis=0)
                agg_df[f"{col}_std"] = stacked.std(axis=0)

            except:
                # Non numeric and not BHS (e.g. "Type") -> keep first
                pass
    
    return agg_df


def analyze_gradual_vs_mixed_vs_abrupt(baselines):
    
    aggregated_results = {}

    for baseline, baseline_paths in baselines.items():

        aggregated_results[baseline] = {}

        # ---------------------------------
        # Clinical Metrics (ME / STD / BHS)
        # ---------------------------------

        for shift_name, path_key in {
            "gradual": "gradual_shifts_path",
            "mixed": "mixed_shifts_path",
            "abrupt": "abrupt_shifts_path"
        }.items():

            dfs = [
                pd.read_csv(os.path.join(
                        baseline_paths[path_key],
                        "aggregate_metrics",
                        f"aggregate_{baseline}_metrics",
                        "evaluation_metrics.csv"
                    ),
                    usecols=["Type", "ME", "STD", "BHS_Grade"]            
                ),

                pd.read_csv(os.path.join(
                        baseline_paths[f"{path_key}_seed_41"],
                        "aggregate_metrics",
                        f"aggregate_{baseline}_metrics",
                        "evaluation_metrics.csv"
                    ),
                    usecols=["Type", "ME", "STD", "BHS_Grade"]  
                ),

                pd.read_csv(os.path.join(
                        baseline_paths[f"{path_key}_seed_40"],
                        "aggregate_metrics",
                        f"aggregate_{baseline}_metrics",
                        "evaluation_metrics.csv"
                    ),
                    usecols=["Type", "ME", "STD", "BHS_Grade"]  
                )
            ]

            aggregated_results[baseline][f"{shift_name}_clinical"] = (
                aggregate_seed_dataframes(
                    dfs,
                    bhs_columns=["BHS_Grade"]   
                )
            )

        # -------------------------------------
        # Continual Learning Metrics (AE / BWT)
        # -------------------------------------

        for shift_name, path_key in {
            "gradual": "gradual_shifts_path",
            "mixed": "mixed_shifts_path",
            "abrupt": "abrupt_shifts_path"
        }.items():

            # ---------------------------
            # SBP
            # ---------------------------
            sbp_dfs = [
                pd.read_csv(os.path.join(
                        baseline_paths[path_key],
                        "aggregate_metrics",
                        "sbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                ),

                pd.read_csv(os.path.join(
                        baseline_paths[f"{path_key}_seed_41"],
                        "aggregate_metrics",
                        "sbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                ),

                pd.read_csv(os.path.join(
                        baseline_paths[f"{path_key}_seed_40"],
                        "aggregate_metrics",
                        "sbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                )
            ]

            # ---------------------------
            # DBP
            # ---------------------------
            dbp_dfs = [
                pd.read_csv(os.path.join(
                        baseline_paths[path_key],
                        "aggregate_metrics",
                        "dbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                ),

                pd.read_csv(os.path.join(
                        baseline_paths[f"{path_key}_seed_41"],
                        "aggregate_metrics",
                        "dbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                ),

                pd.read_csv(os.path.join(
                        baseline_paths[f"{path_key}_seed_40"],
                        "aggregate_metrics",
                        "dbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                )
            ]

            aggregated_results[baseline][f"{shift_name}_sbp_cl"] = (
                aggregate_seed_dataframes(sbp_dfs)
            )

            aggregated_results[baseline][f"{shift_name}_dbp_cl"] = (
                aggregate_seed_dataframes(dbp_dfs)
            )

    
    baseline_display_names = {
        "running_mean": "cumulative-mean",
        "no_adapt": "no-adapt",
        "first_batch_finetune": "first-batch",
        "online": "online",
        "online_from_scratch": "online*",
        "feature_replay": "feat.replay",
        "lwf": "LwF",
        "ewc": "EWC",
        "agem": "AGEM"
    }

    set_mapping = {
        "1": "gradual",
        "2": "mixed",
        "3": "abrupt"
    }
    
    latex_table = generate_gradual_vs_mixed_vs_abrupt_overleaf_table(
        aggregated_results,
        baseline_display_names,
        set_mapping
    )
    
    print()
    print(latex_table)
    print()


def analyze_gradual_vs_mixed_vs_abrupt_ppg_ecg(baselines):
    
    aggregated_results = {}

    for baseline, baseline_paths in baselines.items():

        aggregated_results[baseline] = {}

        # ---------------------------------
        # Clinical Metrics (ME / STD / BHS)
        # ---------------------------------

        for shift_name, path_key in {
            "gradual": "gradual_shifts_path",
            "mixed": "mixed_shifts_path",
            "abrupt": "abrupt_shifts_path"
        }.items():
            
            if baseline == 'feature_replay_ppg' or baseline == 'feature_replay_ppg_ecg':
                # Note that baseline_paths[path_key] will differ for PPG or PPG/ECG cases 
                dfs = [
                        pd.read_csv(os.path.join(
                                baseline_paths[path_key],
                                "aggregate_metrics",
                                "aggregate_feature_replay_metrics",
                                "evaluation_metrics.csv"
                            ),
                            usecols=["Type", "ME", "STD", "BHS_Grade"]            
                        ),
                    ]
            else:    
                dfs = [
                    pd.read_csv(os.path.join(
                            baseline_paths[path_key],
                            "aggregate_metrics",
                            f"aggregate_{baseline}_metrics",
                            "evaluation_metrics.csv"
                        ),
                        usecols=["Type", "ME", "STD", "BHS_Grade"]            
                    ),
                ]

            # Use PPG/ECG suffix to distinguish results
            aggregated_results[baseline][f"{shift_name}_clinical"] = (
                aggregate_seed_dataframes(
                    dfs,
                    bhs_columns=["BHS_Grade"]   
                )
            )

        # -------------------------------------
        # Continual Learning Metrics (AE / BWT)
        # -------------------------------------

        for shift_name, path_key in {
            "gradual": "gradual_shifts_path",
            "mixed": "mixed_shifts_path",
            "abrupt": "abrupt_shifts_path"
        }.items():

            # ---------------------------
            # SBP
            # ---------------------------
            sbp_dfs = [
                pd.read_csv(os.path.join(
                        baseline_paths[path_key],
                        "aggregate_metrics",
                        "sbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                ),
            ]

            # ---------------------------
            # DBP
            # ---------------------------
            dbp_dfs = [
                pd.read_csv(os.path.join(
                        baseline_paths[path_key],
                        "aggregate_metrics",
                        "dbp_aggregate_baseline_metrics.csv"
                    ),
                    usecols=["AE_mean", "BWT_mean"]  
                ),
            ]

            aggregated_results[baseline][f"{shift_name}_sbp_cl"] = (
                aggregate_seed_dataframes(sbp_dfs)
            )

            aggregated_results[baseline][f"{shift_name}_dbp_cl"] = (
                aggregate_seed_dataframes(dbp_dfs)
            )

    
    baseline_display_names = {
        "running_mean": "cumulative-mean",
        "feature_replay_ppg": "feat.replay (PPG)",
        "feature_replay_ppg_ecg": "feat.replay (PPG+ECG)",
    }

    set_mapping = {
        "1": "gradual",
        "2": "mixed",
        "3": "abrupt"
    }
    
    latex_table = generate_gradual_vs_mixed_vs_abrupt_ppg_ecg_overleaf_table(
        aggregated_results,
        baseline_display_names,
        set_mapping
    )
    
    print()
    print(latex_table)
    print()
    

def analyze_drift_detection_methods(baselines):
    
    baseline = 'feature_replay'
    path_keys = [
        "always_on_path", 
        "drift_aware_mmd_path",
        "random_path",
        "drift_aware_lsdd_path", 
    ]
    
    baseline_paths = baselines[baseline]
    aggregated_results = {}

    aggregated_results[baseline] = {}

    # ---------------------------------
    # Clinical Metrics (ME / STD / BHS)
    # ---------------------------------           
    for path_key in path_keys:
        dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]            
            ),
            
            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_41"],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_40"],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]  
            )
        ]

        aggregated_results[baseline][f"{path_key}_clinical"] = (
            aggregate_seed_dataframes(
                dfs,
                bhs_columns=["BHS_Grade"]   # adapt if column name differs
            )
        )

    # -------------------------------------
    # Continual Learning Metrics (AE / BWT)
    # -------------------------------------
    for path_key in path_keys:
        # ---------------------------
        # SBP
        # ---------------------------
        sbp_dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_41"],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_40"],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            )
        ]

        # ---------------------------
        # DBP
        # ---------------------------
        dbp_dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_41"],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_40"],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            )
        ]

        aggregated_results[baseline][f"{path_key}_sbp_cl"] = (
            aggregate_seed_dataframes(sbp_dfs)
        )

        aggregated_results[baseline][f"{path_key}_dbp_cl"] = (
            aggregate_seed_dataframes(dbp_dfs)
        )
    
    # -----------------
    # Number of updates 
    # -----------------
    for path_key in path_keys:

        # collect all seed-specific paths
        seed_paths = [
            baseline_paths[path_key],
            baseline_paths[f"{path_key}_seed_41"],
            baseline_paths[f"{path_key}_seed_40"],
        ]

        experiment_dir = Path(seed_paths[0])

        subject_results = {}

        # iterate subjects
        for item in experiment_dir.iterdir():

            if item.is_dir() and item.name.startswith("subject_"):

                subject_folder = item.name

                skipped_per_seed = []
                performed_per_seed = []
                total_per_seed = []

                # loop over seeds
                for seed_path in seed_paths:

                    csv_path = os.path.join(
                        seed_path,
                        subject_folder,
                        baseline,
                        "param_update_log.csv"
                    )

                    df = pd.read_csv(
                        csv_path,
                        usecols=["n_updated_params"]
                    )
                    #print(df)

                    n_total = len(df)
                    n_skipped = (df["n_updated_params"] == 0).sum()
                    n_performed = (df["n_updated_params"] > 0).sum()

                    #print(n_total, n_skipped, n_performed)
                    
                    total_per_seed.append(n_total)
                    skipped_per_seed.append(n_skipped)
                    performed_per_seed.append(n_performed)

                # averages over seeds
                avg_total = np.mean(total_per_seed)
                avg_skipped = np.mean(skipped_per_seed)
                avg_performed = np.mean(performed_per_seed)

                #print(avg_total, avg_skipped, avg_performed)
                
                # fraction of saved/skipped updates
                skipped_fraction = avg_skipped / avg_total if avg_total > 0 else 0.0

                subject_results[subject_folder] = {
                    "avg_total_updates": avg_total,
                    "avg_skipped_updates": avg_skipped,
                    "avg_performed_updates": avg_performed,
                    "skipped_fraction": skipped_fraction,
                }

                #print(
                #    f"{path_key} | {subject_folder} | "
                #    f"Skipped fraction: {skipped_fraction:.3f}"
                #)


        # aggregate across subjects
        all_subject_fractions = [
            v["skipped_fraction"]
            for v in subject_results.values()
        ]

        aggregated_results[baseline][f"{path_key}_updates"] = {
            "per_subject": subject_results,
            "mean_skipped_fraction": np.mean(all_subject_fractions),
            "std_skipped_fraction": np.std(all_subject_fractions),
        }
        
    set_mapping = {
        "Always": "always_on_path",
        "MMD": "drift_aware_mmd_path",
        "Random": "random_path",
        "LSDD": "drift_aware_lsdd_path",
    }
    
    latex_table = generate_drift_detection_methods_overleaf_table(
        aggregated_results,
        set_mapping
    )
    
    print()
    print(latex_table)
    print()
    
    
def analyze_mmd_embeddings_and_buffer_sizes(baselines):
    baseline = 'feature_replay'
    path_keys = [ 
        "embed_dim_128_buffer_size_64_path",
        "embed_dim_128_buffer_size_32_path",
        "embed_dim_128_buffer_size_16_path",
        "embed_dim_32_buffer_size_64_path",
        "embed_dim_32_buffer_size_32_path",
        "embed_dim_32_buffer_size_16_path",
        "embed_dim_16_buffer_size_64_path",
        "embed_dim_16_buffer_size_32_path",
        "embed_dim_16_buffer_size_16_path"
    ]
    
    baseline_paths = baselines[baseline]
    aggregated_results = {}

    aggregated_results[baseline] = {}

    # ---------------------------------
    # Clinical Metrics (ME / STD / BHS)
    # ---------------------------------           
    for path_key in path_keys:
        dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]            
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_41"],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_40"],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]  
            )
        ]

        aggregated_results[baseline][f"{path_key}_clinical"] = (
            aggregate_seed_dataframes(
                dfs,
                bhs_columns=["BHS_Grade"]   # adapt if column name differs
            )
        )

    # -------------------------------------
    # Continual Learning Metrics (AE / BWT)
    # -------------------------------------
    for path_key in path_keys:
        sbp_dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_41"],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_40"],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            )
        ]

        # ---------------------------
        # DBP
        # ---------------------------
        dbp_dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_41"],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),

            pd.read_csv(os.path.join(
                    baseline_paths[f"{path_key}_seed_40"],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            )
        ]

        aggregated_results[baseline][f"{path_key}_sbp_cl"] = (
            aggregate_seed_dataframes(sbp_dfs)
        )

        aggregated_results[baseline][f"{path_key}_dbp_cl"] = (
            aggregate_seed_dataframes(dbp_dfs)
        )
    
    # -----------------
    # Number of updates 
    # -----------------
    for path_key in path_keys:

        # collect all seed-specific paths
        seed_paths = [
            baseline_paths[path_key],
            baseline_paths[f"{path_key}_seed_41"],
            baseline_paths[f"{path_key}_seed_40"],
        ]

        experiment_dir = Path(seed_paths[0])

        subject_results = {}

        # iterate subjects
        for item in experiment_dir.iterdir():

            if item.is_dir() and item.name.startswith("subject_"):

                subject_folder = item.name

                skipped_per_seed = []
                performed_per_seed = []
                total_per_seed = []

                # loop over seeds
                for seed_path in seed_paths:

                    csv_path = os.path.join(
                        seed_path,
                        subject_folder,
                        baseline,
                        "param_update_log.csv"
                    )

                    df = pd.read_csv(
                        csv_path,
                        usecols=["n_updated_params"]
                    )
                    #print(df)

                    n_total = len(df)
                    n_skipped = (df["n_updated_params"] == 0).sum()
                    n_performed = (df["n_updated_params"] > 0).sum()

                    #print(n_total, n_skipped, n_performed)
                    
                    total_per_seed.append(n_total)
                    skipped_per_seed.append(n_skipped)
                    performed_per_seed.append(n_performed)

                # averages over seeds
                avg_total = np.mean(total_per_seed)
                avg_skipped = np.mean(skipped_per_seed)
                avg_performed = np.mean(performed_per_seed)

                #print(avg_total, avg_skipped, avg_performed)
                
                # fraction of saved/skipped updates
                skipped_fraction = avg_skipped / avg_total if avg_total > 0 else 0.0

                subject_results[subject_folder] = {
                    "avg_total_updates": avg_total,
                    "avg_skipped_updates": avg_skipped,
                    "avg_performed_updates": avg_performed,
                    "skipped_fraction": skipped_fraction,
                }

                #print(
                #    f"{path_key} | {subject_folder} | "
                #    f"Skipped fraction: {skipped_fraction:.3f}"
                #)


        # aggregate across subjects
        all_subject_fractions = [
            v["skipped_fraction"]
            for v in subject_results.values()
        ]

        aggregated_results[baseline][f"{path_key}_updates"] = {
            "per_subject": subject_results,
            "mean_skipped_fraction": np.mean(all_subject_fractions),
            "std_skipped_fraction": np.std(all_subject_fractions),
        }
        
    set_mapping = {
        "128": ["embed_dim_128_buffer_size_64_path", "embed_dim_128_buffer_size_32_path", "embed_dim_128_buffer_size_16_path"],
        "32": ["embed_dim_32_buffer_size_64_path", "embed_dim_32_buffer_size_32_path", "embed_dim_32_buffer_size_16_path"],
        "16": ["embed_dim_16_buffer_size_64_path", "embed_dim_16_buffer_size_32_path", "embed_dim_16_buffer_size_16_path"]
    }
    
    latex_table = generate_mmd_embeddings_and_buffer_sizes_overleaf_table(
        aggregated_results,
        set_mapping
    )
    
    print()
    print(latex_table)
    print()

def analyze_mmd_embeddings_and_buffer_sizes_ppg_ecg(baselines):
    baseline = 'feature_replay'
    path_keys = [ 
        "embed_dim_128_buffer_size_64_path",
        "embed_dim_128_buffer_size_32_path",
        "embed_dim_128_buffer_size_16_path",
        "embed_dim_32_buffer_size_64_path",
        "embed_dim_32_buffer_size_32_path",
        "embed_dim_32_buffer_size_16_path",
        "embed_dim_16_buffer_size_64_path",
        "embed_dim_16_buffer_size_32_path",
        "embed_dim_16_buffer_size_16_path"
    ]
    
    baseline_paths = baselines[f'{baseline}_ppg_ecg']
    aggregated_results = {}

    aggregated_results[baseline] = {}

    # ---------------------------------
    # Clinical Metrics (ME / STD / BHS)
    # ---------------------------------           
    for path_key in path_keys:
        dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    f"aggregate_{baseline}_metrics",
                    "evaluation_metrics.csv"
                ),
                usecols=["Type", "ME", "STD", "BHS_Grade"]            
            ),
        ]

        aggregated_results[baseline][f"{path_key}_clinical"] = (
            aggregate_seed_dataframes(
                dfs,
                bhs_columns=["BHS_Grade"]   # adapt if column name differs
            )
        )

    # -------------------------------------
    # Continual Learning Metrics (AE / BWT)
    # -------------------------------------
    for path_key in path_keys:
        sbp_dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    "sbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),
        ]

        # ---------------------------
        # DBP
        # ---------------------------
        dbp_dfs = [
            pd.read_csv(os.path.join(
                    baseline_paths[path_key],
                    "aggregate_metrics",
                    "dbp_aggregate_baseline_metrics.csv"
                ),
                usecols=["AE_mean", "BWT_mean"]  
            ),
        ]

        aggregated_results[baseline][f"{path_key}_sbp_cl"] = (
            aggregate_seed_dataframes(sbp_dfs)
        )

        aggregated_results[baseline][f"{path_key}_dbp_cl"] = (
            aggregate_seed_dataframes(dbp_dfs)
        )
    
    # -----------------
    # Number of updates 
    # -----------------
    for path_key in path_keys:

        # collect all seed-specific paths
        seed_paths = [
            baseline_paths[path_key],
        ]

        experiment_dir = Path(seed_paths[0])

        subject_results = {}

        # iterate subjects
        for item in experiment_dir.iterdir():

            if item.is_dir() and item.name.startswith("subject_"):

                subject_folder = item.name

                skipped_per_seed = []
                performed_per_seed = []
                total_per_seed = []

                # loop over seeds
                for seed_path in seed_paths:

                    csv_path = os.path.join(
                        seed_path,
                        subject_folder,
                        f'{baseline}',
                        "param_update_log.csv"
                    )

                    df = pd.read_csv(
                        csv_path,
                        usecols=["n_updated_params"]
                    )
                    #print(df)

                    n_total = len(df)
                    n_skipped = (df["n_updated_params"] == 0).sum()
                    n_performed = (df["n_updated_params"] > 0).sum()

                    #print(n_total, n_skipped, n_performed)
                    
                    total_per_seed.append(n_total)
                    skipped_per_seed.append(n_skipped)
                    performed_per_seed.append(n_performed)

                # averages over seeds
                avg_total = np.mean(total_per_seed)
                avg_skipped = np.mean(skipped_per_seed)
                avg_performed = np.mean(performed_per_seed)

                #print(avg_total, avg_skipped, avg_performed)
                
                # fraction of saved/skipped updates
                skipped_fraction = avg_skipped / avg_total if avg_total > 0 else 0.0

                subject_results[subject_folder] = {
                    "avg_total_updates": avg_total,
                    "avg_skipped_updates": avg_skipped,
                    "avg_performed_updates": avg_performed,
                    "skipped_fraction": skipped_fraction,
                }

                #print(
                #    f"{path_key} | {subject_folder} | "
                #    f"Skipped fraction: {skipped_fraction:.3f}"
                #)


        # aggregate across subjects
        all_subject_fractions = [
            v["skipped_fraction"]
            for v in subject_results.values()
        ]

        aggregated_results[baseline][f"{path_key}_updates"] = {
            "per_subject": subject_results,
            "mean_skipped_fraction": np.mean(all_subject_fractions),
            "std_skipped_fraction": np.std(all_subject_fractions),
        }
        
    set_mapping = {
        "256": ["embed_dim_128_buffer_size_64_path", "embed_dim_128_buffer_size_32_path", "embed_dim_128_buffer_size_16_path"],
        "64": ["embed_dim_32_buffer_size_64_path", "embed_dim_32_buffer_size_32_path", "embed_dim_32_buffer_size_16_path"],
        "32": ["embed_dim_16_buffer_size_64_path", "embed_dim_16_buffer_size_32_path", "embed_dim_16_buffer_size_16_path"]
    }
    
    latex_table = generate_mmd_embeddings_and_buffer_sizes_ppg_ecg_overleaf_table(
        aggregated_results,
        set_mapping
    )
    
    print()
    print(latex_table)
    print()
    

# ---------------------
# LaTeX Table Generator
# ---------------------
def generate_resource_profile_table(pi_profile, pixel_profile, config_file_path):

    baseline = 'Always-on'
    method = 'MMD'
    
    with open(config_file_path, "r") as f:
        setup = yaml.safe_load(f)
    embed_dim = int(setup.get('embed_dim'))
    
    pi_baseline = pi_profile[baseline]
    px_baseline = pixel_profile[baseline] 
    pi  = pi_profile[method]
    px  = pixel_profile[method]

    def fmt(mean, std, decimals=1):
        return rf"${mean:.{decimals}f} \pm {std:.{decimals}f}$"

    # ── Latency rows: (latex_label, dict_key, indent_level) ───────────────
    # indent 1 = \quad, indent 2 = \qquad (sub-components of adaptation_s)
    latency_rows = [
        (r'feat. extraction',                           'feature_extraction_s',       1),
        (r'BP prediction',                              'prediction_s',               1),
        (r'drift detection',                            'drift_detection_s',          1),
        (r'head adapt.',                                'head_adapt_s',               1),  
        (r'detector reinit.',                           'drift_detector_reinit_s',    1),  
    ]
    
    # cumulative latency over patients
    mmd_aggregate_latency_labels = {
        "MMD": "cumulative_latency_s",
    }
    always_on_aggregate_latency_labels = {
        "Always-on ": "cumulative_latency_s",
    }
    
    # ── Memory labels & keys ───────────────────────────────────────────────
    aggregate_memory_labels = {
        "peak process memory": "peak_process_rss_mb",
    }

    INDENT = {1: r'\quad', 2: r'\qquad'}

    rows = []

    # ── Header ─────────────────────────────────────────────────────────────
    rows.append(r"        \hline")
    rows.append(r"        \textbf{Metric} & \textbf{Raspberry Pi 5} & \textbf{Google Pixel 10a} \\")
    rows.append(r"        \hline")

    # ── Communication block ──────────────────────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Estimated Communication Requirements}} \\")
    rows.append(r"        \hline")
    rows.append(
        rf"        {INDENT[1]} Setting A & {estimate_communication_costs('A', config_file_path)} kB & {estimate_communication_costs('A', config_file_path)} kB \\"
    )
    rows.append(
        rf"        {INDENT[1]} Setting B & {estimate_communication_costs('B', config_file_path)} MB & {estimate_communication_costs('B', config_file_path)} MB \\"
    )
    rows.append(r"        \hline")
    
    # ── Latency block ──────────────────────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Comput. Requirements $\sim$ Update Latency Breakdown (ms)}} \\")
    rows.append(r"        \hline")
    
    for label, key, level in latency_rows:
        pi_m, pi_s = pi['latency_ms'][key]
        px_m, px_s = px['latency_ms'][key]
        
        indent = INDENT[level]
        rows.append(
            rf"        {indent} {label} & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
        )
    rows.append(r"        \hline")
    
    rows.append(r"        \multicolumn{3}{l}{\textit{Cumulative Latency for 88 subj. (min)}} \\")
    rows.append(r"        \hline")
    
    # MMD cumulative latency 
    for label, key in mmd_aggregate_latency_labels.items():
        pi_m, pi_s = pi['aggregate_latency_ms'][key]['min']
        px_m, px_s = px['aggregate_latency_ms'][key]['min']
        
        rows.append(
            rf"        \quad {label} & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
        )

    # Always-on cumulative latency 
    for label, key in always_on_aggregate_latency_labels.items():
        pi_m, pi_s = pi_baseline['aggregate_latency_ms'][key]['min']
        px_m, px_s = px_baseline['aggregate_latency_ms'][key]['min']
        
        rows.append(
            rf"        \quad {label} & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
        )
    rows.append(r"        \hline")
    
    # ── Memory block ───────────────────────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Memory Requirements $\sim$ Avg. Memory (MB)}} \\")
    rows.append(r"        \hline")
    for label, key in aggregate_memory_labels.items():
        pi_m, pi_s = pi['aggregate_memory_mb'][key]
        px_m, px_s = px['aggregate_memory_mb'][key]
        rows.append(
            rf"        \quad & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
        )
    rows.append(r"        \hline")

    # ── Update-frequency-reduction row ─────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Avg.\ Update Frequency Reduction (\%)}} \\")
    rows.append(r"        \hline")
    pi_m, pi_s = pi['update_freq_reduction']
    px_m, px_s = px['update_freq_reduction']
    rows.append(
        rf"        \quad & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
    )
    rows.append(r"        \hline")

    # ── Sample-annotation-reduction row ─────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Avg.\ Annotations required for one subj. (\# samples)}} \\")
    rows.append(r"        \hline")
    pi_m, pi_s = pi['annotation_samples_reduction']
    px_m, px_s = px['annotation_samples_reduction']
    pi_baseline_m, pi_baseline_s = pi_baseline['annotation_samples_reduction']
    px_baseline_m, px_baseline_s = px_baseline['annotation_samples_reduction']
    rows.append(
        rf"        \quad MMD & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
    )
    rows.append(
        rf"        \quad Always-on & {fmt(pi_baseline_m, pi_baseline_s)} & {fmt(px_baseline_m, px_baseline_s)} \\"
    )
    rows.append(r"        \hline")

    # ── Assemble full table ────────────────────────────────────────────────
    table = "\n".join([
        r"\begin{table}",
        r"    \centering",
        r"    \caption{Resource profiling (over three seeds) of feature replay with MMD, emebedding size " + f"{embed_dim}" + " and buffer size 16.}",
        r"    \begin{tabular}{l|c|c}",
        "\n".join(rows),
        r"    \end{tabular}",
        r"    \label{tab:table_4}",
        r"\end{table}",
    ])

    print(table)
    

def generate_gradual_vs_mixed_vs_abrupt_overleaf_table(
    aggregated_results,
    baseline_display_names,
    set_mapping,
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    def fmt(x):
        """
        Round numeric values to 1 decimals.
        """
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.1f}"
        return str(x)

    latex = []
    latex.append(r"\begin{table*}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Personalization results (over three seeds) on Vital DB for continuous SBP/DBP estimation from PPG, divided into the three subject sets 1/2/3. Parentheses indicate the target thresholds for the clinical standards for both SBP/DBP. CL algorithms are in light gray.}")
    latex.append(r"    \begin{tabular}{l|l|l|l|p{0.15\textwidth}|p{0.16\textwidth}|l}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Set} & \textbf{Algorithm} & \textbf{AE}$\downarrow$ & \textbf{BWT}$\downarrow$ & \makecell[l]{\textbf{ME}$\downarrow$ \\ ($<$5 mmHg)} & \makecell[l]{\textbf{STD}$\downarrow$ \\ ($<$8 mmHg)} & \makecell[l]{\textbf{BHS}$\uparrow$ \\ (A)} \\")
    latex.append(r"        \hline")

    first_set = True

    for set_id, shift_name in set_mapping.items():

        first_algo = True

        for baseline_key, display_name in baseline_display_names.items():

            # ---------------------------
            # Retrieve aggregated metrics
            # ---------------------------

            clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
            sbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_sbp_cl"]
            dbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_dbp_cl"]

            # -----------------------
            # Assumes single-row CSVs
            # -----------------------

            me_sbp_mean = fmt(clinical_df.iloc[0][f"{me_column}_mean"])
            me_sbp_std = fmt(clinical_df.iloc[0][f"{me_column}_std"])
            me_dbp_mean = fmt(clinical_df.iloc[1][f"{me_column}_mean"])
            me_dbp_std = fmt(clinical_df.iloc[1][f"{me_column}_std"])
            
            std_sbp_mean = fmt(clinical_df.iloc[0][f"{std_column}_mean"])
            std_sbp_std = fmt(clinical_df.iloc[0][f"{std_column}_std"])
            std_dbp_mean = fmt(clinical_df.iloc[1][f"{std_column}_mean"])
            std_dbp_std = fmt(clinical_df.iloc[1][f"{std_column}_std"])

            bhs_sbp = clinical_df.iloc[0][bhs_column]
            bhs_dbp = clinical_df.iloc[1][bhs_column]

            ae_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{ae_column}_mean"])
            ae_sbp_std = fmt(sbp_cl_df.iloc[0][f"{ae_column}_std"])
            ae_dbp_mean = fmt(dbp_cl_df.iloc[0][f"{ae_column}_mean"])
            ae_dbp_std = fmt(dbp_cl_df.iloc[0][f"{ae_column}_std"])
            
            bwt_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{bwt_column}_mean"])
            bwt_sbp_std = fmt(sbp_cl_df.iloc[0][f"{bwt_column}_std"])
            bwt_dbp_mean = fmt(dbp_cl_df.iloc[0][f"{bwt_column}_mean"])
            bwt_dbp_std = fmt(dbp_cl_df.iloc[0][f"{bwt_column}_std"])
            
            # ---------
            # Build row
            # ---------

            if first_algo:
                row = f"        {set_id} "
                first_algo = False
            else:
                row = "        "

            if display_name == 'feat.replay':
                row += r"& \cellcolor{lightgray!30}\textbf{feat.replay} "
                row += (
                    r"& \textbf{" + f"{ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f" / {ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f" / {bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f" / {me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f" / {std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{bhs_sbp} / {bhs_dbp}" + "} \\\\"
                )
            elif display_name in {'LwF', 'EWC', 'AGEM'}:
                row += r"& \cellcolor{lightgray!30}" + f"{display_name} "
                row += (
                    f"& {ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f" / {ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "} "
                    f"& {bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f" / {bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "} "
                    f"& {me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f" / {me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "} "
                    f"& {std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f" / {std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "} "
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            else:
                row += (
                    f"& {display_name} "
                    f"& {ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f" / {ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "} "
                    f"& {bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f" / {bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "} "
                    f"& {me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f" / {me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "} "
                    f"& {std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f" / {std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "} "
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            
                if display_name == 'online*' or display_name == 'cumulative-mean':
                    row += r"\cline{2-7}"
               
            latex.append(row)

        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_1}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def generate_gradual_vs_mixed_vs_abrupt_ppg_ecg_overleaf_table(
    aggregated_results,
    baseline_display_names,
    set_mapping,
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    def fmt(x):
        """
        Round numeric values to 1 decimals.
        """
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.1f}"
        return str(x)

    latex = []
    latex.append(r"\begin{table*}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Feature replay personalization results (seed 42) on Vital DB for continuous SBP/DBP estimation from PPG and PPG+ECG, divided into the three subject sets 1/2/3. Parentheses indicate the target thresholds for the clinical standards for both SBP/DBP.}")
    latex.append(r"    \begin{tabular}{l|l|l|l|p{0.15\textwidth}|p{0.16\textwidth}|l}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Set} & \textbf{Algorithm} & \textbf{AE}$\downarrow$ & \textbf{BWT}$\downarrow$ & \makecell[l]{\textbf{ME}$\downarrow$ \\ ($<$5 mmHg)} & \makecell[l]{\textbf{STD}$\downarrow$ \\ ($<$8 mmHg)} & \makecell[l]{\textbf{BHS}$\uparrow$ \\ (A)} \\")
    latex.append(r"        \hline")

    first_set = True

    for set_id, shift_name in set_mapping.items():

        first_algo = True

        for baseline_key, display_name in baseline_display_names.items():

            # ---------------------------
            # Retrieve aggregated metrics
            # ---------------------------

            clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
            sbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_sbp_cl"]
            dbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_dbp_cl"]

            # -----------------------
            # Assumes single-row CSVs
            # -----------------------

            me_sbp_mean = fmt(clinical_df.iloc[0][f"{me_column}_mean"])
            me_sbp_std = fmt(clinical_df.iloc[0][f"{me_column}_std"])
            me_dbp_mean = fmt(clinical_df.iloc[1][f"{me_column}_mean"])
            me_dbp_std = fmt(clinical_df.iloc[1][f"{me_column}_std"])
            
            std_sbp_mean = fmt(clinical_df.iloc[0][f"{std_column}_mean"])
            std_sbp_std = fmt(clinical_df.iloc[0][f"{std_column}_std"])
            std_dbp_mean = fmt(clinical_df.iloc[1][f"{std_column}_mean"])
            std_dbp_std = fmt(clinical_df.iloc[1][f"{std_column}_std"])

            bhs_sbp = clinical_df.iloc[0][bhs_column]
            bhs_dbp = clinical_df.iloc[1][bhs_column]

            ae_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{ae_column}_mean"])
            ae_sbp_std = fmt(sbp_cl_df.iloc[0][f"{ae_column}_std"])
            ae_dbp_mean = fmt(dbp_cl_df.iloc[0][f"{ae_column}_mean"])
            ae_dbp_std = fmt(dbp_cl_df.iloc[0][f"{ae_column}_std"])
            
            bwt_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{bwt_column}_mean"])
            bwt_sbp_std = fmt(sbp_cl_df.iloc[0][f"{bwt_column}_std"])
            bwt_dbp_mean = fmt(dbp_cl_df.iloc[0][f"{bwt_column}_mean"])
            bwt_dbp_std = fmt(dbp_cl_df.iloc[0][f"{bwt_column}_std"])
            
            # ---------
            # Build row
            # ---------

            if first_algo:
                row = f"        {set_id} "
                first_algo = False
            else:
                row = "        "

            if display_name == 'feat.replay (PPG+ECG)':
                row += r"& \textbf{feat.replay (PPG+ECG)} "
                row += (
                    r"& \textbf{" + f"{ae_sbp_mean}" + f" / {ae_dbp_mean}" + "} "
                    r"& \textbf{" + f"{bwt_sbp_mean}" + f" / {bwt_dbp_mean}" + "} "
                    r"& \textbf{" + f"{me_sbp_mean}" + f" / {me_dbp_mean}" + "} "
                    r"& \textbf{" + f"{std_sbp_mean}" + f" / {std_dbp_mean}" + "} "
                    r"& \textbf{" + f"{bhs_sbp} / {bhs_dbp}" + "} \\\\"
                )
            else:
                row += (
                    f"& {display_name} "
                    f"& {ae_sbp_mean}" f" / {ae_dbp_mean}"
                    f"& {bwt_sbp_mean}" + f" / {bwt_dbp_mean}"
                    f"& {me_sbp_mean}" + f" / {me_dbp_mean}"
                    f"& {std_sbp_mean}" + f" / {std_dbp_mean}"
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            
                if display_name == 'online*' or display_name == 'cumulative-mean':
                    row += r"\cline{2-7}"
               
            latex.append(row)

        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_1}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def generate_drift_detection_methods_overleaf_table(
    aggregated_results, 
    set_mapping,
    baseline_key="feature_replay",
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    def fmt(x):
        """
        Round numeric values to 1 decimals.
        """
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.1f}"
        return str(x)

    latex = []
    latex.append(r"\begin{table}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Drift-aware personalization results (over three seeds) for continuous SBP estimation from PPG (DBP omitted for conciseness). We report the fraction of skipped updates relatively to the always-on baseline.}")
    latex.append(r"    \begin{tabular}{p{1cm}|c|c|c|c}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Metric} & \textbf{Always} & \textbf{MMD} & \textbf{Random} & \textbf{LSDD} \\")
    latex.append(r"        \hline")
    
    for metric in ["AE", "BWT", "ME", "STD", "BHS", r"Skip\%"]:
        if metric == "AE":
            row = rf"       {metric}$\downarrow$"
        elif metric == "BWT":
            row = rf"       {metric}$\downarrow$"
        elif metric == "ME":
            row = rf"       {metric}$\downarrow$"
        elif metric == "STD":
            row = rf"       {metric}$\downarrow$"
        elif metric == "BHS":
            row = rf"       {metric}$\uparrow$"
        else:
            row = rf"       {metric}$\uparrow$"
        
        for algo, shift_name in set_mapping.items():
            if metric == "AE":
                sbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_sbp_cl"]
                dbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_dbp_cl"]
                
                ae_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{ae_column}_mean"])
                ae_sbp_std = fmt(sbp_cl_df.iloc[0][f"{ae_column}_std"])
                ae_dbp_mean = fmt(dbp_cl_df.iloc[0][f"{ae_column}_mean"])
                ae_dbp_std = fmt(dbp_cl_df.iloc[0][f"{ae_column}_std"])
                
                #if shift_name == "drift_aware_mmd_path":
                #    row += r"& \textbf{" + f"{ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f" / {ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "}" + "} "
                #else:
                #    row += f"& {ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f" / {ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "} "
                if shift_name == "drift_aware_mmd_path":
                    row += r"& \textbf{" + f"{ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + "} "
                else:
                    row += f"& {ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}"
                    
            elif metric == "BWT":
                sbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_sbp_cl"]
                dbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_dbp_cl"]
                
                bwt_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{bwt_column}_mean"])
                bwt_sbp_std = fmt(sbp_cl_df.iloc[0][f"{bwt_column}_std"])
                bwt_dbp_mean = fmt(dbp_cl_df.iloc[0][f"{bwt_column}_mean"])
                bwt_dbp_std = fmt(dbp_cl_df.iloc[0][f"{bwt_column}_std"])
                
                #if shift_name == "drift_aware_mmd_path":
                #    row += r"& \textbf{" + f"{bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f" / {bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "}" + "} "
                #else:
                #    row += f"& {bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f" / {bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "} "
                if shift_name == "drift_aware_mmd_path":
                    row += r"& \textbf{" + f"{bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + "} "
                else:
                    row += f"& {bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" 
                    
            elif metric == "ME":
                clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
                
                me_sbp_mean = fmt(clinical_df.iloc[0][f"{me_column}_mean"])
                me_sbp_std = fmt(clinical_df.iloc[0][f"{me_column}_std"])
                me_dbp_mean = fmt(clinical_df.iloc[1][f"{me_column}_mean"])
                me_dbp_std = fmt(clinical_df.iloc[1][f"{me_column}_std"])
                
                #if shift_name == "drift_aware_mmd_path":
                #   row += r"& \textbf{" + f"{me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f" / {me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "}" + "} "
                #else:
                #    row += f"& {me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f" / {me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "} "
                
                if shift_name == "drift_aware_mmd_path":
                    row += r"& \textbf{" + f"{me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + "} "
                else:
                    row += f"& {me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}"
                                       
            elif metric == "STD":
                clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
                
                std_sbp_mean = fmt(clinical_df.iloc[0][f"{std_column}_mean"])
                std_sbp_std = fmt(clinical_df.iloc[0][f"{std_column}_std"])
                std_dbp_mean = fmt(clinical_df.iloc[1][f"{std_column}_mean"])
                std_dbp_std = fmt(clinical_df.iloc[1][f"{std_column}_std"])
    
                #if shift_name == "drift_aware_mmd_path":
                #   row += r"& \textbf{" + f"{std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f" / {std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "}" + "} "
                #lse:
                #    row += f"& {std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f" / {std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "} "
                
                if shift_name == "drift_aware_mmd_path":
                    row += r"& \textbf{" + f"{std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + "} "
                else:
                    row += f"& {std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}"
                       
            elif metric == "BHS":
                clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
                
                bhs_sbp = clinical_df.iloc[0][bhs_column]
                bhs_dbp = clinical_df.iloc[1][bhs_column]
                
                if shift_name == "drift_aware_mmd_path":
                    row += r"& \textbf{" + f"{bhs_sbp}" + "} "
                else:
                    row += f"& {bhs_sbp}"
            else:
                mean_skipped_fraction = aggregated_results[baseline_key][f"{shift_name}_updates"]["mean_skipped_fraction"]
                mean_skipped_fraction = fmt(mean_skipped_fraction * 100)
                std_skipped_fraction = aggregated_results[baseline_key][f"{shift_name}_updates"]["std_skipped_fraction"]
                std_skipped_fraction = fmt(std_skipped_fraction * 100)
                
                if shift_name == "drift_aware_mmd_path":
                    row += r"& \textbf{" + f"{mean_skipped_fraction}" + r"{\tiny $\pm$" + f"{std_skipped_fraction}" + "}" + r"} "
                else:
                    row += f"& {mean_skipped_fraction}" + r"{\tiny $\pm$" + f"{std_skipped_fraction}" + "}"
                    
        latex.append(row + r" \\")

    latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_2}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_mmd_embeddings_and_buffer_sizes_overleaf_table(
    aggregated_results, 
    set_mapping,
    baseline_key="feature_replay",
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    def fmt(x):
        """
        Round numeric values to 1 decimals.
        """
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.1f}"
        return str(x)

    latex = []
    latex.append(r"\begin{table}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Ablation study on MMD, over smaller embedding (128/32/16) and buffer (64/32/16) sizes. For conciseness, we report the SBP results averaged over three seeds but w/o variance. For embedding size 8, MMD was numerically unstable.}")
    latex.append(r"    \begin{tabular}{p{0.5cm}|p{1.5cm}|p{1.5cm}|p{1.2cm}|p{2cm}}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Emb.} &  \textbf{ME$\downarrow$} & \textbf{STD$\downarrow$} & \textbf{BHS$\uparrow$} & \textbf{Skip\%$\uparrow$} \\")
    latex.append(r"        \hline")
    
    
    for embedding_size, path_keys in set_mapping.items():
        if int(embedding_size) == 16:
            row = f"        " + r"\textbf{" + f"{embedding_size}" + "} & "
        else:
            row = f"        {embedding_size} & "
        
        ## Average Error (ME) for SBP
        #for path in path_keys:
        #    sbp_cl_df = aggregated_results[baseline_key][f"{path}_sbp_cl"]
        #    ae_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{ae_column}_mean"])
        #    
        #    if int(path.split('_')[-2]) == 16:
        #        row += f"{ae_sbp_mean}"
        #    else:
        #        row += f"{ae_sbp_mean} / "
        #row += " & "
        
        # Average Mean Error (ME) for SBP
        for path in path_keys:
            clinical_df = aggregated_results[baseline_key][f"{path}_clinical"]
            me_sbp_mean = fmt(clinical_df.iloc[0][f"{me_column}_mean"])
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 16:
                    row += r"\textbf{" + f"{me_sbp_mean}" + "}"
                else:
                    row += f"{me_sbp_mean}"
            else:
                row += f"{me_sbp_mean} / "
        row += " & "
        
        # Average Mean Error Std (ME) for SBP
        for path in path_keys:
            clinical_df = aggregated_results[baseline_key][f"{path}_clinical"]
            std_sbp_mean = fmt(clinical_df.iloc[0][f"{std_column}_mean"])
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 16:
                    row += r"\textbf{" + f"{std_sbp_mean}" + "}"
                else:
                    row += f"{std_sbp_mean}"
            else:
                row += f"{std_sbp_mean} / "
        row += " & "
                
        # BHS Grade for SBP
        for path in path_keys:
            clinical_df = aggregated_results[baseline_key][f"{path}_clinical"]
            bhs_sbp = clinical_df.iloc[0][bhs_column]            
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 16:
                    row += r"\textbf{" + f"{bhs_sbp}" + "}"
                else:
                    row += f"{bhs_sbp}"
            else:
                row += f"{bhs_sbp} / "
        
        row += " & "
        
        # Skipped Updates Fraction      
        for path in path_keys:
            mean_skipped_fraction = aggregated_results[baseline_key][f"{path}_updates"]["mean_skipped_fraction"]
            mean_skipped_fraction = fmt(mean_skipped_fraction * 100)
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 16:
                    row += r"\textbf{" + f"{mean_skipped_fraction}" + "}"
                else:
                    row += f"{mean_skipped_fraction}"
            else:
                row += f"{mean_skipped_fraction} / "
                    
        latex.append(row + r" \\")
    
    latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_3}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_mmd_embeddings_and_buffer_sizes_ppg_ecg_overleaf_table(
    aggregated_results, 
    set_mapping,
    baseline_key="feature_replay",
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    def fmt(x):
        """
        Round numeric values to 1 decimals.
        """
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.1f}"
        return str(x)

    latex = []
    latex.append(r"\begin{table}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Ablation study on MMD, over smaller embedding (256/64/32) and buffer (64/32/16) sizes. For conciseness, we report only the SBP results.}")
    latex.append(r"    \begin{tabular}{p{0.5cm}|p{1.5cm}|p{1.5cm}|p{1.2cm}|p{2cm}}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Emb.} &  \textbf{ME$\downarrow$} & \textbf{STD$\downarrow$} & \textbf{BHS$\uparrow$} & \textbf{Skip\%$\uparrow$} \\")
    latex.append(r"        \hline")
    
    
    for embedding_size, path_keys in set_mapping.items():
        if int(embedding_size) == 64:
            row = f"        " + r"\textbf{" + f"{embedding_size}" + "} & "
        else:
            row = f"        {embedding_size} & "
        
        ## Average Error (ME) for SBP
        #for path in path_keys:
        #    sbp_cl_df = aggregated_results[baseline_key][f"{path}_sbp_cl"]
        #    ae_sbp_mean = fmt(sbp_cl_df.iloc[0][f"{ae_column}_mean"])
        #    
        #    if int(path.split('_')[-2]) == 16:
        #        row += f"{ae_sbp_mean}"
        #    else:
        #        row += f"{ae_sbp_mean} / "
        #row += " & "
        
        # Average Mean Error (ME) for SBP
        for path in path_keys:
            clinical_df = aggregated_results[baseline_key][f"{path}_clinical"]
            me_sbp_mean = fmt(clinical_df.iloc[0][f"{me_column}_mean"])
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 64:
                    row += r"\textbf{" + f"{me_sbp_mean}" + "}"
                else:
                    row += f"{me_sbp_mean}"
            else:
                row += f"{me_sbp_mean} / "
        row += " & "
        
        # Average Mean Error Std (ME) for SBP
        for path in path_keys:
            clinical_df = aggregated_results[baseline_key][f"{path}_clinical"]
            std_sbp_mean = fmt(clinical_df.iloc[0][f"{std_column}_mean"])
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 64:
                    row += r"\textbf{" + f"{std_sbp_mean}" + "}"
                else:
                    row += f"{std_sbp_mean}"
            else:
                row += f"{std_sbp_mean} / "
        row += " & "
                
        # BHS Grade for SBP
        for path in path_keys:
            clinical_df = aggregated_results[baseline_key][f"{path}_clinical"]
            bhs_sbp = clinical_df.iloc[0][bhs_column]            
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 64:
                    row += r"\textbf{" + f"{bhs_sbp}" + "}"
                else:
                    row += f"{bhs_sbp}"
            else:
                row += f"{bhs_sbp} / "
        
        row += " & "
        
        # Skipped Updates Fraction      
        for path in path_keys:
            mean_skipped_fraction = aggregated_results[baseline_key][f"{path}_updates"]["mean_skipped_fraction"]
            mean_skipped_fraction = fmt(mean_skipped_fraction * 100)
            
            if int(path.split('_')[-2]) == 16:
                if int(embedding_size) == 64:
                    row += r"\textbf{" + f"{mean_skipped_fraction}" + "}"
                else:
                    row += f"{mean_skipped_fraction}"
            else:
                row += f"{mean_skipped_fraction} / "
                    
        latex.append(row + r" \\")
    
    latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_3}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def generate_ppg_vs_ppg_ecg_overleaf_table(
    aggregated_results, 
    set_mapping,
    baseline_key="feature_replay",
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    def fmt(x):
        """
        Round numeric values to 1 decimals.
        """
        if isinstance(x, (int, float, np.floating)):
            return f"{x:.1f}"
        return str(x)
    
    latex = []
    latex.append(r"\begin{table*}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Personalization results on Vital DB for continuous SBP/DBP estimation from PPG vs PPG+ECG. The baseline employed for personalization is feature replay (reservoir buffer of 64 features)  on the gradual shifts set (1). The reported metrics are the same as those of ~\ref{tab:table_1}.}")
    latex.append(r"    \begin{tabular}{l|l|l|p{0.15\textwidth}|p{0.16\textwidth}|l}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Algorithm} & \textbf{AE}$\downarrow$ & \textbf{BWT}$\downarrow$ & \makecell[l]{\textbf{ME}$\downarrow$ \\ ($<$5 mmHg)} & \makecell[l]{\textbf{STD}$\downarrow$ \\ ($<$8 mmHg)} & \makecell[l]{\textbf{BHS}$\uparrow$ \\ (A)} \\")
    latex.append(r"        \hline")

    for algo, shift_name in set_mapping.items():

        # ---------------------------
        # Retrieve aggregated metrics
        # ---------------------------

        clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
        sbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_sbp_cl"]
        dbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_dbp_cl"]

        # -----------------------
        # Assumes single-row CSVs
        # -----------------------

        ae_sbp = fmt(sbp_cl_df.iloc[0][ae_column])
        ae_dbp = fmt(dbp_cl_df.iloc[0][ae_column])
        
        bwt_sbp = fmt(sbp_cl_df.iloc[0][bwt_column])
        bwt_dbp = fmt(dbp_cl_df.iloc[0][bwt_column])

        me_sbp = fmt(clinical_df.iloc[0][me_column])
        me_dbp = fmt(clinical_df.iloc[1][me_column])
        
        std_sbp = fmt(clinical_df.iloc[0][std_column])
        std_dbp = fmt(clinical_df.iloc[1][std_column])

        bhs_sbp = clinical_df.iloc[0][bhs_column]
        bhs_dbp = clinical_df.iloc[1][bhs_column]

        # ---------
        # Build row
        # ---------
        if algo == "PPG":
            row = (
                r"\textbf{PPG} "
                r"& \textbf{" + f"{ae_sbp} / {ae_dbp}" + "} "
                r"& \textbf{" + f"{bwt_sbp} / {bwt_dbp}" + "} "
                r"& \textbf{" + f"{me_sbp} / {me_dbp}" + "} "
                r"& \textbf{" + f"{std_sbp} / {std_dbp}" + "} "
                r"& \textbf{" + f"{bhs_sbp} / {bhs_dbp}" + "} \\\\"
            )
        else:
            row = (
                f"{algo} "
                f"& {ae_sbp} / {ae_dbp} "
                f"& {bwt_sbp} / {bwt_dbp} "
                f"& {me_sbp} / {me_dbp} "
                f"& {std_sbp} / {std_dbp} "
                f"& {bhs_sbp} / {bhs_dbp} \\\\"
            )
    
        latex.append(row)

        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_4}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def analyze_logs_and_plot(args):
    
    print(f"[Log Analysis] Analyzing logs ...")
    
    # Baselines and corresponding experiment folder
    # Collect results on gradual shifts/mixed/shifts/abrupt shifts for each baseline
    # -> experiment path is hardcoded, bad
    baselines_ppg = {
        'running_mean': {
            'gradual_shifts_path': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_gradual_shifts/personalization_running_mean_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_08_03-11_38_11",
            'gradual_shifts_path_seed_41': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_running_mean_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_08_03-11_41_44",
            'gradual_shifts_path_seed_40': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_running_mean_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_08_03-11_41_07",
            'mixed_shifts_path': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_mixed_shifts/personalization_running_mean_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_08_03-12_23_55",
            'mixed_shifts_path_seed_41': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_running_mean_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_08_03-12_26_12",
            'mixed_shifts_path_seed_40': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_running_mean_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_08_03-12_26_05",
            'abrupt_shifts_path': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_abrupt_shifts/personalization_running_mean_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_08_03-12_03_06", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_running_mean_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_08_03-12_05_58",
            'abrupt_shifts_path_seed_40': "./logs/personalization_running_mean_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_running_mean_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_08_03-12_06_25",
        },
        'no_adapt': {
            'gradual_shifts_path': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-18_31_51",
            'gradual_shifts_path_seed_41': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-10_18_43",
            'gradual_shifts_path_seed_40': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-10_18_11",
            'mixed_shifts_path': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-18_36_29",
            'mixed_shifts_path_seed_41': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-10_19_20",
            'mixed_shifts_path_seed_40': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-10_19_08",
            'abrupt_shifts_path': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-18_34_54", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-10_19_57",
            'abrupt_shifts_path_seed_40': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-10_19_46",
        }, 
        'first_batch_finetune': {
            'gradual_shifts_path': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_18-09_59_24",
            'gradual_shifts_path_seed_41': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-10_47_26",
            'gradual_shifts_path_seed_40': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-10_47_26",
            'mixed_shifts_path': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-09_59_11",
            'mixed_shifts_path_seed_41': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-11_19_06",
            'mixed_shifts_path_seed_40': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-11_19_06",
            'abrupt_shifts_path': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_18-09_59_39", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-10_55_56",
            'abrupt_shifts_path_seed_40': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-10_56_01",
        }, 
        'online': { 
            'gradual_shifts_path': "./logs/personalization_online_proto_ppg_calibration_size_1_gradual_shifts/personalization_online_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-19_07_08",
            'gradual_shifts_path_seed_41': "./logs/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-11_18_10",
            'gradual_shifts_path_seed_40': "./logs/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-11_18_10",
            'mixed_shifts_path': "./logs/personalization_online_proto_ppg_calibration_size_1_mixed_shifts/personalization_online_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-20_04_21",
            'mixed_shifts_path_seed_41': "./logs/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-12_16_55",
            'mixed_shifts_path_seed_40': "./logs/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-12_17_04",
            'abrupt_shifts_path': "./logs/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-19_28_05",
            'abrupt_shifts_path_seed_41': "./logs/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-11_30_58",
            'abrupt_shifts_path_seed_40': "./logs/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-11_30_58",
        }, 
        'online_from_scratch': {
            'gradual_shifts_path': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-19_54_16",
            'gradual_shifts_path_seed_41': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-11_51_09",
            'gradual_shifts_path_seed_40': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-11_51_09",
            'mixed_shifts_path': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-21_42_06",
            'mixed_shifts_path_seed_41': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-13_18_15",
            'mixed_shifts_path_seed_40': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-13_18_15",
            'abrupt_shifts_path': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-20_27_07", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-12_11_32",
            'abrupt_shifts_path_seed_40': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-12_11_18",
        }, 
        'feature_replay': {
            # Experiments over different shift types
            'gradual_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-20_41_18",
            'gradual_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-12_24_09",
            'gradual_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-12_24_13",
            'mixed_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-23_19_51",
            'mixed_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-14_19_00",
            'mixed_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-14_19_00",
            'abrupt_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-21_27_56", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-12_48_51",
            'abrupt_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-12_48_51",
            # Experiments over different drift detection mechanisms
            'drift_aware_mmd_path': "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd/personalization_feature_replay_proto_ppg_drift_aware_mmd-Proto-2026_08_06-10_15_20",
            'drift_aware_mmd_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_41-Proto-2026_08_06-10_16_01",
            'drift_aware_mmd_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_40-Proto-2026_08_06-10_15_44",
            'drift_aware_lsdd_path': "./logs/personalization_feature_replay_proto_ppg_drift_aware_lsdd/personalization_feature_replay_proto_ppg_drift_aware_lsdd-Proto-2026_08_06-10_16_19",
            'drift_aware_lsdd_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_41/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_41-Proto-2026_08_06-10_17_09",
            'drift_aware_lsdd_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_40/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_40-Proto-2026_08_06-10_17_03",
            'random_path': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_drift_aware_random/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_drift_aware_random-Proto-2026_08_07-15_31_15",
            'random_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_drift_aware_random_seed_41/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_drift_aware_random_seed_41-Proto-2026_08_07-15_37_36",
            'random_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_drift_aware_random_seed_40/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_drift_aware_random_seed_40-Proto-2026_08_07-15_37_33",
            'always_on_path': "./logs/personalization_feature_replay_proto_ppg_drift_aware_always_on/personalization_feature_replay_proto_ppg_drift_aware_always_on-Proto-2026_08_06-10_10_45",
            'always_on_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_drift_aware_always_on_seed_41/personalization_feature_replay_proto_ppg_drift_aware_always_on_seed_41-Proto-2026_08_06-10_14_49",
            'always_on_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_drift_aware_always_on_seed_40/personalization_feature_replay_proto_ppg_drift_aware_always_on_seed_40-Proto-2026_08_06-10_14_26",
            # Experiments over buffer sizes and model embedding sizes
            # embed dim 128 and buffer size 64 correspond to baseline MMD compared with Al;ways on, Random and LSDD 
            'embed_dim_128_buffer_size_64_path': "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd/personalization_feature_replay_proto_ppg_drift_aware_mmd-Proto-2026_08_06-10_15_20",
            'embed_dim_128_buffer_size_64_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_41-Proto-2026_08_06-10_16_01",
            'embed_dim_128_buffer_size_64_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_40-Proto-2026_08_06-10_15_44",
            'embed_dim_128_buffer_size_32_path': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_drift_aware_mmd-Proto-2026_08_08-18_32_06",
            'embed_dim_128_buffer_size_32_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_drift_aware_mmd_seed_41-Proto-2026_08_08-12_28_00",
            'embed_dim_128_buffer_size_32_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_drift_aware_mmd_seed_40-Proto-2026_08_08-12_28_46",
            'embed_dim_128_buffer_size_16_path': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_drift_aware_mmd-Proto-2026_08_08-18_32_27",
            'embed_dim_128_buffer_size_16_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_drift_aware_mmd_seed_41-Proto-2026_08_08-12_28_11",
            'embed_dim_128_buffer_size_16_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_drift_aware_mmd_seed_40-Proto-2026_08_08-12_28_57",
            'embed_dim_32_buffer_size_64_path': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_drift_aware_mmd-Proto-2026_08_07-23_11_14",
            'embed_dim_32_buffer_size_64_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_drift_aware_mmd_seed_41-Proto-2026_08_07-23_15_14",
            'embed_dim_32_buffer_size_64_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_drift_aware_mmd_seed_40-Proto-2026_08_07-23_14_43",
            'embed_dim_32_buffer_size_32_path': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_drift_aware_mmd-Proto-2026_08_08-02_45_33",
            'embed_dim_32_buffer_size_32_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_drift_aware_mmd_seed_41-Proto-2026_08_08-03_12_29",
            'embed_dim_32_buffer_size_32_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_drift_aware_mmd_seed_40-Proto-2026_08_08-03_09_43",
            'embed_dim_32_buffer_size_16_path': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_drift_aware_mmd-Proto-2026_08_08-06_31_52",            
            'embed_dim_32_buffer_size_16_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_drift_aware_mmd_seed_41-Proto-2026_08_08-06_56_55",
            'embed_dim_32_buffer_size_16_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_drift_aware_mmd_seed_40-Proto-2026_08_08-06_54_33",
            'embed_dim_16_buffer_size_64_path': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_drift_aware_mmd-Proto-2026_08_07-23_15_36",
            'embed_dim_16_buffer_size_64_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_drift_aware_mmd_seed_41-Proto-2026_08_07-23_16_09",
            'embed_dim_16_buffer_size_64_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_drift_aware_mmd_seed_40-Proto-2026_08_07-23_16_00",
            'embed_dim_16_buffer_size_32_path': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_drift_aware_mmd-Proto-2026_08_08-03_00_45",
            'embed_dim_16_buffer_size_32_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_drift_aware_mmd_seed_41-Proto-2026_08_08-03_05_26",
            'embed_dim_16_buffer_size_32_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_drift_aware_mmd_seed_40-Proto-2026_08_08-03_03_39",
            'embed_dim_16_buffer_size_16_path': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_drift_aware_mmd-Proto-2026_08_08-06_38_43",            
            'embed_dim_16_buffer_size_16_path_seed_41': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_drift_aware_mmd_seed_41-Proto-2026_08_08-06_40_30",
            'embed_dim_16_buffer_size_16_path_seed_40': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_drift_aware_mmd_seed_40-Proto-2026_08_08-06_36_55",
            # Experiments over different devices for deployment
            'pi_always_on_path' : "./logs/pi_deployment_feature_replay_always_on/pi_deployment_feature_replay_always_on-Proto-2026_08_08-13_06_53",
            'pi_always_on_path_seed_41' : "./logs/pi_deployment_feature_replay_always_on_seed_41/pi_deployment_feature_replay_always_on_seed_41-Proto-2026_08_08-16_08_04",
            'pi_always_on_path_seed_40' : "./logs/pi_deployment_feature_replay_always_on_seed_40/pi_deployment_feature_replay_always_on_seed_40-Proto-2026_08_08-15_00_41",
            'pi_drift_aware_mmd_path' : "./logs/pi_deployment_feature_replay_drift_aware_mmd/pi_deployment_feature_replay_drift_aware_mmd-Proto-2026_08_08-13_59_17",
            'pi_drift_aware_mmd_path_seed_41' : "./logs/pi_deployment_feature_replay_drift_aware_mmd_seed_41/pi_deployment_feature_replay_drift_aware_mmd_seed_41-Proto-2026_08_08-16_31_20",
            'pi_drift_aware_mmd_path_seed_40' : "./logs/pi_deployment_feature_replay_drift_aware_mmd_seed_40/pi_deployment_feature_replay_drift_aware_mmd_seed_40-Proto-2026_08_08-15_25_03",
            'pixel_always_on_path' : "./logs/pixel_deployment_feature_replay_always_on/pixel_deployment_feature_replay_always_on-Proto-2026_08_08-13_56_11",
            'pixel_always_on_path_seed_41' : "./logs/pixel_deployment_feature_replay_always_on_seed_41/pixel_deployment_feature_replay_always_on_seed_41-Proto-2026_08_08-21_31_32",
            'pixel_always_on_path_seed_40' : "./logs/pixel_deployment_feature_replay_always_on_seed_40/pixel_deployment_feature_replay_always_on_seed_40-Proto-2026_08_08-18_27_07",
            'pixel_drift_aware_mmd_path' : "./logs/pixel_deployment_feature_replay_drift_aware_mmd/pixel_deployment_feature_replay_drift_aware_mmd-Proto-2026_08_08-17_35_24",
            'pixel_drift_aware_mmd_path_seed_41' : "./logs/pixel_deployment_feature_replay_drift_aware_mmd_seed_41/pixel_deployment_feature_replay_drift_aware_mmd_seed_41-Proto-2026_08_08-20_39_05",
            'pixel_drift_aware_mmd_path_seed_40' : "./logs/pixel_deployment_feature_replay_drift_aware_mmd_seed_40/pixel_deployment_feature_replay_drift_aware_mmd_seed_40-Proto-2026_08_08-19_13_20",
        }, 
        'lwf': {
            'gradual_shifts_path': "./logs/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-21_34_18",
            'gradual_shifts_path_seed_41': "./logs/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-12_56_47",
            'gradual_shifts_path_seed_40': "./logs/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-12_56_44",
            'mixed_shifts_path': "./logs/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-00_51_12",
            'mixed_shifts_path_seed_41': "./logs/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-15_08_29",
            'mixed_shifts_path_seed_40': "./logs/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-15_08_30",
            'abrupt_shifts_path': "./logs/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-22_32_36", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-13_26_46",
            'abrupt_shifts_path_seed_40': "./logs/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-13_26_54",
        }, 
        'ewc': {
            'gradual_shifts_path': "./logs/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-22_21_27",
            'gradual_shifts_path_seed_41': "./logs/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-13_27_31",
            'gradual_shifts_path_seed_40': "./logs/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-13_27_31",
            'mixed_shifts_path': "./logs/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-01_47_44",
            'mixed_shifts_path_seed_41': "./logs/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-15_38_14",
            'mixed_shifts_path_seed_40': "./logs/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-15_38_14",
            'abrupt_shifts_path': "./logs/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-23_33_23", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-14_03_32",
            'abrupt_shifts_path_seed_40': "./logs/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-14_03_32",
        }, 
        'agem': {
            'gradual_shifts_path': "./logs/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-23_10_55",
            'gradual_shifts_path_seed_41': "./logs/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-13_59_53",
            'gradual_shifts_path_seed_40': "./logs/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-13_59_53",
            'mixed_shifts_path': "./logs/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-02_36_22",
            'mixed_shifts_path_seed_41': "./logs/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-16_08_27",
            'mixed_shifts_path_seed_40': "./logs/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-16_08_27",
            'abrupt_shifts_path': "./logs/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_18-00_30_19", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-14_39_24",
            'abrupt_shifts_path_seed_40': "./logs/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-14_39_23",
        },
    }
    
    baselines_ppg_ecg = {
        # Experiments with ECG (seed 42)
        'running_mean': {
            'abrupt_shifts_path': "./logs/personalization_running_mean_proto_ppg_ecg_calibration_size_1_abrupt_shifts/personalization_running_mean_proto_ppg_ecg_calibration_size_1_abrupt_shifts-Proto-2026_08_10-13_19_45",
            'gradual_shifts_path': "./logs/personalization_running_mean_proto_ppg_ecg_calibration_size_1_gradual_shifts/personalization_running_mean_proto_ppg_ecg_calibration_size_1_gradual_shifts-Proto-2026_08_10-13_49_56",
            'mixed_shifts_path': "./logs/personalization_running_mean_proto_ppg_ecg_calibration_size_1_mixed_shifts/personalization_running_mean_proto_ppg_ecg_calibration_size_1_mixed_shifts-Proto-2026_08_10-14_14_22",
        },
        'feature_replay_ppg': {
            'gradual_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-20_41_18",
            'mixed_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-23_19_51",
            'abrupt_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-21_27_56", 
        },
        'feature_replay_ppg_ecg': {
            'abrupt_shifts_path' : "./logs/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_abrupt_shifts/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_abrupt_shifts-Proto-2026_08_10-13_28_16",
            'gradual_shifts_path' : "./logs/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts-Proto-2026_08_10-13_28_31",
            'mixed_shifts_path' : "./logs/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_mixed_shifts/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_mixed_shifts-Proto-2026_08_10-13_28_46",
            # Experiments over different embedding and buffer sizes
            'embed_dim_128_buffer_size_64_path': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_128_buffer_size_64_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-13_43_33",
            'embed_dim_128_buffer_size_32_path': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_128_buffer_size_32_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-18_10_52",
            'embed_dim_128_buffer_size_16_path': "./logs/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_128_buffer_size_16_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-19_57_09",
            'embed_dim_32_buffer_size_64_path': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_32_buffer_size_64_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-13_45_39",
            'embed_dim_32_buffer_size_32_path': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_32_buffer_size_32_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-18_07_21",
            'embed_dim_32_buffer_size_16_path': "./logs/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_32_buffer_size_16_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-19_48_24",
            'embed_dim_16_buffer_size_64_path': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_16_buffer_size_64_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-13_45_47",
            'embed_dim_16_buffer_size_32_path': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_16_buffer_size_32_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-17_52_08",
            'embed_dim_16_buffer_size_16_path': "./logs/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_ecg_drift_aware_mmd/personalization_feature_replay_proto_embed_dim_16_buffer_size_16_ppg_ecg_drift_aware_mmd-Proto-2026_08_10-19_20_11",
            # Experiments over different devices for deployment
            'pi_always_on_path' : "./logs/pi_deployment_ppg_ecg_feature_replay_always_on/pi_deployment_ppg_ecg_feature_replay_always_on-Proto-2026_08_11-11_48_05",
            'pi_drift_aware_mmd_path' : "./logs/pi_deployment_ppg_ecg_feature_replay_drift_aware_mmd/pi_deployment_ppg_ecg_feature_replay_drift_aware_mmd-Proto-2026_08_11-12_16_26",
            'pixel_always_on_path' : "./logs/pixel_deployment_ppg_ecg_feature_replay_always_on/pixel_deployment_ppg_ecg_feature_replay_always_on-Proto-2026_08_11-11_49_52",
            'pixel_drift_aware_mmd_path' : "./logs/pixel_deployment_ppg_ecg_feature_replay_drift_aware_mmd/pixel_deployment_ppg_ecg_feature_replay_drift_aware_mmd-Proto-2026_08_11-13_48_32",            
        }
    }
    
    # -------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> all the baselines
    # -------------------------------------------------
    if not args.ecg:
        analyze_gradual_vs_mixed_vs_abrupt(baselines_ppg)
    else:
        analyze_gradual_vs_mixed_vs_abrupt_ppg_ecg(baselines_ppg_ecg)
        
    # ---------------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> only for feature replay with MMD vs LSDD
    # ---------------------------------------------------------
    if not args.ecg:
        analyze_drift_detection_methods(baselines_ppg)
    
    # ---------------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> only for feature replay with MMD but varying model embeddings and buffer sizes
    # ---------------------------------------------------------
    if not args.ecg:
        analyze_mmd_embeddings_and_buffer_sizes(baselines_ppg)
    else:
        analyze_mmd_embeddings_and_buffer_sizes_ppg_ecg(baselines_ppg_ecg)
    
    # -----------------------------
    # Resource Profiling Estimation
    # -----------------------------
    # NOTE: change when backbone is decided
    if not args.ecg:
        config_file_path_ppg = './checkpoints/proto_ppg_percentile_embed_dim_16_group_layer_norm_kq_4/proto_ppg_percentile_embed_dim_16_group_layer_norm_kq_4-Proto-2026_08_07-20_12_07/proto_ppg_percentile_embed_dim_16_group_layer_norm_kq_4/ckpt/config.yaml'
        resource_usage_profile_estimation(config_file_path_ppg)
    else:
        config_file_path_ppg_ecg = './checkpoints/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4-Proto-2026_08_10-09_46_20/proto_ppg_ecg_percentile_embed_dim_32_group_layer_norm_kq_4/ckpt/config.yaml'
        resource_usage_profile_estimation(config_file_path_ppg_ecg)
    
    # --------------------------
    # Resource Profiling
    # -> only for feature replay
    # --------------------------
    baseline = 'feature_replay'
    
    # Pi Profiling
    print("[Log Analysis] Raspberry Pi Model 5 ~ Resource Profiling")
    if not args.ecg:
        pi_profile = resource_profiling(
            baseline=baseline, 
            baselines=baselines_ppg,
            deployment_device='pi'
        )
    else:
        pi_profile_ppg_ecg = resource_profiling_ppg_ecg(
            baseline=baseline, 
            baselines=baselines_ppg_ecg,
            deployment_device='pi'
        )
    
    # Pixel Profiling
    print("[Log Analysis] Google Pixel 10a ~ Resource Profiling")
    if not args.ecg:
        pixel_profile = resource_profiling(
            baseline=baseline, 
            baselines=baselines_ppg,
            deployment_device='pixel'
        )
    else:
        pixel_profile_ppg_ecg = resource_profiling_ppg_ecg(
            baseline=baseline, 
            baselines=baselines_ppg_ecg,
            deployment_device='pixel'
        )
    
    if not args.ecg:
        generate_resource_profile_table(pi_profile, pixel_profile, config_file_path_ppg)
    else:
        generate_resource_profile_table(pi_profile_ppg_ecg, pixel_profile_ppg_ecg, config_file_path_ppg_ecg)
        
    print("[Log Analysis] Results Analysis Completed ✓")
    
    
def parseargs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_folder_path', default='', type=str, help='path to the logs folder with the experiment to analyze')
    parser.add_argument('--config_yaml_path', default='', type=str, help='path to the configuration YAML file for the experiments')
    parser.add_argument('--fig_root', default='', type=str, help='path to figure folder where to store the result analysis outputs')
    parser.add_argument('--exp_fig_root', default='', type=str, help='path to figure folder of each subject analyzed in an experiment (subfolder inside fig_root)')
    parser.add_argument('--exp_pi_deployment_drift_aware_fig_root', default='', type=str, help='deployment on Pi drift-aware experiment folder')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='whether to load only ecg or not')
        
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    
    # Collect metrics from the log file folder
    analyze_logs_and_plot(args)
    
    
        
