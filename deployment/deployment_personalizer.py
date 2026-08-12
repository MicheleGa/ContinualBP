import os
import sys
import copy
import json
import pickle
import subprocess
from collections import defaultdict
from statistics import mean
import pandas as pd
import numpy as np
import torch
from training_utils.metrics import call_metric
from data.online_dataset import OnlineSubjectDataset
from data.online_dataset_aurora import AuroraOnlineSubjectDataset
from data.preprocessing_utils.data_visualization import (
    plot_subject_annotation_blocks, plot_aurora_subject_annotation_blocks, 
)
from deployment.run_subject import run_subject


# -----------------------------
# Raspberry Pi remote execution
# -----------------------------
def _scp_pi(src: str, dst: str) -> None:
    """Run a single scp transfer, raising RuntimeError on failure.
 
    Args:
        src: Local path OR ``user@host:remote_path`` string.
        dst: Destination path (local or ``user@host:remote_path``).
    """
    cmd = ["scp"]
    cmd += [src, dst]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[Pi] scp failed ({src} -> {dst})\n"
            f"  stderr: {result.stderr.strip()}"
        )
 
 
def _ssh_pi(pi_user: str, pi_host: str, remote_cmd: str) -> str:
    """Execute *remote_cmd* on the Pi over SSH, returning captured stdout.
 
    Args:
        pi_user: SSH username.
        pi_host: Hostname / IP of the Pi.
        remote_cmd: Shell command to run on the Pi.
 
    Returns:
        Combined stdout string from the remote process.
 
    Raises:
        RuntimeError: If the remote command exits with a non-zero status.
    """
    cmd = ["ssh"]
    cmd += [f"{pi_user}@{pi_host}", remote_cmd]
 
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[Pi] SSH command failed on {pi_host}:\n"
            f"  cmd:    {remote_cmd}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result.stdout
 
 
def run_subject_on_pi(
    subject_id : str,
    local_baseline_path: str,
    data_stream_path: str,
    model_weights_path: str,
    config_path: str,
    pi_user: str,
    pi_host: str,
    pi_remote_dir: str
) -> tuple:
    r"""Run ``run_subject`` on a Raspberry Pi over SSH and return the results.
 
    The Pi is expected to have a **CLI wrapper script** (``pi_script``) that:
 
    1. Accepts four positional arguments::
 
           python3 run_subject_cli.py <data_pkl> <weights_pt> <config_pkl> <results_pkl>
 
    2. Calls ``run_subject(data_path, weights_path, config_path)`` internally.
    3. Saves the three return values ``(baseline_outputs, baseline_targets,
       profiling_report)`` as a single pickle at ``<results_pkl>``.
 
    A minimal ``run_subject_cli.py`` looks like::
 
        import sys, pickle
        from deployment.run_subject import run_subject
        data, weights, cfg, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        results = run_subject(data, weights, cfg)   # (outputs, targets, report)
        with open(out, "wb") as f:
            pickle.dump(results, f)
 
    Args:
        data_stream_path:  Local path to the serialised data blocks pickle.
        model_weights_path: Local path to the model checkpoint (.pt/.pth).
        config_path:        Local path to the experiment config pickle.
        device:             String that specifies where the computation will occur (cpu/gpu)
        pi_user:            SSH username on the Pi (e.g. ``"pi"``).
        pi_host:            Hostname or IP of the Pi (e.g. ``"raspberrypi.local"``).
        pi_remote_dir:      Absolute path to the working directory on the Pi
                            (e.g. ``"/home/pi/edge_project"``).
 
    Returns:
        A 3-tuple ``(baseline_outputs, baseline_targets, profiling_report)``
        identical to what the local ``run_subject`` would return.
 
    Raises:
        RuntimeError: On any SSH / scp failure or if the Pi script errors out.
    """
    target = f"{pi_user}@{pi_host}"
 
    # -------------------------------------------------------------
    # 1. Derive remote filenames (flat, no subdirectory collisions)
    # -------------------------------------------------------------
    data_remote = f"{pi_remote_dir}/{os.path.basename(data_stream_path)}"
    weights_remote = f"{pi_remote_dir}/{os.path.basename(model_weights_path)}"
    config_remote = f"{pi_remote_dir}/{os.path.basename(config_path)}"
    results_remote = f"{pi_remote_dir}/results_{subject_id}.pkl"
    results_local = os.path.join(local_baseline_path, f"results_from_pi_{subject_id}.pkl")
 
    try:
        # ----------------------------------------------
        # 2. Transfer data, weights and config to the Pi
        # ----------------------------------------------
        print(f"[Pi] Transferring files to {pi_host} ...")
        _scp_pi(data_stream_path,   f"{target}:{data_remote}")
        _scp_pi(model_weights_path, f"{target}:{weights_remote}")
        _scp_pi(config_path,        f"{target}:{config_remote}")
        print("[Pi] Transfer complete.")
  
        # --------------------------------
        # 3. Execute run_subject on the Pi
        # --------------------------------
        remote_cmd = (
            f"cd {pi_remote_dir} && "
            f"source ./venv/bin/activate && "
            f"python run_subject_cli.py "
            f"{data_remote} {weights_remote} {config_remote} {results_remote}"
        )
        print(f"[Pi] Running inference on {pi_host} ...")
        stdout = _ssh_pi(pi_user, pi_host, remote_cmd)
        if stdout.strip():
            print(f"[Pi] Remote output:\n{stdout.strip()}")
        print("[Pi] Inference complete.")
 
        ## --------------------------
        ## 4. Retrieve results pickle
        ## --------------------------
        print(f"[Pi] Fetching results from {pi_host} ...")
        _scp_pi(f"{target}:{results_remote}", results_local)
 
        with open(results_local, "rb") as f:
            baseline_outputs, baseline_targets, profiling_report = pickle.load(f)
        print("[Pi] Results loaded successfully.")
 
    finally:
        # ---------------------------------------------------------------
        # 5. Clean up remote files (best-effort – never crash the caller)
        # ---------------------------------------------------------------
        remote_files = " ".join([
            data_remote, weights_remote, config_remote, results_remote
        ])
        try:
            _ssh_pi(
                pi_user, pi_host,
                f"rm -f {remote_files}"
            )
            print(f"[Pi] Remote files removed from {pi_host}.")
        except RuntimeError as cleanup_err:
            print(f"[Pi] Warning: could not remove remote files: {cleanup_err}")
 
    return baseline_outputs, baseline_targets, profiling_report


def _scp_pixel(src: str, dst: str) -> None:
    """
    Run a single SCP transfer to/from the Google Pixel.

    Args:
        src: Local path OR remote path.
        dst: Destination path.
    """

    cmd = [
        "scp",
        "-P", "8022",
        src,
        dst
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[Pixel] scp failed ({src} -> {dst})\n"
            f"stderr:\n{result.stderr.strip()}"
        )
        
        
def _ssh_pixel(user: str, host: str, remote_cmd: str) -> str:
    """
    Execute a remote SSH command on the Google Pixel.
    """

    cmd = [
        "ssh",
        "-p", "8022",
        f"{user}@{host}",
        remote_cmd
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[Pixel] SSH command failed:\n"
            f"cmd: {remote_cmd}\n\n"
            f"stderr:\n{result.stderr.strip()}"
        )

    return result.stdout

        
def run_subject_on_pixel(
    subject_id: str,
    local_baseline_path: str,
    data_stream_path: str,
    model_weights_path: str,
    config_path: str,
    pixel_user: str,
    pixel_host: str,
    pixel_remote_dir: str
) -> tuple:
    """
    Run run_subject remotely on a Google Pixel via Termux SSH.
    """

    target = f"{pixel_user}@{pixel_host}"

    # ----------------
    # Remote filenames
    # ----------------
    # NOTE: do not touch the path syntax, they are setup to match the format with the port number
    data_remote = (
        f"{pixel_remote_dir}/"
        f"{os.path.basename(data_stream_path)}"
    )

    weights_remote = (
        f"{pixel_remote_dir}/"
        f"{os.path.basename(model_weights_path)}"
    )

    config_remote = (
        f"{pixel_remote_dir}/"
        f"{os.path.basename(config_path)}"
    )

    results_remote = (
        f"{pixel_remote_dir}/results_{subject_id}.pkl"
    )

    results_local = os.path.join(
        local_baseline_path,
        f"results_from_pixel_{subject_id}.pkl"
    )

    try:

        # --------------
        # Transfer files
        # --------------

        print(f"[Pixel] Transferring files to {pixel_host} ...")

        _scp_pixel(data_stream_path, f"{target}:{data_remote}")
        _scp_pixel(model_weights_path, f"{target}:{weights_remote}")
        _scp_pixel(config_path, f"{target}:{config_remote}")

        print("[Pixel] Transfer complete.")

        # --------------------
        # Run remote inference
        # --------------------

        remote_cmd = (
            f"proot-distro login ubuntu -- bash -c '"
            f"cd ~/ContinualBP/deployment && "
            f"source venv/bin/activate && "
            f"python {pixel_remote_dir}/run_subject_cli.py "
            f"{data_remote} "
            f"{weights_remote} "
            f"{config_remote} "
            f"{results_remote}"
            f"'"
        )

        print(f"[Pixel] Running inference on {pixel_host} ...")

        stdout = _ssh_pixel(pixel_user, pixel_host, remote_cmd)

        if stdout.strip():
            print(f"[Pixel] Remote output:\n{stdout.strip()}")

        print("[Pixel] Inference complete.")

        # -------------
        # Fetch results
        # -------------

        print(f"[Pixel] Fetching results from {pixel_host} ...")

        _scp_pixel(f"{target}:{results_remote}", results_local)

        with open(results_local, "rb") as f:
            baseline_outputs, baseline_targets, profiling_report = pickle.load(f)

        print("[Pixel] Results loaded successfully.")

    finally:

        # ----------------------------------------------------------
        # Cleanup remote files
        # ----------------------------------------------------------

        remote_files = " ".join([
            data_remote,
            weights_remote,
            config_remote,
            results_remote
        ])

        try:

            cleanup_cmd = (
                f"rm -f {remote_files}"
            )

            _ssh_pixel(
                pixel_user,
                pixel_host,
                cleanup_cmd
            )

            print(f"[Pixel] Remote files removed from {pixel_host}.")

        except RuntimeError as cleanup_err:

            print(
                f"[Pixel] Warning: could not remove remote files:\n"
                f"{cleanup_err}"
            )

    return (
        baseline_outputs,
        baseline_targets,
        profiling_report
    )
     
            
def personalize_feature_replay(baseline, dataset, subject_id, figs_subj_dir, logs_subj_dir, device, config):
    
    # ---- INITIALIZATION ----
    figs_baseline_path = os.path.join(figs_subj_dir, baseline)
    os.makedirs(figs_baseline_path, exist_ok=True)
    
    logs_baseline_path = os.path.join(logs_subj_dir, baseline)
    os.makedirs(logs_baseline_path, exist_ok=True)
    
    # Set stream kwargs based on the dataset and optionally
    # plot subject blocks and SBP/DBP/MAP drifts
    stream_kwargs = dict(
        subject_id=subject_id,
        batch_size=config['personalization_batch_size'],
        num_batches=config['num_batches'],
        num_blocks=config['num_blocks']
    )
    
    if config['plot_personalization']:    
        blocks = dataset.get_subject_blocks(**stream_kwargs)
        if 'aurora' in config['dataset_name'].lower():
            plot_aurora_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(figs_baseline_path, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
        
        elif 'vital_db' in config['dataset_name'].lower():
            plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath=os.path.join(figs_baseline_path, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=True)
        else:
            raise ValueError("Dataset not recognized for plotting annotation blocks. Supported: 'aurora', 'vital_db'.")
        
    # Send data, model, config to the edge device
    block_samples_id_list = list(dataset.get_subject_blocks(**stream_kwargs))
    blocks_list = []
    for block_idx in range(len(block_samples_id_list)):
        block_info = block_samples_id_list[block_idx]
        sample_ids = block_info["sample_ids"]
        sample_batch = [dataset.__getitem__(sid) for sid in sample_ids]
        signals = torch.stack([x for x, _, _ in sample_batch]).cpu().numpy().astype(np.float32)
        targets = torch.stack([y for _, y, _ in sample_batch]).cpu().numpy().astype(np.float32)
        blocks_list.append({
            "subject_id": block_info["subject_id"],
            "block_idx": block_info["block_idx"],
            "batch_idx": block_info["batch_idx"],
            "signals": signals,
            "targets": targets
        })
            
    # Data
    data_stream_path = os.path.join(logs_baseline_path, f"{subject_id}_blocks.pkl")
    with open(data_stream_path, "wb") as f:
        pickle.dump(blocks_list, f)
    
    # Model
    model_weights_path = config['pretrained_model_ckpt_path']
    
    # Config
    config_path = os.path.join(logs_baseline_path, f"./{subject_id}_experiment_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    
    # Run on-device: dispatch to Raspberry Pi, Google Pixel, or local machine
    if config['use_pi']:
        
        print('[Deployment] Performing deployment on the Raspberry Pi!')
        
        baseline_outputs, baseline_targets, profiling_report = run_subject_on_pi(
            subject_id=subject_id,
            local_baseline_path=logs_baseline_path,
            data_stream_path=data_stream_path,
            model_weights_path=model_weights_path,
            config_path=config_path,
            pi_user='iris',
            pi_host='toaster.local',
            pi_remote_dir='~/Documents/ContinualBP/deployment'
        )
        
        # run_subject_on_pi already removes remote files; only clean up local temps
        if os.path.exists(data_stream_path):
            os.remove(data_stream_path)
        if os.path.exists(config_path):
            os.remove(config_path)
            
    elif config['use_pixel']:
        print('[Deployment] Performing deployment on the Google Pixel phone!')

        baseline_outputs, baseline_targets, profiling_report = run_subject_on_pixel(
            subject_id=subject_id,
            local_baseline_path=logs_baseline_path,
            data_stream_path=data_stream_path,
            model_weights_path=model_weights_path,
            config_path=config_path,
            pixel_user='u0_a365',
            pixel_host='localhost',
            pixel_remote_dir='/data/data/com.termux/files/home/ContinualBP/deployment'
        )

        # Remove temporary local files
        if os.path.exists(data_stream_path):
            os.remove(data_stream_path)
        if os.path.exists(config_path):
            os.remove(config_path)
    else:
        
        print('[Deployment] Performing deployment on the laptop!')
        
        baseline_outputs, baseline_targets, profiling_report = run_subject(
            data_stream_path, model_weights_path, config_path, device, verbose=False
        )
        # Remove temporary files after local execution
        if os.path.exists(data_stream_path):
            os.remove(data_stream_path)
        if os.path.exists(config_path):
            os.remove(config_path)
    
    # Save targets/predictions log per baseline
    log_json_path = os.path.join(logs_baseline_path, "targets_log.json")
    with open(log_json_path, "w") as f:
        json.dump(np.concatenate(baseline_targets, axis=0).tolist(), f)
    
    log_json_path = os.path.join(logs_baseline_path, "predictions_log.json")
    with open(log_json_path, "w") as f:
        json.dump(np.concatenate(baseline_outputs, axis=0).tolist(), f)
        
    # Prepare concatenated outputs/targets
    outs_and_tgts = {}
    if len(baseline_outputs) > 0:
        outs_and_tgts = (
            np.concatenate([o for o in baseline_outputs if o is not None], axis=0),
            np.concatenate([t for t in baseline_targets if t is not None], axis=0)
        )
    else:
        outs_and_tgts = (np.empty((0,)), np.empty((0,)))
              
    return outs_and_tgts, profiling_report


def flatten_profiling_report(report):
    r"""
    Flatten a profiling report into a CSV-friendly dictionary.

    Per-step quantities are summarized into:
        - mean
        - total (sum)
        - count

    Boolean quantities are summarized as:
        - adaptation_rate (= mean of booleans)
        - n_true
        - count

    End-of-personalization quantities are copied unchanged.
    """

    flat = {}

    per_step = report.get("per_step", {})

    for key, values in per_step.items():

        if not isinstance(values, list):
            continue

        n = len(values)

        if n == 0:
            flat[f"{key}_mean"] = 0.0
            flat[f"{key}_sum"] = 0.0
            flat[f"{key}_count"] = 0
            continue

        # Boolean metrics (currently only did_adapt)
        if isinstance(values[0], bool):

            rate = float(np.mean(values))
            n_true = int(np.sum(values))

            if key == "did_adapt":
                flat["adaptation_rate"] = rate
                flat["n_adaptations"] = n_true
                flat["n_steps"] = n
            else:
                flat[f"{key}_mean"] = rate
                flat[f"{key}_sum"] = n_true
                flat[f"{key}_count"] = n

        else:

            flat[f"{key}_mean"] = float(np.mean(values))
            flat[f"{key}_sum"] = float(np.sum(values))
            flat[f"{key}_count"] = n

    #
    # Aggregate metrics produced at the end of run_subject()
    #

    final_keys = [
        "cumulative_latency_s",
        "model_memory_mb",
        "rss_before_tta_mb",
        "peak_tta_rss_mb",
        "peak_incremental_tta_mb",
        "peak_process_rss_mb",
        "tm_peak_run_kb",
    ]

    for key in final_keys:
        flat[key] = report.get(key, 0.0)

    return flat


def aggregate_profiling_reports(reports):
    r"""
    Aggregate profiling reports across subjects.

    For every flattened metric compute:

        mean
        std
        min
        max
    """

    if len(reports) == 0:
        return {}

    aggregate = defaultdict(list)

    for report in reports:

        flat = flatten_profiling_report(report)

        for key, value in flat.items():
            aggregate[key].append(value)

    summary = {}

    for key, values in aggregate.items():

        values = np.asarray(values, dtype=float)

        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))

    return summary


def personalization(config, device):
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
    
    # Global results per baseline
    global_outs_and_tgts = {
        b: [] for b in baselines 
    }
    
    global_profiling_reports = {
        b: [] for b in baselines
    }
    
    # Create aggregate directory
    figs_agg_dir = os.path.join(config['figure_path'], 'aggregate_metrics')
    if not os.path.exists(figs_agg_dir):
        os.makedirs(figs_agg_dir)
    
    logs_agg_dir = os.path.join(config['logs_path'], 'aggregate_metrics')
    if not os.path.exists(logs_agg_dir):
        os.makedirs(logs_agg_dir)
        
    if config['setup_type'] == 'drift':
        print(f"[Personalization] Personalization performed with feature-based drift detection")
    
    for subject_counter, subject_id in enumerate(personalization_subjects):
        print(f"[Personalization] {subject_counter + 1}/{len(personalization_subjects)} personalizing model on subject {subject_id}")
        
        figs_subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        os.makedirs(figs_subj_dir, exist_ok=True)
        
        logs_subj_dir = os.path.join(config['logs_path'], f"subject_{subject_id}")
        os.makedirs(logs_subj_dir, exist_ok=True)
        
        outs_and_tgts = {}
        profiling_reports = {}
        
        # Personalize each different baseline
        for b in baselines:
            print(f"[Personalization] Personalizing baseline {b} on subject {subject_id}")

            if b == 'feature_replay':
                baseline_outs_and_tgts, baseline_profiling_metrics = personalize_feature_replay(
                    baseline=b, 
                    dataset=online_physio_dataset, 
                    subject_id=subject_id, 
                    figs_subj_dir=figs_subj_dir,
                    logs_subj_dir=logs_subj_dir, 
                    device=device, 
                    config=config
                )
            else:    
                raise ValueError('Invalid baseline name!')
            
            # Collect results
            outs_and_tgts[b] = baseline_outs_and_tgts
            profiling_reports[b] = baseline_profiling_metrics
            if b in global_profiling_reports:
                global_profiling_reports[b].append(baseline_profiling_metrics)
            
        if outs_and_tgts is None:
            raise ValueError("Outputs and targets for a subject cannot be None ...")

        # ---- Call metric plots for each baseline ----
        for b, (outs, tgts) in outs_and_tgts.items():
            if outs.shape[0] > 0:
                print(f"[Personalization] {subject_counter + 1}/{len(personalization_subjects)} Results for {subject_id} with baseline {b}")
                call_metric(tgts, outs, config, figure_savepath=os.path.join(figs_subj_dir, b), log_savepath=os.path.join(logs_subj_dir, b), plot=True)
                # append to global for aggregated metrics later
                if b in global_outs_and_tgts:
                    global_outs_and_tgts[b].append((outs, tgts))
        
        # ---- Save profiling reports ----
        profiling_dir = os.path.join(logs_subj_dir, 'profiling')
        os.makedirs(profiling_dir, exist_ok=True)

        profiling_csv_rows = []

        for b, report in profiling_reports.items():
            # save raw json
            profiling_json_path = os.path.join(
                profiling_dir,
                f'{b}_profiling_report.json'
            )

            with open(profiling_json_path, 'w') as f:
                json.dump(report, f, indent=4)

            # save flattened csv row
            flat_report = flatten_profiling_report(report)
            flat_report['baseline'] = b
            flat_report['subject_id'] = subject_id

            profiling_csv_rows.append(flat_report)

        if len(profiling_csv_rows) > 0:
            profiling_df = pd.DataFrame(profiling_csv_rows)

            profiling_csv_path = os.path.join(
                profiling_dir,
                'profiling_summary.csv'
            )

            profiling_df.to_csv(profiling_csv_path, index=False)

        print(f"[Personalization] Personalization on {subject_counter + 1}/{len(personalization_subjects)} subject completed ✓")

        if config['num_personalization_subjects'] > 0 and subject_counter > config['num_personalization_subjects']:
            break

    # ---- Aggregate call_metric across subjects ----
    for b, data_list in global_outs_and_tgts.items():
        if len(data_list) > 0:
            print(f'[Personalization] Aggregated personalization results (BHS/AAMI/Bland-Altman/R²) for the baseline {b}')
            all_outs = np.concatenate([o for o, _ in data_list], axis=0)
            all_tgts = np.concatenate([t for _, t in data_list], axis=0)
            
            aggregated_metric_fig_path = os.path.join(figs_agg_dir, f"aggregate_{b}_metrics")
            os.makedirs(aggregated_metric_fig_path, exist_ok=True)
            aggregated_metric_log_path = os.path.join(logs_agg_dir, f"aggregate_{b}_metrics")
            os.makedirs(aggregated_metric_log_path, exist_ok=True)
            
            call_metric(all_tgts, all_outs, config, figure_savepath=aggregated_metric_fig_path, log_savepath=aggregated_metric_log_path, plot=True)

    print(f"[Personalization] Saved aggregated metrics to {figs_agg_dir} and {logs_agg_dir} ✓")

    # ---- Aggregate profiling metrics ----
    aggregate_profiling_dir = os.path.join(logs_agg_dir, 'profiling')
    os.makedirs(aggregate_profiling_dir, exist_ok=True)

    aggregate_rows = []

    for b, reports in global_profiling_reports.items():
        if len(reports) == 0:
            continue

        agg_report = aggregate_profiling_reports(reports)
        agg_report['baseline'] = b

        aggregate_rows.append(agg_report)

        agg_json_path = os.path.join(
            aggregate_profiling_dir,
            f'{b}_aggregate_profiling.json'
        )

        with open(agg_json_path, 'w') as f:
            json.dump(agg_report, f, indent=4)

    if len(aggregate_rows) > 0:
        aggregate_df = pd.DataFrame(aggregate_rows)

        aggregate_csv_path = os.path.join(
            aggregate_profiling_dir,
            'aggregate_profiling_summary.csv'
        )

        aggregate_df.to_csv(aggregate_csv_path, index=False)
        
    print(f"[Personalization] Personalization completed ✓")