
from collections import OrderedDict
import os
import sys
folders_to_add = ['models']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import shutil
import argparse
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import torch
from thop import profile
from models.Proto import Proto
from models.component_factory import BPRegressor


def analyze_logs_and_plot(log_file_path, fig_root):
    
    print("[Log Analysis] Analyzing log files ...")
    for baseline_name in [
        'no_adapt',
        'first_batch_finetune',
        'online',
        'online_from_scratch',
        'feature_replay',
        'lwf',
        'ewc',
        'agem'
    ]:
        baseline_fig_root = os.path.join(fig_root, baseline_name)
        if os.path.exists(baseline_fig_root):
            shutil.rmtree(baseline_fig_root)
        os.makedirs(baseline_fig_root)

        OUTPUT_SUMMARY_CSV = os.path.join(baseline_fig_root, "parsed_per_patient_summary.csv")
        OUTPUT_MAE_PLOT = os.path.join(baseline_fig_root, "parsed_per_patient_mae_ranked.png")
        OUTPUT_GRADE_PLOT = os.path.join(baseline_fig_root, "parsed_bhs_grade_distribution.png")


        # --------------------------------
        # Regex patterns matching the logs
        # --------------------------------

        re_subject_header = re.compile(
            r"Results for (p\d+) with baseline ([a-zA-Z0-9_]+)"
        )

        re_sbp_mae = re.compile(
            r"SBP MAE μ ([\d\.]+) ± ([\d\.]+)"
        )

        re_dbp_mae = re.compile(
            r"DBP MAE μ ([\d\.]+) ± ([\d\.]+)"
        )

        re_sbp_bhs_perc = re.compile(
            r"SBP:\s*([\d\.]+)%\s*([\d\.]+)%\s*([\d\.]+)%"
        )

        re_dbp_bhs_perc = re.compile(
            r"DBP:\s*([\d\.]+)%\s*([\d\.]+)%\s*([\d\.]+)%"
        )

        re_sbp_grade = re.compile(
            r"BHS standard grade for SBP: ([A-D])"
        )

        re_dbp_grade = re.compile(
            r"BHS standard grade for DBP: ([A-D])"
        )


        records = []
        current = {}

        with open(log_file_path, "r") as f:
            for line in f.readlines():
                line = line.strip()

                # ---------------------------
                # Detect header: subject + baseline
                # ---------------------------
                m = re_subject_header.search(line)
                if m:
                    # If we were collecting a previous block, store it
                    if "subject" in current:
                        records.append(current)
                    subject_id, baseline = m.groups()
                    current = {
                        "subject": subject_id,
                        "baseline": baseline,
                    }
                    continue

                # ---------------------------
                # Extract SBP/DBP MAE ± std
                # ---------------------------
                m = re_sbp_mae.search(line)
                if m:
                    current["SBP_MAE"] = float(m.group(1))
                    current["SBP_STD"] = float(m.group(2))
                    continue

                m = re_dbp_mae.search(line)
                if m:
                    current["DBP_MAE"] = float(m.group(1))
                    current["DBP_STD"] = float(m.group(2))
                    continue

                # ---------------------------
                # Extract BHS percentage lines
                # ---------------------------
                m = re_sbp_bhs_perc.search(line)
                if m:
                    current["SBP_p<=5"] = float(m.group(1))
                    current["SBP_p<=10"] = float(m.group(2))
                    current["SBP_p<=15"] = float(m.group(3))
                    continue

                m = re_dbp_bhs_perc.search(line)
                if m:
                    current["DBP_p<=5"] = float(m.group(1))
                    current["DBP_p<=10"] = float(m.group(2))
                    current["DBP_p<=15"] = float(m.group(3))
                    continue

                # ---------------------------
                # Extract BHS letter grade
                # ---------------------------
                m = re_sbp_grade.search(line)
                if m:
                    current["SBP_BHS"] = m.group(1)
                    continue

                m = re_dbp_grade.search(line)
                if m:
                    current["DBP_BHS"] = m.group(1)
                    continue

        # Add final block
        if "subject" in current:
            records.append(current)

        # ---------------------------
        # Create DataFrame
        # ---------------------------
        df = pd.DataFrame(records)

        # N.B. — filter only continual_replay baseline
        df = df[df["baseline"] == baseline_name].reset_index(drop=True)
        df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"[Log Analysis] Saved → {OUTPUT_SUMMARY_CSV}")

        # Sort by increasing SBP MAE
        df_sbp_sorted = df.sort_values(by="SBP_MAE", ascending=True).reset_index(drop=True)

        # Save to CSV
        df_sbp_sorted.to_csv(os.path.join(baseline_fig_root, "df_sorted_by_SBP_MAE.csv"),
                            index=False)

        print("[Log Analysis] Saved → df_sorted_by_SBP_MAE.csv")

        # --------
        # PLOTTING
        # --------

        # Plot 1 — Ranked MAE
        plt.figure(figsize=(12, 6))
        plt.plot(df["SBP_MAE"].sort_values().values, label="SBP MAE")
        plt.plot(df["DBP_MAE"].sort_values().values, label="DBP MAE")
        plt.title("Per-Patient MAE (Ranked) — Parsed From Logs")
        plt.xlabel("Patients (sorted)")
        plt.ylabel("MAE (mmHg)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_MAE_PLOT, dpi=300)
        plt.close()
        print(f"[Log Analysis] Saved → {OUTPUT_MAE_PLOT}")


        # Plot 2 — BHS Grade distribution
        print(baseline_name)
        plt.figure(figsize=(10, 4))

        grades_sbp = df["SBP_BHS"].value_counts()
        grades_dbp = df["DBP_BHS"].value_counts()

        # Build a common index (union of SBP & DBP grades)
        all_grades = sorted(set(grades_sbp.index).union(set(grades_dbp.index)))

        # Reindex so both have the same length
        grades_sbp = grades_sbp.reindex(all_grades, fill_value=0)
        grades_dbp = grades_dbp.reindex(all_grades, fill_value=0)

        width = 0.35
        idx = np.arange(len(all_grades))

        plt.bar(idx - width / 2, grades_sbp.values, width, label="SBP")
        plt.bar(idx + width / 2, grades_dbp.values, width, label="DBP")

        plt.xticks(idx, all_grades)
        plt.ylabel("Number of Patients")
        plt.title("BHS Grade Distribution — Parsed From Logs")
        plt.grid(axis="y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_GRADE_PLOT, dpi=300)
        plt.close()

        print(f"[Log Analysis] Saved → {OUTPUT_GRADE_PLOT}")

        
def aggregate_patient_level_target_statistics_and_plot(fig_root, exp_fig_root):
    
    print("[Result Analysis] Patient level SBP statistics vs SBP MAE ...")
    for baseline_name in [
        #'no_adapt',
        #'first_batch_finetune',
        #'online',
        #'online_from_scratch',
        'feature_replay',
        #'lwf',
        #'ewc',
        #'agem'
    ]:
        baseline_fig_root = os.path.join(fig_root, baseline_name)

        # Read CSV file  with sorted SBP MAE
        df_sbp_sorted = pd.read_csv(os.path.join(baseline_fig_root, "df_sorted_by_SBP_MAE.csv"))
        
        personalization_subjects = df_sbp_sorted['subject'].tolist()
        rows = []

        for subject_id in personalization_subjects:
            path = os.path.join(
                exp_fig_root,
                f"subject_{subject_id}",
                f"subject_{subject_id}_target_stats_over_time.csv"
            )
            df = pd.read_csv(path)

            rows.append({
                "subject": subject_id,
                "mean_SBP": df["mean_SBP"].mean(),
                "std_SBP": df["std_SBP"].mean(),
                "std_mean_SBP_over_time": df["mean_SBP"].std(),
                "mean_std_SBP_over_time": df["std_SBP"].mean()
            })

        df_patients = pd.DataFrame(rows)       
        df_patients.to_csv(
            os.path.join(baseline_fig_root, "patient_target_summary.csv"),
            index=False
        )
     
        df_merged = df_sbp_sorted.merge(df_patients, on="subject")
        df_merged = df_merged.sort_values("SBP_MAE")

        df_merged.to_csv(
            os.path.join(baseline_fig_root, "patient_sbp_mae_with_target_stats.csv"), 
            index=False
        )
        
        # --------
        # PLOTTING
        # --------
        
        df = df_merged.sort_values("SBP_MAE").reset_index(drop=True)
        
        # Plot 1: SBP variability vs personalization difficulty
        rank = np.arange(len(df))

        plt.figure(figsize=(12, 8))
        plt.plot(rank, df["std_mean_SBP_over_time"], label="Temporal drift (std of mean SBP)")
        plt.plot(rank, df["mean_std_SBP_over_time"], label="Avg within-block SBP std")

        plt.xlabel("Patient rank (low → high SBP MAE)")
        plt.ylabel("SBP variability [mmHg]")
        plt.title("SBP variability vs personalization difficulty")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(baseline_fig_root, "sbp_variability_vs_personalization_difficulty.png"),
            dpi=300
        )
        plt.close()
        print("[Result Analysis] Saved → sbp_variability_vs_personalization_difficulty.png")
        
        # Plot 2: Rank-segmented violin / box plots
        # Create rank bins
        df["rank_bin"] = pd.qcut(
            df.index,
            q=[0, 0.25, 0.75, 1.0],
            labels=["Low", "Medium", "High"]
        )

        plt.figure(figsize=(10, 7))
        sns.violinplot(
            data=df,
            x="rank_bin",
            y="std_mean_SBP_over_time",
            inner="box"
        )

        plt.xlabel("Patient group (by SBP MAE)")
        plt.ylabel("Temporal SBP drift")
        plt.title("SBP drift across personalization difficulty regimes")
        plt.tight_layout()
        plt.savefig(
            os.path.join(baseline_fig_root, "sbp_drift_across_personalization_difficulty.png"),
            dpi=300
        )
        plt.close()
        print("[Result Analysis] Saved → sbp_drift_across_personalization_difficulty.png")
        
        # Plot 3: 2D density + contour plot (subgroups as regions)
        plt.figure(figsize=(10, 9))
        sns.kdeplot(
            data=df,
            x="std_mean_SBP_over_time",
            y="mean_std_SBP_over_time",
            fill=True,
            cmap="Blues",
            thresh=0.05
        )

        plt.scatter(
            df["std_mean_SBP_over_time"],
            df["mean_std_SBP_over_time"],
            c=df["SBP_MAE"],
            cmap="plasma",
            s=30,
            edgecolor="k",
            alpha=0.7
        )

        plt.xlabel("Temporal SBP drift")
        plt.ylabel("Avg within-block SBP std")
        plt.title("Patient density w.r.t. SBP variability")
        plt.colorbar(label="SBP MAE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(baseline_fig_root, "patient_density_in_sbp_variability.png"),
            dpi=300
        )
        plt.close()
        print(f"[Result Analysis] Saved → patient_density_in_sbp_variability.png")
        
        # ---- Correlation Analysis ----
        rho, p = spearmanr(
            df_merged["std_mean_SBP_over_time"],
            df_merged["SBP_MAE"]
        )

        print(f"[Result Analysis] Correlation between temporal SBP drift and personalization MAE for {baseline_name}:")
        print(f"\t Spearman ρ = {rho:.3f}")
        print(f"\t R² (ρ²) = {rho**2:.3f}")
        print(f"\t p-value = {p:.2e}")
    

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


def collect_subject_metrics(exp_root, total_adapt_macs, total_adapt_prediction_macs, total_test_macs, drift_aware=False):
    rows = []

    for subj_dir in os.listdir(exp_root):
        if not subj_dir.startswith("subject_"):
            continue

        subject_id = subj_dir.split("_")[1]

        path = os.path.join(
            exp_root,
            subj_dir,
            "feature_replay",
            "param_update_log.csv"
        )

        df = pd.read_csv(path)
        total_blocks = df.shape[0]
        num_updates = (df["n_updated_params"] > 0).sum()

        total_adapt_macs_patient = total_adapt_macs * num_updates
        total_adapt_pred_macs_patient = total_adapt_prediction_macs * total_blocks
        total_test_macs_patient = total_test_macs * total_blocks

        total_macs_patient = (
            total_adapt_macs_patient
            + total_adapt_pred_macs_patient
            + total_test_macs_patient
        )

        rows.append({
            "subject": subject_id,
            "drift_aware": drift_aware,
            "total_blocks": total_blocks,
            "num_updates": num_updates,
            "total_macs": total_macs_patient,
            "adapt_macs": total_adapt_macs_patient
        })

    return pd.DataFrame(rows)


def read_feature_replay_aa(csv_path):
    df = pd.read_csv(csv_path)
    aa_value = df.loc[df["Baseline"] == "feature_replay", "AA_mean"].iloc[0]
    return aa_value


def relative_degradation(aware, unaware):
    """
    Positive value = worse performance (higher AA).
    Negative value = improvement.
    """
    return (aware - unaware) / unaware * 100


def resource_usage_profile(config_file_path, root_savepath, exp_fig_root, exp_fig_root_drift_aware):
    
    OUT_FIG_ROOT = os.path.join(root_savepath, "resource_usage_profile")
    os.makedirs(OUT_FIG_ROOT, exist_ok=True)
        
    # Get configuration parameters
    with open(config_file_path, "r") as f:
        setup = yaml.safe_load(f)

    # Setup Configuration
    precision_bits = int(setup.get('precision_bits', 32)) # Deafult to float32 if not specified
    adapt_bs = int(setup.get('personalization_batch_size'))
    test_bs = int(setup.get('validation_batch_size'))
    
    # NOTE: during an update the best model is selected helding out 0.25 of the personalization batch as validation, 
    # therefore the personliazaiton batch size and the replayed samples are personalization_bs * 0.75
    # -> overestimation with the full adapt_bs
    num_replay_samples_per_update = adapt_bs
    
    input_seq_len_s = int(setup.get('input_seq_len_s'))
    sampling_frequency = setup.get('fs')
    input_channels = 2 if setup.get('ecg') else 1
    feature_embed_dim = int(setup.get('embed_dim'))
    replay_buffer_size = int(setup.get('replay_buffer_size'))
    
    steps_per_update = int(setup.get('personalization_steps'))
    alpha_backward = 2 # fwd+bwd MACs as 2 times the fwd MACs
    
    # ---- Resource Usage Estimation for the Model ----
    print("[Resource Usage Profile] Instantiating the encoder and the head ...")

    # Instantiate encoder
    encoder = Proto(setup.get('ecg'), sampling_frequency, input_seq_len_s, feature_embed_dim)
    x = torch.rand(1, input_seq_len_s * sampling_frequency, input_channels) # N.B. batch size 1 for profiling
    
    encoder_forward_macs_sample, encoder_params = profile(encoder, inputs=(x,))
    encoder_forward_m_macs_sample = (encoder_forward_macs_sample) / 1e6
    encoder_params_mb = bytes_from_params(encoder_params, precision_bits=precision_bits) / (1024**2)
    
    encoder_act_bytes = calculate_activation_memory(encoder, x)
    encoder_act_bytes_mb = encoder_act_bytes / (1024**2)
        
    print(f'[Resource Usage Profile] Proto Encoder has:') 
    print(f'\t- {encoder_params} params ({encoder_params_mb:.2f} MB)')
    print(f'\t- {encoder_forward_m_macs_sample:.2f} M MACs per sample')
    print(f'\t- {encoder_act_bytes_mb:.2f} MB forward peak activation bytes')
    
    # Instantiate head
    head = BPRegressor(encoder.embed_dim, 3)
    y = torch.rand(1, encoder.embed_dim) # N.B. batch size 1 for profiling
    
    head_forward_macs_sample, head_params = profile(head, inputs=(y,))
    head_forward_k_macs_sample = (head_forward_macs_sample) / 1e3
    head_params_kb = bytes_from_params(head_params, precision_bits=precision_bits) / 1024
    
    head_act_bytes = calculate_activation_memory(head, y)
    head_act_bytes_kb = head_act_bytes / 1024

    print(f'[Resource Usage Profile] Proto Head has:') 
    print(f'\t- {head_params} params ({head_params_kb:.2f} kB)')
    print(f'\t- {head_forward_k_macs_sample:.2f} k MACs per sample')
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
    
    # Compute AA, MACs and number of updates per drift unaware cases
    df_unaware = collect_subject_metrics(
        exp_fig_root,
        total_adapt_macs=total_adapt_macs,
        total_adapt_prediction_macs=total_adapt_prediction_macs,
        total_test_macs=total_test_macs,
        drift_aware=False
    )
    
    # Read AA without drift-aware
    sbp_path_unaware = os.path.join(
        exp_fig_root,
        "aggregate_metrics",
        "sbp_aggregate_baseline_metrics.csv"
    )
    dbp_path_unaware = os.path.join(
        exp_fig_root,
        "aggregate_metrics",
        "dbp_aggregate_baseline_metrics.csv"
    )
    aa_sbp_unaware = read_feature_replay_aa(sbp_path_unaware)
    aa_dbp_unaware = read_feature_replay_aa(dbp_path_unaware)

    # Compute AA, MACs and number of updates per drift unaware cases
    df_aware = collect_subject_metrics(
        exp_fig_root_drift_aware,
        total_adapt_macs=total_adapt_macs,
        total_adapt_prediction_macs=total_adapt_prediction_macs,
        total_test_macs=total_test_macs,
        drift_aware=True
    )
    
    # Read AA with drift-aware
    sbp_path_aware = os.path.join(
        exp_fig_root_drift_aware,
        "aggregate_metrics",
        "sbp_aggregate_baseline_metrics.csv"
    )
    dbp_path_aware = os.path.join(
        exp_fig_root_drift_aware,
        "aggregate_metrics",
        "dbp_aggregate_baseline_metrics.csv"
    )
    aa_sbp_aware = read_feature_replay_aa(sbp_path_aware)
    aa_dbp_aware = read_feature_replay_aa(dbp_path_aware)
    
    print(f"SBP AA (drift-unaware):     {aa_sbp_unaware:.4f}")
    print(f"SBP AA (drift-aware): {aa_sbp_aware:.4f}")

    print(f"DBP AA (drift-unaware):     {aa_dbp_unaware:.4f}")
    print(f"DBP AA (drift-aware): {aa_dbp_aware:.4f}")

    # Compute degradation
    sbp_deg = relative_degradation(aa_sbp_aware, aa_sbp_unaware)
    dbp_deg = relative_degradation(aa_dbp_aware, aa_dbp_unaware)

    # Report
    print(f"SBP AA (unaware → aware): {aa_sbp_unaware:.4f} → {aa_sbp_aware:.4f}")
    print(f"SBP AA degradation: {sbp_deg:+.2f}%")

    print(f"DBP AA (unaware → aware): {aa_dbp_unaware:.4f} → {aa_dbp_aware:.4f}")
    print(f"DBP AA degradation: {dbp_deg:+.2f}%")

    
    df_all = pd.concat([df_unaware, df_aware], ignore_index=True)
    
    # Report mean +/-std per subject (drift unware vs drift aware   )
    summary = (
        df_all
        .groupby("drift_aware")
        .agg(
            mean_updates=("num_updates", "mean"),
            std_updates=("num_updates", "std"),
            mean_total_macs=("total_macs", "mean"),
            std_total_macs=("total_macs", "std"),
        )
        .reset_index()
    )

    print(summary)
    
    # Relative MACs savings
    df_merged = pd.merge(
        df_unaware,
        df_aware,
        on="subject",
        suffixes=("_unaware", "_aware")
    )

    df_merged["relative_macs_savings"] = ((
        df_merged["total_macs_unaware"] - df_merged["total_macs_aware"]
    ) / df_merged["total_macs_unaware"]) * 100

    df_merged["relative_update_reduction"] = ((
        df_merged["num_updates_unaware"] - df_merged["num_updates_aware"]
    ) / df_merged["num_updates_unaware"]) * 100
        
    print(
        "Relative MAC savings: "
        f"{df_merged['relative_macs_savings'].mean():.3f} ± "
        f"{df_merged['relative_macs_savings'].std():.3f}"
    )

    print(
        "Relative update reduction: "
        f"{df_merged['relative_update_reduction'].mean():.3f} ± "
        f"{df_merged['relative_update_reduction'].std():.3f}"
    )

    # Read CSV file with patient ID, sorted by SBP MAE and with the SBP variability over time
    df_sbp = pd.read_csv(os.path.join(
        root_savepath, 
        "feature_replay",
        "patient_sbp_mae_with_target_stats.csv"
        )
    )
    
    # The column "std_mean_SBP_over_time" report the variability of SBP (higher for difficult patient)
    df_analysis = pd.merge(
        df_merged,
        df_sbp[["subject", "std_mean_SBP_over_time"]],
        on="subject"
    )
    df_analysis = df_analysis.sort_values("std_mean_SBP_over_time").reset_index(drop=True)

    # MACs savings vs SBP varaibility (i.e. patient difficulty)
    df_analysis["drift_bin"] = pd.qcut(
        df_analysis.index,
        q=[0, 0.25, 0.75, 1.0],
        labels=["Low", "Medium", "High"]
    )

    plt.figure(figsize=(10, 7))
    sns.violinplot(
        data=df_analysis,
        x="drift_bin",
        y="relative_macs_savings",
        inner="box"
    )

    plt.xlabel("SBP drift regime")
    plt.ylabel("Relative MAC savings")
    plt.title("Computational savings vs SBP drift variability")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root_savepath, "feature_replay", "mac_savings_vs_sbp_drift.png"),
        dpi=300
    )
    plt.close()
    
    # Relative MACs saving vs SBP variability (i.e. patient difficulty)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        df_analysis["std_mean_SBP_over_time"],
        df_analysis["relative_macs_savings"],
        alpha=0.7
    )

    plt.xlabel("SBP temporal variability")
    plt.ylabel("Relative MAC savings")
    plt.title("Computational savings vs SBP variability")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root_savepath, "feature_replay", "updates_vs_sbp_variability.png"),
        dpi=300
    )
    plt.close()
    
    # Number of updates vs SBP variability (i.e. patient difficulty)
    plt.figure(figsize=(10, 8))
    plt.scatter(
        df_analysis["std_mean_SBP_over_time"],
        df_analysis["num_updates_aware"],
        alpha=0.7
    )

    plt.xlabel("SBP temporal variability (mmHg)")
    plt.ylabel("Number of drift-aware updates")
    plt.title("Update frequency vs SBP variability")
    plt.tight_layout()
    plt.savefig(
        os.path.join(root_savepath, "feature_replay", "updates_vs_sbp_variability.png"),
        dpi=300
    )
    plt.close()

    
def parseargs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_file_path', default='', type=str, help='path to the log file to parse')
    parser.add_argument('--config_yaml_path', default='', type=str, help='path to the configuration YAML file for the experiments')
    parser.add_argument('--fig_root', default='', type=str, help='path to figure folder where to store the result analysis outputs')
    parser.add_argument('--exp_fig_root', default='', type=str, help='path to figure folder of each subject analyzed in an experiment (subfolder inside fig_root)')
    parser.add_argument('--exp_fig_root_drift_aware', default='', type=str, help='path to figure folder of each subject analyzed in a drift aware experiment (subfolder inside fig_root)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    
    analyze_logs_and_plot(args.log_file_path, args.fig_root)
    
    aggregate_patient_level_target_statistics_and_plot(args.fig_root, args.exp_fig_root)
    
    resource_usage_profile(args.config_yaml_path, args.fig_root, args.exp_fig_root, args.exp_fig_root_drift_aware)
    
        
