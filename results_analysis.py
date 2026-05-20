
import os
from pathlib import Path
import sys
import shutil
import json
import pickle
import argparse
import re
from collections import Counter
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr


def resource_profiling(baseline, mmd_profiling_path, lsdd_profiling_path, deployment_device):
 
    mmd_profiling_path = Path(mmd_profiling_path)
    lsdd_profiling_path = Path(lsdd_profiling_path)
    
    # Containers 
    per_step_latency_keys = [
        # per TTA step (appended in loop)
        'feature_extraction_s',    # enc(signals) latency
        'prediction_s',            # pred(feats) latency
        'drift_detection_s',        # drift_detector.predict() loop latency
        'adaptation_s',            # entire do_adapt block latency
        'head_adapt_s',            # prediction head adapt latency
        'drift_detector_reinit_s', # drift detector re-init latency
        'total_step_s',            # wall time for the full TTA step
    ]
    
    # Memory key definitions
    # Scalar fields: one value per subject
    aggregate_memory_keys = [
        'model_memory_mb',         # fixed RAM cost of keeping the model in memory.
        'rss_before_tta_mb',       # fixed cost that must fit in RAM regardless of the data stream.
        'peak_tta_rss_mb',         # highest RSS observed during the TTA loop
        'peak_incremental_tta_mb', # memory the TTA algorithm itself requires on top of the loaded model
        'peak_process_rss_mb',     # process-wide RSS: upper-bound RAM budget
        'tm_peak_run_kb',          # peak Python-heap usage over the entire run (kB)
    ]
    
    # Per-step delta fields: aggregated as max-per-subject
    per_step_memory_keys = [
        'tm_current_kb',            # absolute Python-heap usage at the end of each step
        'tm_head_adapt_kb',         # delta during head adaptation
        'tm_detector_reinit_kb',    # delta during detector reinit
    ]
 
    # Per_subject_means[key] = list of one mean per subject
    # -> give same weight to subjects with lots/few adaptations
    per_step_per_subject_means = {k: [] for k in per_step_latency_keys}
 
    # Scalar memory: list of one value per subject → mean ± std
    aggregate_memory_per_subject   = {k: [] for k in aggregate_memory_keys}
    
    # Per-step memory: list of per-subject MAX delta → mean ± std of maxes
    # Rationale: memory is governed by its peak, not its average.
    # Averaging signed deltas collapses to ~0 due to PyTorch allocator caching.
    per_step_memory_peak_per_subject = {k: [] for k in per_step_memory_keys}
 
    # Drift-aware update count per subject
    subject_frequency_savings = [] 
    subject_updates_drift = {}
    subject_total_update_opportunities = {}
    
    if not profiling_path.exists():
        print(f"Error: The path {profiling_path} does not exist.")
        return
    
    for subject_dir in profiling_path.iterdir():
        
        if subject_dir.is_dir() and subject_dir.name.startswith("subject_"):
            subject_id = subject_dir.name.split('_')[-1]
            
            profiling_json_path = subject_dir / "profiling" / f"{baseline}_profiling_report.json"
            
            with open(profiling_json_path, "r", encoding="utf-8") as f:
                prof = json.load(f)
            
            pred_json_path = subject_dir / baseline / "predictions_log.json"
            tgt_json_path = subject_dir / baseline / "targets_log.json"
            
            with open(pred_json_path, "r", encoding="utf-8") as f:
                baseline_outputs = json.load(f)
                
            with open(tgt_json_path, "r", encoding="utf-8") as f:
                baseline_targets = json.load(f)
            
            if False: 
                plot_blood_pressure_results(subject_id, baseline_outputs=baseline_outputs, baseline_targets=baseline_targets)
            
            # ---------------------------------------------------
            # Update frequency reduction statistics
            # ---------------------------------------------------
 
            did_adapt = prof['per_step']['did_adapt']
            total_possible_updates = len(did_adapt)
            actual_updates = did_adapt.count(True)
 
            saved_updates = total_possible_updates - actual_updates
            subject_saving = (
                saved_updates / total_possible_updates
                if total_possible_updates > 0 else 0.0
            )
 
            # Store per-subject statistics
            subject_frequency_savings.append(subject_saving)
            subject_updates_drift[subject_id] = actual_updates
            subject_total_update_opportunities[subject_id] = (
                total_possible_updates
            )
            
            # -------------------------------
            # Collect Latencies & Peak memory
            # -------------------------------
            
            # Latency
            for key in per_step_latency_keys:
                values = prof['per_step'][key]
                if len(values) > 0:
                    per_step_per_subject_means[key].append(np.mean(values))
 
            # Memory
            # Scalar memory collection
            for key in aggregate_memory_keys:
                aggregate_memory_per_subject[key].append(prof[key])
 
            # Per-step memory collection (peak per subject)
            # We take max() across steps because the worst-case allocation spike
            # is what determines whether the device runs out of RAM.
            # Small or negative deltas elsewhere are allocator noise, not savings.
            for key in per_step_memory_keys:
                values = prof['per_step'].get(key, [])
                if len(values) > 0:
                    per_step_memory_peak_per_subject[key].append(max(values))
 
    # Clinical Metrics Report
    aggregated_clinical_metrics_csv = pd.read_csv(
        os.path.join(
            profiling_path,
            "aggregate_metrics", 
            f"aggregate_{baseline}_metrics", 
            "evaluation_metrics.csv"
        )
    )
    print(f"\n--> Deployment ~ Clinical Metrics: ====")
    print(aggregated_clinical_metrics_csv)
    
    # Calculate the adaptation frequency reduction
    mean_frequency_saving = np.mean(subject_frequency_savings) * 100
    std_frequency_saving = np.std(subject_frequency_savings) * 100
 
    print(
        f"\n--> Deployment ~ Update Frequency Reduction:"
        f" {mean_frequency_saving:.1f}% ± "
        f"{std_frequency_saving:.1f}%"
    )
    
    # Latency
    print("\n--> Per-step Latency Summary (mean ± std across subjects):")
    for key in per_step_latency_keys:
        arr = np.array(per_step_per_subject_means[key])
        print(f"{key:45s}  {arr.mean()*1e3:7.2f} ± {arr.std()*1e3:6.2f} ms")
 
    # Scalar memory summary
    # Each value is already the right aggregate for the subject (a peak or
    # a fixed snapshot), so mean ± std across subjects is directly meaningful.
    print("\n--> Memory Summary — Aggregate (mean ± std across subjects):")
    labels = {
        'model_memory_mb':         'Model footprint (load cost)',
        'rss_before_tta_mb':       'RSS before TTA loop (fixed cost)',
        'peak_tta_rss_mb':         'Peak RSS during TTA loop (absolute)',
        'peak_incremental_tta_mb': 'Peak incremental TTA memory',
        'peak_process_rss_mb':     'Process-wide peak RSS (incl. imports)',
        'tm_peak_run_kb':          'Peak Python-heap usage over the entire run (kB)'
    }
    for key in aggregate_memory_keys:
        arr = np.array(aggregate_memory_per_subject[key])
        print(f"  {labels[key]:50s}  {arr.mean():7.2f} ± {arr.std():5.2f} MB")
 
    # Per-step memory summary
    # Reported as mean ± std of the per-subject MAX delta.
    # This answers: "what is the worst-case allocation spike for this section
    # across subjects?"
    print("\n--> Memory Summary — Per-step MAX delta (mean ± std across subjects):")
    print("    NOTE: each subject contributes its worst-case step delta;")
    print("    small values reflect allocator caching, not true zero cost.\n")
    step_mem_labels = {
        'tm_current_kb':            "Absolute Python-heap usage at the end of each step",
        'tm_head_adapt_kb':         "Delta during head adaptation",
        'tm_detector_reinit_kb':    "Delta during detector reinit"
    }
    for key in per_step_memory_keys:
        arr = np.array(per_step_memory_peak_per_subject.get(key, []))
        if arr.size > 0:
            print(f"  {step_mem_labels[key]:45s}  {arr.mean():7.2f} ± {arr.std():5.2f} MB")
    
    # --------
    # PLOTTING
    # --------      
 
    # Plot 1: Drift-aware updates per subject
    subject_ids = list(subject_updates_drift.keys())
 
    drift_updates = [
        subject_updates_drift[sid]
        for sid in subject_ids
    ]
 
    max_updates_per_subject = [
        subject_total_update_opportunities[sid]
        for sid in subject_ids
    ]
 
    x_ticks = np.arange(len(subject_ids))
 
    plt.figure(figsize=(12, 8))
 
    # Actual performed updates
    plt.bar(
        x_ticks,
        drift_updates,
        color="steelblue",
        width=0.7,
        label="Performed updates"
    )
 
    # Maximum possible updates
    plt.plot(
        x_ticks,
        max_updates_per_subject,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label="Maximum possible updates"
    )
 
    plt.xticks(x_ticks, [""] * len(x_ticks))
 
    plt.gca().yaxis.set_major_locator(
        plt.MaxNLocator(integer=True)
    )
 
    plt.xlabel("Subjects", fontsize=14)
    plt.ylabel("Number of updates", fontsize=14)
    plt.suptitle(
        f"Drift-aware adaptation frequency per subject\n"
        f"Average update reduction: "
        f"{mean_frequency_saving:.1f}% ± "
        f"{std_frequency_saving:.1f}%",
        fontsize=16
    )
 
    plt.legend(fontsize=12)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
 
    save_path = (
        f"./drift_updates_per_subject_on_"
        f"{deployment_device}.png"
    )
 
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Log Analysis] Saved → {save_path}")
 
    # Plot 2: Latency breakdown
    # Two-panel figure: top = per-step latencies, bottom = calibration
    # latencies. Horizontal grouped bars with mean ± std error bars.
    # Per-step and calibration are kept separate because they differ by
    # ~3 orders of magnitude and combining them would crush the per-step bars.
    #
    # Per-step labels and colors — ordered from cheapest to most expensive
    # so the reader's eye naturally moves from background cost to peak cost.
    per_step_display = {
        'prediction_s':             ('Prediction  ph(feats)',           '#4393c3'),
        'feature_extraction_s':     ('Feature extraction  enc(x)',      '#2166ac'),
        'drift_detection_s':        ('Drift detection  detector loop',  '#92c5de'),
        'head_adapt_s':             ('Head adaptation  train loop',     '#d6604d'),
        'drift_detector_reinit_s':  ('Detector reinit  reference copy', '#f4a582'),
        'adaptation_s':             ('Full adaptation block',           '#b2182b'),
        'total_step_s':             ('Total TTA step  (wall time)',     '#333333'),
    }
 
    fig, ax = plt.subplots(figsize=(12, 8))
 
    # Panel A: per-step latencies
    step_keys   = list(per_step_display.keys())
    step_labels = [per_step_display[k][0] for k in step_keys]
    step_colors = [per_step_display[k][1] for k in step_keys]
 
    step_means = np.array([
        np.array(per_step_per_subject_means[k]).mean() * 1e3   # → ms
        for k in step_keys
    ])
    step_stds = np.array([
        np.array(per_step_per_subject_means[k]).std() * 1e3
        for k in step_keys
    ])
 
    y_pos = np.arange(len(step_keys))
    bars  = ax.barh(
        y_pos, step_means,
        xerr=step_stds,
        color=step_colors,
        edgecolor='white',
        height=0.6,
        capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor='#555555')
    )
    # Annotate each bar with its value
    for bar, mean, std in zip(bars, step_means, step_stds):
        ax.text(
            bar.get_width() + std + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f} ± {std:.1f} ms",
            va='center', ha='left', fontsize=9, color='#333333'
        )
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(step_labels, fontsize=11)
    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.grid(axis='x', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    # Add extra x-axis margin so annotations don't clip
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
 
    fig.suptitle(f"Per-step latencies  (mean ± std across subjects) on {deployment_device}", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = f"./latency_profile_on_{deployment_device}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Log Analysis] Saved → {save_path}")
 
    # Plot 3: Memory breakdown
    model_arr       = np.array(aggregate_memory_per_subject['model_memory_mb'])
    incremental_arr = np.array(aggregate_memory_per_subject['peak_incremental_tta_mb'])
    rss_before_arr  = np.array(aggregate_memory_per_subject['rss_before_tta_mb'])
    
    runtime_arr     = rss_before_arr - model_arr
 
    subject_ids = list(subject_updates_drift.keys())
    n_subjects  = len(subject_ids)
    x = np.arange(n_subjects)
 
    per_step_mem_display = {
        'tm_current_kb':            ('Abs. Python-heap usage (step end)',  '#d6604d'),
        'tm_head_adapt_kb':         ('Delta during head adaptation',       '#4393c3'),
        'tm_detector_reinit_kb':    ('Delta during detector reinit',       "#5ca708"),
    }
 
    fig, (ax_stack, ax_step_mem) = plt.subplots(
        2, 1,
        figsize=(14, 11),
        gridspec_kw={'height_ratios': [3, len(per_step_mem_display)]}
    )
 
    # Panel A: stacked RSS per subject
    ax_stack.bar(x, runtime_arr,     label="Library / runtime baseline", color='lightgrey',  width=0.6)
    ax_stack.bar(x, model_arr,       label="Model footprint",             color='steelblue',  width=0.6,
                 bottom=runtime_arr)
    ax_stack.bar(x, incremental_arr, label="TTA incremental memory  [headline]",
                 color='darkorange', width=0.6,
                 bottom=runtime_arr + model_arr)
 
    # Mean reference lines
    ax_stack.axhline(
        (runtime_arr + model_arr).mean(),
        color='steelblue', linewidth=1.2, linestyle='--', alpha=0.7,
        label=f"Mean fixed cost  {(runtime_arr + model_arr).mean():.1f} MB"
    )
    ax_stack.axhline(
        (runtime_arr + model_arr + incremental_arr).mean(),
        color='darkorange', linewidth=1.2, linestyle='--', alpha=0.7,
        label=f"Mean total peak  {(runtime_arr + model_arr + incremental_arr).mean():.1f} MB"
    )
 
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels([""] * n_subjects)
    ax_stack.set_xlabel("Subjects", fontsize=12)
    ax_stack.set_ylabel("RSS Memory (MB)", fontsize=12)
    ax_stack.set_title("Absolute RSS breakdown per subject", fontsize=13)
    ax_stack.legend(fontsize=10, loc='upper left')
    ax_stack.grid(axis='y', alpha=0.35)
    ax_stack.spines[['top', 'right']].set_visible(False)
 
    # Panel B: per-step worst-case allocation spikes
    mem_keys   = list(per_step_mem_display.keys())
    mem_labels = [per_step_mem_display[k][0] for k in mem_keys]
    mem_colors = [per_step_mem_display[k][1] for k in mem_keys]
 
    mem_means = np.array([
        np.array(per_step_memory_peak_per_subject.get(k, [0])).mean()
        for k in mem_keys
    ])
    mem_stds = np.array([
        np.array(per_step_memory_peak_per_subject.get(k, [0])).std()
        for k in mem_keys
    ])
 
    y_pos_m = np.arange(len(mem_keys))
    bars_m  = ax_step_mem.barh(
        y_pos_m, mem_means,
        xerr=mem_stds,
        color=mem_colors,
        edgecolor='white',
        height=0.6,
        capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor='#555555')
    )
    for bar, mean, std in zip(bars_m, mem_means, mem_stds):
        ax_step_mem.text(
            bar.get_width() + std + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f} ± {std:.2f} MB",
            va='center', ha='left', fontsize=9, color='#333333'
        )
 
    ax_step_mem.set_yticks(y_pos_m)
    ax_step_mem.set_yticklabels(mem_labels, fontsize=11)
    ax_step_mem.set_xlabel("Memory delta (MB)", fontsize=12)
    ax_step_mem.set_title(
        "Per-step worst-case allocation spike  (max across steps, mean ± std across subjects)\n"
        "Note: values near zero reflect allocator caching, not true zero cost",
        fontsize=11
    )
    ax_step_mem.grid(axis='x', alpha=0.35)
    ax_step_mem.spines[['top', 'right']].set_visible(False)
    ax_step_mem.set_xlim(right=ax_step_mem.get_xlim()[1] * 1.4)
 
    fig.suptitle(f"Memory profile on {deployment_device}", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = f"./memory_profile_on_{deployment_device}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Log Analysis] Saved → {save_path}") 
    
           
# ---------------------
# LaTeX Table Generator
# ---------------------

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
                agg_df[col] = stacked.mean(axis=0)

            except:
                # Non numeric and not BHS (e.g. "Type") -> keep first
                pass

    return agg_df


def fmt(x):
    """
    Round numeric values to 1 decimals.
    """
    if isinstance(x, (int, float, np.floating)):
        return f"{x:.1f}"
    return str(x)

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

    latex = []
    latex.append(r"\begin{table*}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Personalization results (averaged over three seeds) on Vital DB for continuous SBP/DBP estimation from PPG, divided into the three subject sets 1/2/3. Parentheses indicate the target thresholds for the clinical standards for both SBP/DBP. CL algorithms are in light gray.}")
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

            if first_algo:
                row = f"        {set_id} "
                first_algo = False
            else:
                row = "        "

            if display_name == 'feat.replay':
                row += r"& \cellcolor{lightgray!30}\textbf{feat.replay} "
                row += (
                    r"& \textbf{" + f"{ae_sbp} / {ae_dbp}" + "} "
                    r"& \textbf{" + f"{bwt_sbp} / {bwt_dbp}" + "} "
                    r"& \textbf{" + f"{me_sbp} / {me_dbp}" + "} "
                    r"& \textbf{" + f"{std_sbp} / {std_dbp}" + "} "
                    r"& \textbf{" + f"{bhs_sbp} / {bhs_dbp}" + "} \\\\"
                )
            elif display_name in {'LwF', 'EWC', 'AGEM'}:
                row += r"& \cellcolor{lightgray!30}" + f"{display_name} "
                row += (
                    f"& {ae_sbp} / {ae_dbp} "
                    f"& {bwt_sbp} / {bwt_dbp} "
                    f"& {me_sbp} / {me_dbp} "
                    f"& {std_sbp} / {std_dbp} "
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            else:
                row += (
                    f"& {display_name} "
                    f"& {ae_sbp} / {ae_dbp} "
                    f"& {bwt_sbp} / {bwt_dbp} "
                    f"& {me_sbp} / {me_dbp} "
                    f"& {std_sbp} / {std_dbp} "
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            
                if display_name == 'online*':
                    row += r"\cline{2-7}"
                    
            latex.append(row)

        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_1}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


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
                    bhs_columns=["BHS_Grade"]   # adapt if column name differs
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
        "no_adapt": "no adapt",
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
    
    print(latex_table)

def generate_mmd_vs_lsdd_overleaf_table(
    aggregated_results, 
    set_mapping,
    baseline_key="feature_replay",
    ae_column="AE_mean",
    bwt_column="BWT_mean",
    me_column="ME",
    std_column="STD",
    bhs_column="BHS_Grade"
):
    latex = []
    latex.append(r"\begin{table*}")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Drift-aware Personalization results on Vital DB for continuous SBP/DBP estimation from PPG. The baseline employed for personalization is feature replay (reservoir buffer of 64 features) with the drift detectors considered in this study: MMD and LSDD. The reported metrics are the same as those of ~\ref{tab:table_1} plus the number of skipped updates (rightmost column) per drift detector.}")
    latex.append(r"    \begin{tabular}{l|l|l|p{0.15\textwidth}|p{0.16\textwidth}|l|l}")
    latex.append(r"        \hline")
    latex.append(r"        \textbf{Algorithm} & \textbf{AE}$\downarrow$ & \textbf{BWT}$\downarrow$ & \makecell[l]{\textbf{ME}$\downarrow$ \\ ($<$5 mmHg)} & \makecell[l]{\textbf{STD}$\downarrow$ \\ ($<$8 mmHg)} & \makecell[l]{\textbf{BHS}$\uparrow$ \\ (A)} & \makecell[l]{\textbf{Skipped} \\ \textbf{Fraction\%}$\uparrow$} \\")
    latex.append(r"        \hline")

    for algo, shift_name in set_mapping.items():

        # ---------------------------
        # Retrieve aggregated metrics
        # ---------------------------

        clinical_df = aggregated_results[baseline_key][f"{shift_name}_clinical"]
        sbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_sbp_cl"]
        dbp_cl_df = aggregated_results[baseline_key][f"{shift_name}_dbp_cl"]
        skipped_fraction = aggregated_results[baseline_key][f"{shift_name}_updates"]["mean_skipped_fraction"]

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
        
        skipped_fraction = fmt(skipped_fraction * 100)

        bhs_sbp = clinical_df.iloc[0][bhs_column]
        bhs_dbp = clinical_df.iloc[1][bhs_column]

        # ---------
        # Build row
        # ---------
        
        row = (
            f"{algo} "
            f"& {ae_sbp} / {ae_dbp} "
            f"& {bwt_sbp} / {bwt_dbp} "
            f"& {me_sbp} / {me_dbp} "
            f"& {std_sbp} / {std_dbp} "
            f"& {bhs_sbp} / {bhs_dbp} "
            f"& {skipped_fraction} \\\\"
        )
    
        latex.append(row)

        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_2}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def analyze_mmd_vs_lsdd(baselines):
    
    baseline = 'feature_replay'
    path_keys = ["drift_aware_mmd", "drift_aware_lsdd"]
    
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
            
            # TODO: could expand with seeds
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

            # TODO: could expand with seeds
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

            # TODO: could expand with seeds
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
            
            # TODO: could expand with seeds
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
        "MMD": "drift_aware_mmd",
        "LSDD": "drift_aware_lsdd",
    }
    
    latex_table = generate_mmd_vs_lsdd_overleaf_table(
        aggregated_results,
        set_mapping
    )
    
    print(latex_table)
    
            
def analyze_logs_and_plot():
    
    print(f"[Log Analysis] Analyzing logs ...")
    
    # Baselines and corresponding experiment folder
    # Collect results on gradual shifts/mixed/shifts/abrupt shifts for each baseline
    # -> experiment path is hardcoded, bad
    baselines = {
        'no_adapt': {
            'gradual_shifts_path': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-18_31_51",
            'gradual_shifts_path_seed_41': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-18_45_02",
            'gradual_shifts_path_seed_40': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_no_adapt_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-18_44_55",
            'mixed_shifts_path': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-18_36_29",
            'mixed_shifts_path_seed_41': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_17-18_45_36",
            'mixed_shifts_path_seed_40': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_no_adapt_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_17-18_45_29",
            'abrupt_shifts_path': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-18_34_54", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_17-18_46_04",
            'abrupt_shifts_path_seed_40': "./logs/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_no_adapt_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_17-18_45_57",
        }, 
        'first_batch_finetune': {
            'gradual_shifts_path': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_18-09_59_24",
            'gradual_shifts_path_seed_41': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_18-10_18_02",
            'gradual_shifts_path_seed_40': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_first_batch_finetune_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_18-11_33_55",
            'mixed_shifts_path': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-09_59_11",
            'mixed_shifts_path_seed_41': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_18-10_36_06",
            'mixed_shifts_path_seed_40': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_first_batch_finetune_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_18-11_34_19",
            'abrupt_shifts_path': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_18-09_59_39", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_18-10_22_57",
            'abrupt_shifts_path_seed_40': "./logs/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_first_batch_finetune_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_18-10_43_39",
        }, 
        'online': { 
            'gradual_shifts_path': "./logs/personalization_online_proto_ppg_calibration_size_1_gradual_shifts/personalization_online_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-19_07_08",
            'gradual_shifts_path_seed_41': "./logs/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-19_32_27",
            'gradual_shifts_path_seed_40': "./logs/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_online_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-19_32_27",
            'mixed_shifts_path': "./logs/personalization_online_proto_ppg_calibration_size_1_mixed_shifts/personalization_online_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-20_04_21",
            'mixed_shifts_path_seed_41': "./logs/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_17-20_19_48",
            'mixed_shifts_path_seed_40': "./logs/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_online_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_17-20_19_48",
            'abrupt_shifts_path': "./logs/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-19_28_05",
            'abrupt_shifts_path_seed_41': "./logs/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_17-19_43_03",
            'abrupt_shifts_path_seed_40': "./logs/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_online_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_17-19_43_03",
        }, 
        'online_from_scratch': {
            'gradual_shifts_path': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-19_54_16",
            'gradual_shifts_path_seed_41': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-20_18_59",
            'gradual_shifts_path_seed_40': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_online_from_scratch_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-20_18_52",
            'mixed_shifts_path': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-21_42_06",
            'mixed_shifts_path_seed_41': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_17-21_55_56",
            'mixed_shifts_path_seed_40': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_online_from_scratch_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_17-21_56_06",
            'abrupt_shifts_path': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-20_27_07", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_17-20_39_56",
            'abrupt_shifts_path_seed_40': "./logs/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_online_from_scratch_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_17-20_40_09",
        }, 
        'feature_replay': {
            'gradual_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-20_41_18",
            'gradual_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-21_05_45",
            'gradual_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-21_05_24",
            'mixed_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-23_19_51",
            'mixed_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_17-23_31_04",
            'mixed_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_17-23_31_47",
            'abrupt_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-21_27_56", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_17-21_37_24",
            'abrupt_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_17-21_37_24",
            "drift_aware_mmd": "",
            "drift_aware_lsdd": "",
            "laptop_drift_aware_mmd": "./logs/laptop_deployment_feature_replay_drift_aware_mmd/laptop_deployment_feature_replay_drift_aware_mmd-Proto-2026_05_19-17_36_17",
            "laptop_drift_aware_lsdd": "./logs/laptop_deployment_feature_replay_drift_aware_lsdd/laptop_deployment_feature_replay_drift_aware_lsdd-Proto-2026_05_19-17_31_26",
            "pi_drift_aware_mmd" : "./logs/pi_deployment_feature_replay_drift_aware_mmd/pi_deployment_feature_replay_drift_aware_mmd-Proto-2026_05_19-13_17_50",
            "pi_drift_aware_lsdd" : "./logs/pi_deployment_feature_replay_drift_aware_lsdd/pi_deployment_feature_replay_drift_aware_lsdd-Proto-2026_05_19-15_52_49",
            "pixel_drift_aware_mmd" : "./logs/pixel_deployment_feature_replay_drift_aware_mmd/pixel_deployment_feature_replay_drift_aware_mmd-Proto-2026_05_19-13_25_41",
            "pixel_drift_aware_lsdd" : "./logs/pixel_deployment_feature_replay_drift_aware_lsdd/pixel_deployment_feature_replay_drift_aware_lsdd-Proto-2026_05_19-17_21_19",
            "ppg_ecg_gradual_shifts" : "./logs/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts-Proto-2026_05_18-15_38_22",
        }, 
        'lwf': {
            'gradual_shifts_path': "./logs/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-21_34_18",
            'gradual_shifts_path_seed_41': "./logs/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-21_56_26",
            'gradual_shifts_path_seed_40': "./logs/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_lwf_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-21_56_28",
            'mixed_shifts_path': "./logs/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-00_51_12",
            'mixed_shifts_path_seed_41': "./logs/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_18-00_57_39",
            'mixed_shifts_path_seed_40': "./logs/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_lwf_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_18-00_57_54",
            'abrupt_shifts_path': "./logs/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-22_32_36", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_17-22_38_38",
            'abrupt_shifts_path_seed_40': "./logs/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_lwf_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_17-22_38_36",
        }, 
        'ewc': {
            'gradual_shifts_path': "./logs/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-22_21_27",
            'gradual_shifts_path_seed_41': "./logs/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-22_43_05",
            'gradual_shifts_path_seed_40': "./logs/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_ewc_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-22_42_45",
            'mixed_shifts_path': "./logs/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-01_47_44",
            'mixed_shifts_path_seed_41': "./logs/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_18-01_50_53",
            'mixed_shifts_path_seed_40': "./logs/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_ewc_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_18-01_50_56",
            'abrupt_shifts_path': "./logs/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-23_33_23", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_17-23_35_35",
            'abrupt_shifts_path_seed_40': "./logs/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_ewc_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_17-23_35_15",
        }, 
        'agem': {
            'gradual_shifts_path': "./logs/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-23_10_55",
            'gradual_shifts_path_seed_41': "./logs/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_17-23_32_07",
            'gradual_shifts_path_seed_40': "./logs/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_agem_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_17-23_31_45",
            'mixed_shifts_path': "./logs/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_18-02_36_22",
            'mixed_shifts_path_seed_41': "./logs/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_18-02_38_29",
            'mixed_shifts_path_seed_40': "./logs/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_agem_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_18-02_38_29",
            'abrupt_shifts_path': "./logs/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_18-00_30_19", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_18-00_29_42",
            'abrupt_shifts_path_seed_40': "./logs/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_agem_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_18-00_29_36",
        },
    }
    
    # -------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> all the baselines
    # -------------------------------------------------
    analyze_gradual_vs_mixed_vs_abrupt(baselines)
    
    # ---------------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> only for feature replay with Online MMD vs Online LSDD
    # ---------------------------------------------------------
    #analyze_mmd_vs_lsdd(baselines)
    
    # --------------------------
    # Resource Profiling
    # -> only for feature replay
    # --------------------------
    
    baseline = 'feature_replay'
    
    # Laptop Profiling
    print("[Log Analysis] Laptop Resource Profiling")
    resource_profiling(
        baseline=baseline, 
        mmd_profiling_path=baselines[baseline]['laptop_drift_aware_mmd'], 
        lsdd_profiling_path=baselines[baseline]['laptop_drift_aware_lsdd'],
        deployment_device='laptop'
    )

    # Pixel Profiling
    print("[Log Analysis] Google Pixel Resource Profiling")
    resource_profiling(
        baseline=baseline, 
        mmd_profiling_path=baselines[baseline]['pixel_drift_aware_mmd'], 
        lsdd_profiling_path=baselines[baseline]['pixel_drift_aware_lsdd'],
        deployment_device='pixel'
    )
    
    
    # Pi Profiling
    print("[Log Analysis] Raspberry Pi Resource Profiling")
    resource_profiling(
        baseline=baseline, 
        mmd_profiling_path=baselines[baseline]['pi_drift_aware_mmd'], 
        lsdd_profiling_path=baselines[baseline]['pi_drift_aware_lsdd'],
        deployment_device='pixel'
    )
    
    print("[Log Analysis] Results Analysis Completed ✓")
    
    
def parseargs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_folder_path', default='', type=str, help='path to the logs folder with the experiment to analyze')
    parser.add_argument('--config_yaml_path', default='', type=str, help='path to the configuration YAML file for the experiments')
    parser.add_argument('--fig_root', default='', type=str, help='path to figure folder where to store the result analysis outputs')
    parser.add_argument('--exp_fig_root', default='', type=str, help='path to figure folder of each subject analyzed in an experiment (subfolder inside fig_root)')
    parser.add_argument('--exp_pi_deployment_drift_aware_fig_root', default='', type=str, help='deployment on Pi drift-aware experiment folder')

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    
    # Collect metrics from the log file folder
    analyze_logs_and_plot()
    
    
        
