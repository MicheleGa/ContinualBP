
import os
from pathlib import Path
import sys
folders_to_add = ['data']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from data.preprocessing_utils.data_visualization import plot_drift_aware_updates_per_subject, plot_latency_breakdown, plot_memory_breakdown


# -------------------
# Metrics Aggregation
# -------------------

# ─────────────────────────────────────────────────────────────────────────────
# Shared key / label definitions
# ─────────────────────────────────────────────────────────────────────────────

_PER_STEP_LATENCY_KEYS = [
    'feature_extraction_s',     # enc(signals) latency
    'prediction_s',             # pred(feats) latency
    'drift_detection_s',        # drift_detector.predict() loop latency
    'adaptation_s',             # entire do_adapt block latency
    'head_adapt_s',             # prediction head adapt latency
    'drift_detector_reinit_s',  # drift detector re-init latency
    'total_step_s',             # wall time for the full TTA step
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


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

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
            "default": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path"]),
            "seed_40": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path_seed_40"]),
            "seed_41": Path(baseline_paths[f"{deployment_device}_drift_aware_mmd_path_seed_41"]),
        }
    }

    results = {}
    for detector_name, seed_paths in profiling_paths.items():
        results[detector_name] = _run_profiling_report(
            baseline, seed_paths, detector_name, deployment_device
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed data collection
# ─────────────────────────────────────────────────────────────────────────────

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
        print(f"[{detector_name}] Warning: path {profiling_path} does not exist — skipping.")
        return None

    per_step_per_subject_means       = {k: [] for k in _PER_STEP_LATENCY_KEYS}
    aggregate_memory_per_subject     = {k: [] for k in _AGGREGATE_MEMORY_KEYS}
    per_step_memory_peak_per_subject = {k: [] for k in _PER_STEP_MEMORY_KEYS}
    subject_frequency_savings        = []
    subject_updates_drift            = {}
    subject_total_update_opportunities = {}

    for subject_dir in profiling_path.iterdir():
        if not (subject_dir.is_dir() and subject_dir.name.startswith("subject_")):
            continue

        subject_id = subject_dir.name.split('_')[-1]

        profiling_json_path = subject_dir / "profiling" / f"{baseline}_profiling_report.json"
        with open(profiling_json_path, "r", encoding="utf-8") as f:
            prof = json.load(f)

        # ── Update frequency reduction ────────────────────────────
        did_adapt         = prof['per_step']['did_adapt']
        total_possible    = len(did_adapt)
        actual_updates    = did_adapt.count(True)
        saved             = total_possible - actual_updates
        subject_frequency_savings.append(
            saved / total_possible if total_possible > 0 else 0.0
        )
        subject_updates_drift[subject_id]             = actual_updates
        subject_total_update_opportunities[subject_id] = total_possible

        # ── Latency (mean per subject) ────────────────────────────
        for key in _PER_STEP_LATENCY_KEYS:
            values = prof['per_step'][key]
            if values:
                per_step_per_subject_means[key].append(np.mean(values))

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
        "aggregate_memory":    aggregate_memory_per_subject,
        "per_step_memory":     per_step_memory_peak_per_subject,
        "frequency_savings":   subject_frequency_savings,
        "updates_drift":       subject_updates_drift,
        "total_opportunities": subject_total_update_opportunities,
        "clinical_metrics":    clinical_metrics_csv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-seed averaging
# ─────────────────────────────────────────────────────────────────────────────

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

    latency_pooled   = _pool_subjects(seed_data_list, "per_step_latency", _PER_STEP_LATENCY_KEYS)
    agg_mem_pooled   = _pool_subjects(seed_data_list, "aggregate_memory",  _AGGREGATE_MEMORY_KEYS)
    step_mem_pooled  = _pool_subjects(seed_data_list, "per_step_memory",   _PER_STEP_MEMORY_KEYS)

    # ── Frequency savings: pool raw per-subject values across seeds ──────────
    # FIX 1: was np.mean per seed → (n_seeds,); now all values → (n_seeds * n_subjects,)
    freq_pooled = np.array([
        v for d in seed_data_list for v in d["frequency_savings"]
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

    # total_opportunities is determined by the data stream, not the seed
    avg_total_opportunities = seed_data_list[0]["total_opportunities"]

    # ── Clinical metrics: concatenate all subjects across seeds ──────────────
    # FIX 2: was element-wise mean → 88 averaged rows (Approach B);
    # now row-concatenation → 264 real subject observations (Approach A)
    pooled_clinical = pd.concat(
        [d["clinical_metrics"] for d in seed_data_list],
        ignore_index=True,
    )

    return {
        "latency_pooled":          latency_pooled,
        "agg_mem_pooled":          agg_mem_pooled,
        "step_mem_pooled":         step_mem_pooled,
        "freq_pooled":             freq_pooled,
        "avg_updates_drift":       avg_updates_drift,
        "avg_total_opportunities": avg_total_opportunities,
        "pooled_clinical_metrics": pooled_clinical,
        "n_seeds":                 len(seed_data_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report orchestrator
# ─────────────────────────────────────────────────────────────────────────────

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
    avg     = _average_seed_results(seed_data_list)
    n_seeds = avg["n_seeds"]
    # Suffix updated: pooled population is the unit, not seeds
    suffix  = f"(mean ± std over subjects, {n_seeds} seed(s) pooled)"

    # ── Clinical metrics ──────────────────────────────────────────
    # Pooled DataFrame has n_seeds * n_subjects rows — summarise column-wise
    pooled_clinical = avg["pooled_clinical_metrics"]
    numeric_cols    = pooled_clinical.select_dtypes(include=[np.number]).columns
    clinical_summary = pooled_clinical[numeric_cols].agg(["mean", "std"])
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

    # ── Per-step latency ──────────────────────────────────────────
    print(f"\n[{detector_name}] Per-step Latency Summary {suffix}:")
    latency_stats = {}
    for key in _PER_STEP_LATENCY_KEYS:
        arr = avg["latency_pooled"][key]          # pooled over all subjects
        m, s = float(arr.mean()) * 1e3, float(arr.std()) * 1e3
        latency_stats[key] = (m, s)
        print(f"  {key:45s}  {m:7.2f} ± {s:6.2f} ms")

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
    plot_drift_aware_updates_per_subject(
        subject_updates_drift=avg["avg_updates_drift"],
        subject_total_update_opportunities=avg["avg_total_opportunities"],
        mean_frequency_saving=mean_freq,
        std_frequency_saving=std_freq,
        deployment_device=deployment_device,
        detector_name=detector_name,
        n_seeds=n_seeds,
    )

    plot_latency_breakdown(
        per_step_per_subject_means=avg["latency_pooled"],  # key rename only
        deployment_device=deployment_device,
        detector_name=detector_name,
        n_seeds=n_seeds,
    )
    
    # ── Return results ────────────────────────────────────────────
    return {
        #"clinical_metrics":        clinical_summary,        # column-wise summary
        "frequency_saving_mean":   mean_freq,
        "frequency_saving_std":    std_freq,
        "latency_ms":              latency_stats,
        "aggregate_memory_mb":     agg_mem_stats,
        "update_freq_reduction":   (mean_freq, std_freq),
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


def analyze_mmd_vs_lsdd(baselines):
    
    baseline = 'feature_replay'
    path_keys = ["drift_aware_mmd_path", "drift_aware_lsdd_path"]
    
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
        "MMD": "drift_aware_mmd_path",
        "LSDD": "drift_aware_lsdd_path",
    }
    
    latex_table = generate_mmd_vs_lsdd_overleaf_table(
        aggregated_results,
        set_mapping
    )
    
    print()
    print(latex_table)
    print()
    


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
    
    print()
    print(latex_table)
    print()
    
    


def analyze_ppg_vs_ppg_ecg(baselines):
    
    baseline = 'feature_replay'
    path_keys = ["gradual_shifts_path", "ppg_ecg_gradual_shifts_path"]
    
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

            # TODO: could expand with seeds
        ]

        aggregated_results[baseline][f"{path_key}_sbp_cl"] = (
            aggregate_seed_dataframes(sbp_dfs)
        )

        aggregated_results[baseline][f"{path_key}_dbp_cl"] = (
            aggregate_seed_dataframes(dbp_dfs)
        )
        
    set_mapping = {
        "PPG": "gradual_shifts_path",
        "PPG+ECG": "ppg_ecg_gradual_shifts_path",
    }
    
    latex_table = generate_ppg_vs_ppg_ecg_overleaf_table(
        aggregated_results,
        set_mapping
    )
    
    print(latex_table)
    
def estimate_communication_costs(setting):
    def fmt(value, decimals=1):
        return rf"${value:.{decimals}f}$"
    
    # One sample memory
    input_data_length_s = 10
    input_data_freq = 125
    sample_memory = input_data_length_s * input_data_freq
    
    input_data_batch = 4
    batch_memory = sample_memory * input_data_batch
    
    number_of_updates = 72
    memory_transmitted_for_all_updates = batch_memory * number_of_updates
    
    # FP 32 precision
    feature_extractor_fp_params = 193384
    prediction_head_fp_params = 8451
    
    # --- Calculation Logic ---
    # Total parameters in the model
    total_params = feature_extractor_fp_params + prediction_head_fp_params
    
    # 1 FP32 value = 4 bytes. 
    # Convert total bytes to Megabytes (MB) using 1 MB = 1,024 * 1,024 bytes
    model_memory_bytes = total_params * 4
    model_memory_mb = model_memory_bytes / (1024 * 1024)
    
    # Input data values are typically integers or floats. Assuming standard 4-byte (FP32) float 
    # values for the transmitted sensor/input data:
    data_memory_bytes = memory_transmitted_for_all_updates * 4
    data_memory_mb = data_memory_bytes / (1024 * 1024)
    
    if setting == "A":
        # Return the MB of the model (feat.ext. + head)
        return fmt(model_memory_mb)
    elif setting == 'B':  # Added missing colon
        # Return the MB of the model (feat.ext. + head) + memory transmitted for all updates 
        return fmt(model_memory_mb + data_memory_mb)
    else:
        raise ValueError("Incorrect setting was passed as input argument")
    
        
    
# ---------------------
# LaTeX Table Generator
# ---------------------
def generate_resource_profile_table(pi_profile, pixel_profile):

    method = 'MMD'
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
        (r'adaptation (head adapt. + detector reint.)', 'adaptation_s',               1),
        (r'head adapt.',                                'head_adapt_s',               2),  # ← sub-component
        (r'detector reinit.',                           'drift_detector_reinit_s',    2),  # ← sub-component
        (r'total',                                      'total_step_s',               1),
    ]

    # ── Memory labels & keys ───────────────────────────────────────────────
    memory_labels = {
        #r'model\_memory\_mb':          'model_memory_mb',
        #r'rss\_before\_tta\_mb':       'rss_before_tta_mb',
        #r'peak\_tta\_rss\_mb':         'peak_tta_rss_mb',
        #r'peak\_incremental\_tta\_mb': 'peak_incremental_tta_mb',
        r'peak process memory':         'peak_process_rss_mb',
        #r'tm\_peak\_run\_kb':          'tm_peak_run_kb',
    }

    INDENT = {1: r'\quad', 2: r'\qquad'}

    rows = []

    # ── Header ─────────────────────────────────────────────────────────────
    rows.append(r"        \hline")
    rows.append(r"        \textbf{Metric} & \textbf{Raspberry Pi 5} & \textbf{Google Pixel 10a} \\")
    rows.append(r"        \hline")

    # ── Communication block ──────────────────────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Estimated Communication Requirements (MB)}} \\")
    rows.append(r"        \hline")
    rows.append(
        rf"        {INDENT[1]} Setting A & {estimate_communication_costs('A')} & {estimate_communication_costs('A')} \\"
    )
    rows.append(
        rf"        {INDENT[1]} Setting B & {estimate_communication_costs('B')} & {estimate_communication_costs('B')} \\"
    )
    rows.append(r"        \hline")
    
    # ── Latency block ──────────────────────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Computational Requirements $\sim$ Avg. Latency (ms)}} \\")
    rows.append(r"        \hline")
    for label, key, level in latency_rows:
        pi_m, pi_s = pi['latency_ms'][key]
        px_m, px_s = px['latency_ms'][key]
        indent = INDENT[level]
        rows.append(
            rf"        {indent} {label} & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
        )
    rows.append(r"        \hline")

    # ── Memory block ───────────────────────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Memory Requirements $\sim$ Avg. Memory (MB)}} \\")
    rows.append(r"        \hline")
    for label, key in memory_labels.items():
        pi_m, pi_s = pi['aggregate_memory_mb'][key]
        px_m, px_s = px['aggregate_memory_mb'][key]
        rows.append(
            rf"        \quad {label} & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
        )
    rows.append(r"        \hline")

    # ── Update-frequency-reduction row ─────────────────────────────────────
    rows.append(r"        \multicolumn{3}{l}{\textit{Profiled Avg.\ Update Frequency Reduction (\%)}} \\")
    rows.append(r"        \hline")
    pi_m, pi_s = pi['update_freq_reduction']
    px_m, px_s = px['update_freq_reduction']
    rows.append(
        rf"        \quad freq.\ reduction & {fmt(pi_m, pi_s)} & {fmt(px_m, px_s)} \\"
    )
    rows.append(r"        \hline")

    # ── Assemble full table ────────────────────────────────────────────────
    table = "\n".join([
        r"\begin{table*}",
        r"    \centering",
        r"    \caption{Resource profile comparison across different edge devices (\( \mu \) $\pm$ \( \sigma \)) of the feature replay CL algorithm with MMD drift detector adapting to gradual shifts (72 time steps).}",
        r"    \begin{tabular}{l|c|c}",
        "\n".join(rows),
        r"    \end{tabular}",
        r"    \label{tab:table_3}",
        r"\end{table*}",
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
    latex.append(r"    \caption{Personalization results (averaged over three seeds) on Vital DB with deployment on the laptop GPU for continuous SBP/DBP estimation from PPG, divided into the three subject sets 1/2/3. Parentheses indicate the target thresholds for the clinical standards for both SBP/DBP. CL algorithms are in light gray.}")
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
                    r"& \textbf{" + f"{ae_sbp_mean} / {ae_dbp_mean}" + "} "
                    r"& \textbf{" + f"{bwt_sbp_mean} / {bwt_dbp_mean}" + "} "
                    r"& \textbf{" + f"{me_sbp_mean} / {me_dbp_mean}" + "} "
                    r"& \textbf{" + f"{std_sbp_mean} / {std_dbp_mean}" + "} "
                    r"& \textbf{" + f"{bhs_sbp} / {bhs_dbp}" + "} \\\\"
                )
            elif display_name in {'LwF', 'EWC', 'AGEM'}:
                row += r"& \cellcolor{lightgray!30}" + f"{display_name} "
                row += (
                    f"& {ae_sbp_mean} / {ae_dbp_mean} "
                    f"& {bwt_sbp_mean} / {bwt_dbp_mean} "
                    f"& {me_sbp_mean} / {me_dbp_mean} "
                    f"& {std_sbp_mean} / {std_dbp_mean} "
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            else:
                row += (
                    f"& {display_name} "
                    f"& {ae_sbp_mean} / {ae_dbp_mean} "
                    f"& {bwt_sbp_mean} / {bwt_dbp_mean} "
                    f"& {me_sbp_mean} / {me_dbp_mean} "
                    f"& {std_sbp_mean} / {std_dbp_mean} "
                    f"& {bhs_sbp} / {bhs_dbp} \\\\"
                )
            
                if display_name == 'online*':
                    row += r"\cline{2-7}"
                    
            """
            if display_name == 'feat.replay':
                row += r"& \cellcolor{lightgray!30}\textbf{feat.replay} "
                row += (
                    r"& \textbf{" + f"{ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f"/{ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f"/{bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f"/{me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f"/{std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "}" + "} "
                    r"& \textbf{" + f"{bhs_sbp}/{bhs_dbp}" + "} \\\\"
                )
            elif display_name in {'LwF', 'EWC', 'AGEM'}:
                row += r"& \cellcolor{lightgray!30}" + f"{display_name} "
                row += (
                    f"& {ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f"/{ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "} "
                    f"& {bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f"/{bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "} "
                    f"& {me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f"/{me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "} "
                    f"& {std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f"/{std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "} "
                    f"& {bhs_sbp}/{bhs_dbp} \\\\"
                )
            else:
                row += (
                    f"& {display_name} "
                    f"& {ae_sbp_mean}" + r"{\tiny $\pm$" + f"{ae_sbp_std}" + "}" + f"/{ae_dbp_mean}" + r"{\tiny $\pm$" + f"{ae_dbp_std}" + "} "
                    f"& {bwt_sbp_mean}" + r"{\tiny $\pm$" + f"{bwt_sbp_std}" + "}" + f"/{bwt_dbp_mean}" + r"{\tiny $\pm$" + f"{bwt_dbp_std}" + "} "
                    f"& {me_sbp_mean}" + r"{\tiny $\pm$" + f"{me_sbp_std}" + "}" + f"/{me_dbp_mean}" + r"{\tiny $\pm$" + f"{me_dbp_std}" + "} "
                    f"& {std_sbp_mean}" + r"{\tiny $\pm$" + f"{std_sbp_std}" + "}" + f"/{std_dbp_mean}" + r"{\tiny $\pm$" + f"{std_dbp_std}" + "} "
                    f"& {bhs_sbp}/{bhs_dbp} \\\\"
                )
            """
               
            latex.append(row)

        latex.append(r"        \hline")

    latex.append(r"    \end{tabular}")
    latex.append(r"    \label{tab:table_1}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


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
    latex.append(r"    \caption{Drift-aware Personalization results (averaged over three seeds) on Vital DB with deployment on the laptop CPU for continuous SBP/DBP estimation from PPG. The baseline employed for personalization is feature replay, adapting to gradual shifts with the MMD/LSDD drift detectors. The reported metrics are the same as those of Table ~\ref{tab:table_1} plus the number of skipped updates (rightmost column) per drift detector, relatively to the number of updates without drift detection (72).}")
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
        if algo == "MMD":
            row = (
                r"\textbf{MMD} "
                r"& \textbf{" + f"{ae_sbp} / {ae_dbp}" + "} "
                r"& \textbf{" + f"{bwt_sbp} / {bwt_dbp}" + "} "
                r"& \textbf{" + f"{me_sbp} / {me_dbp}" + "} "
                r"& \textbf{" + f"{std_sbp} / {std_dbp}" + "} "
                r"& \textbf{" + f"{bhs_sbp} / {bhs_dbp}" + "}"
                r"& \textbf{" + f"{skipped_fraction}" + "} \\\\"
            )
        else:
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


def analyze_logs_and_plot():
    
    print(f"[Log Analysis] Analyzing logs ...")
    
    # Baselines and corresponding experiment folder
    # Collect results on gradual shifts/mixed/shifts/abrupt shifts for each baseline
    # -> experiment path is hardcoded, bad
    baselines = {
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
            'gradual_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts-Proto-2026_05_17-20_41_18",
            'gradual_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_41-Proto-2026_05_21-12_24_09",
            'gradual_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_gradual_shifts_seed_40-Proto-2026_05_21-12_24_13",
            'mixed_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts-Proto-2026_05_17-23_19_51",
            'mixed_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_41-Proto-2026_05_21-14_19_00",
            'mixed_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_mixed_shifts_seed_40-Proto-2026_05_21-14_19_00",
            'abrupt_shifts_path': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts-Proto-2026_05_17-21_27_56", 
            'abrupt_shifts_path_seed_41': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_41-Proto-2026_05_21-12_48_51",
            'abrupt_shifts_path_seed_40': "./logs/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_40/personalization_feature_replay_proto_ppg_calibration_size_1_abrupt_shifts_seed_40-Proto-2026_05_21-12_48_51",
            "drift_aware_mmd_path": "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd/personalization_feature_replay_proto_ppg_drift_aware_mmd-Proto-2026_05_20-10_35_46",
            "drift_aware_mmd_path_seed_41": "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_41/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_41-Proto-2026_05_21-13_40_45",
            "drift_aware_mmd_path_seed_40": "./logs/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_40/personalization_feature_replay_proto_ppg_drift_aware_mmd_seed_40-Proto-2026_05_21-12_38_45",
            "drift_aware_lsdd_path": "./logs/personalization_feature_replay_proto_ppg_drift_aware_lsdd/personalization_feature_replay_proto_ppg_drift_aware_lsdd-Proto-2026_05_20-10_35_50",
            "drift_aware_lsdd_path_seed_41": "./logs/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_41/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_41-Proto-2026_05_21-15_28_56",
            "drift_aware_lsdd_path_seed_40": "./logs/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_40/personalization_feature_replay_proto_ppg_drift_aware_lsdd_seed_40-Proto-2026_05_21-14_42_54",
            "pi_drift_aware_mmd_path" : "./logs/pi_deployment_feature_replay_drift_aware_mmd/pi_deployment_feature_replay_drift_aware_mmd-Proto-2026_05_20-10_41_16",
            "pi_drift_aware_mmd_path_seed_41" : "./logs/pi_deployment_feature_replay_drift_aware_mmd_seed_41/pi_deployment_feature_replay_drift_aware_mmd_seed_41-Proto-2026_05_21-11_15_08",
            "pi_drift_aware_mmd_path_seed_40" : "./logs/pi_deployment_feature_replay_drift_aware_mmd_seed_40/pi_deployment_feature_replay_drift_aware_mmd_seed_40-Proto-2026_05_21-10_16_06",
            "pixel_drift_aware_mmd_path" : "./logs/pixel_deployment_feature_replay_drift_aware_mmd/pixel_deployment_feature_replay_drift_aware_mmd-Proto-2026_05_20-10_41_36",
            "pixel_drift_aware_mmd_path_seed_41" : "./logs/pixel_deployment_feature_replay_drift_aware_mmd_seed_41/pixel_deployment_feature_replay_drift_aware_mmd_seed_41-Proto-2026_05_21-13_26_30",
            "pixel_drift_aware_mmd_path_seed_40" : "./logs/pixel_deployment_feature_replay_drift_aware_mmd_seed_40/pixel_deployment_feature_replay_drift_aware_mmd_seed_40-Proto-2026_05_21-10_16_37",
            "ppg_ecg_gradual_shifts_path" : "./logs/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts/personalization_feature_replay_proto_ppg_ecg_calibration_size_1_gradual_shifts-Proto-2026_05_18-15_38_22",
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
    
    # -------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> all the baselines
    # -------------------------------------------------
    analyze_gradual_vs_mixed_vs_abrupt(baselines)
    
    # ---------------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> only for feature replay with MMD vs LSDD
    # ---------------------------------------------------------
    analyze_mmd_vs_lsdd(baselines)
    
    # ---------------------------------------------------------
    # Performance Assessment with Clinical & CL Metrics
    # -> only for feature replay with PPG vs PPG + ECG
    # ---------------------------------------------------------
    analyze_ppg_vs_ppg_ecg(baselines)
    
    # --------------------------
    # Resource Profiling
    # -> only for feature replay
    # --------------------------
    
    baseline = 'feature_replay'
    
    # Laptop Profiling
    #print("[Log Analysis] Laptop Resource Profiling")
    #resource_profiling(
    #    baseline=baseline, 
    #    baselines=baselines, 
    #    deployment_device='laptop'
    #)
    
    # Pi Profiling
    print("[Log Analysis] Raspberry Pi Resource Profiling")
    pi_profile = resource_profiling(
        baseline=baseline, 
        baselines=baselines,
        deployment_device='pi'
    )
    
    # Pixel Profiling
    print("[Log Analysis] Google Pixel Resource Profiling")
    pixel_profile = resource_profiling(
        baseline=baseline, 
        baselines=baselines,
        deployment_device='pixel'
    )
    
    generate_resource_profile_table(pi_profile, pixel_profile)
        
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
    
    
        
