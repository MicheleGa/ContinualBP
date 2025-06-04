import os
import argparse
from distutils.util import strtobool
import lmdb
import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from preprocessing_utils.signal_processing import *
from preprocessing_utils.data_visualization import plot_signals, plot_abp, plot_subject_validity_over_time


def process_windows(abp, ppg, ecg, resp, subject_id, subject_data_list, segment, fs, window_length, window_overlap, args, savepath):
    # Number of measurements in the annotation signal
    sample_n_in_current_subject = len(abp)

    # Divide signals in windows for analysis
    win_start, win_stop = create_windows(window_length, fs, sample_n_in_current_subject, window_overlap)
    n_win = len(win_start)

    # Cycle over windows
    for j in range(0, n_win):

        idx_start = win_start[j]
        idx_stop = win_stop[j]

        window_abp = abp[idx_start: idx_stop + 1]
        window_ppg = ppg[idx_start: idx_stop + 1]
        window_ecg = ecg[idx_start: idx_stop + 1]

        window_resp = None
        if args.resp:
            window_resp = resp[idx_start: idx_stop + 1]

        # Initialize validity flags
        ppg_valid = False
        ecg_valid = False
        abp_valid = False

        # Placeholder for preprocessed signals/features.
        processed_ppg = None
        processed_vpg = None
        processed_apg = None
        processed_ecg = None
        processed_resp = None
        processed_imfs = None
        processed_ppg_freqs = None
        processed_abp = None
        sbp = -1.0
        dbp = -1.0

        # 1. ABP Validity Check
        # Define filtering criteria based on https://www.nature.com/articles/s41597-024-04041-1
        if window_abp.max() <= 200 and window_abp.min() >= 30:  
            temp_sbp, temp_dbp, peaks, valleys = compute_sp_dp(sig=window_abp, fs=fs)
            if temp_sbp > 0 and temp_dbp > 0 and len(peaks) > 0 and len(valleys) > 0:
                # Calculate pulse pressure
                pulse_pressure = temp_sbp - temp_dbp
                
                # 1. Maximum and minimum blood pressure boundaries (200 and 30 mmHg)
                bp_within_extreme_bounds = (30 <= temp_sbp <= 200) and (30 <= temp_dbp <= 200)
                
                # 2. Systolic blood pressure should not be inferior to 60 mmHg
                sbp_above_minimum = temp_sbp >= 60
                
                # 3. Diastolic blood pressure should not exceed 120 mmHg
                dbp_below_maximum = temp_dbp <= 120
                
                # 4. Pulse pressure constraints
                # - Should not be superior to 100 mmHg (high pulse pressure reference)
                # - Should not be inferior to narrow pulse pressure (quarter of SBP value)
                narrow_pulse_pressure = temp_sbp / 4
                pulse_pressure_valid = (narrow_pulse_pressure <= pulse_pressure <= 100)
                
                # Apply all filtering criteria
                abp_is_valid = (bp_within_extreme_bounds and 
                                sbp_above_minimum and 
                                dbp_below_maximum and 
                                pulse_pressure_valid)
                
                if abp_is_valid:
                    abp_valid = True
                    sbp = float(temp_sbp)
                    dbp = float(temp_dbp)
                    processed_abp = window_abp.astype(np.float32)
                else:
                    abp_valid = False
            else:
                abp_valid = False
        else:
                abp_valid = False
        
        # 2. PPG and ECG Validity Check using Autocorrelation Filter
        ppg_is_valid = autocorrelation_filter(window_ppg)
        ecg_is_valid = autocorrelation_filter(window_ecg)

        if ppg_is_valid:
            ppg_valid = True

        if ecg_is_valid:
            ecg_valid = True

        # --- Conditional Preprocessing based on signal validity ---
        if ppg_valid and ecg_valid:
            processed_ppg = window_ppg.astype(np.float32)
            processed_ecg = window_ecg.astype(np.float32)
            if args.resp:
                processed_resp = window_resp.astype(np.float32)

            if args.butterworth_filter:
                processed_ppg = butterworth_filtering(processed_ppg, plot=False, title='PPG-Butter', savepath=savepath)

            if args.resample:
                processed_ppg = resample_signal(processed_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=False, title='PPG-Resampling', savepath=savepath)
                processed_ecg = resample_signal(processed_ecg, original_fs=fs, target_fs=args.target_resample_fs, plot=False, title='ECG-Resampling', savepath=savepath)
                if args.resp:
                    processed_resp = resample_signal(processed_resp, original_fs=fs, target_fs=args.target_resample_fs, plot=False, title='RESP-Resampling', savepath=savepath)

            processed_ppg = standardize(processed_ppg, plot=False, title='PPG-Z-Score', savepath=savepath)
            processed_ecg = standardize(processed_ecg, plot=False, title='ECG-Z-Score', savepath=savepath)
            if args.resp:
                processed_resp = standardize(processed_resp, plot=False, title='RESP-Z-Score', savepath=savepath)

            processed_vpg, processed_apg = calculate_differences(processed_ppg, plot=False, savepath=savepath)

            processed_vpg = standardize(processed_vpg, plot=False, title='VPG-Z-Score', savepath=savepath)
            processed_apg = standardize(processed_apg, plot=False, title='APG-Z-Score', savepath=savepath)

            if args.ppg_emd:
                processed_imfs = get_emd_imfs(processed_ppg, plot=False, savepath=savepath)
                processed_imfs = processed_imfs.astype(np.float32)

            if args.scalogram:
                processed_ppg_freqs = scalogram(processed_ppg, fs=fs, high_freq=12.5, low_freq=0.5, num_scales=16, plot=False, savepath=savepath)
                processed_ppg_freqs = processed_ppg_freqs.astype(np.float32)

        window_data = {
            'lmdb_key': f"{subject_id}_{j}", # The composite LMDB key string for this window
            'ppg_raw': window_ppg.astype(np.float32),
            'ecg_raw': window_ecg.astype(np.float32),
            'resp_raw': window_resp.astype(np.float32) if window_resp is not None else None,
            'abp_raw': window_abp.astype(np.float32),

            'ppg': processed_ppg,
            'vpg': processed_vpg,
            'apg': processed_apg,
            'ecg': processed_ecg,
            'resp': processed_resp,
            'abp': processed_abp,
            'sbp': sbp,
            'dbp': dbp,
            'ppg_emd_imfs': processed_imfs,
            'ppg_scalogram_freqs': processed_ppg_freqs,

            'ppg_valid': ppg_valid,
            'ecg_valid': ecg_valid,
            'abp_valid': abp_valid,
            'is_sig2sig': args.sig2sig
        }
        subject_data_list.append(window_data)


def process_subject(subject_id, args, savepath, lmdb_path, plot_subject=False):
    """
    Processes a single subject's data, writes valid windows directly to LMDB,
    and returns aggregated statistics and LMDB keys for the main process.
    """
    subject_abps = np.load(os.path.join(args.input_folder, 'abp', f'{subject_id}_abp.npy'))
    subject_ppgs = np.load(os.path.join(args.input_folder, 'ppg', f'{subject_id}_ppg.npy'))
    subject_ecgs = np.load(os.path.join(args.input_folder, 'ecg', f'{subject_id}_ecg.npy'))
    subject_resps = np.load(os.path.join(args.input_folder, 'resp', f'{subject_id}_resp.npy')) if args.resp else None

    subject_data_list = [] # List of window_data dicts for the current subject
    fs = args.fs
    window_length = args.window_length
    window_overlap = args.window_overlap

    # First pass: Process windows and collect raw/processed data with validity flags
    process_windows(
        abp=subject_abps, 
        ppg=subject_ppgs, 
        ecg=subject_ecgs, 
        resp=subject_resps,
        subject_id=subject_id,
        subject_data_list=subject_data_list, 
        segment=0,
        fs=fs, 
        window_length=window_length, 
        window_overlap=window_overlap,
        args=args, 
        savepath=savepath
    )
    
    subject_stats = {
        'subject_id': int(subject_id),
        'subject_valid_ppg_ecg_abp_count': 0,
        'subject_valid_ppg_ecg_invalid_abp_count': 0,
        'subject_invalid_ppg_ecg_count': 0,
        'local_discarded_windows_count': 0,
        'local_lmdb_windows_saved_count': 0,
        'supervised_sequence_lengths': [],
        'unlabeled_sequence_lengths': [],
        'ppg_ecg_any_abp_sequence_lengths': [],
        'lmdb_keys_saved': [] # Store the string keys used for this subject in LMDB
    }

    current_supervised_sequence_length = 0
    current_unlabeled_sequence_length = 0
    current_ppg_ecg_any_abp_sequence_length = 0

    LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000
    local_lmdb_env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE, subdir=True, sync=True)

    with local_lmdb_env.begin(write=True) as txn:
        for i, window_data in enumerate(subject_data_list):
            ppg_valid = window_data['ppg_valid']
            ecg_valid = window_data['ecg_valid']
            abp_valid = window_data['abp_valid']

            # Update local statistics (before deciding to save or discard)
            if ppg_valid and ecg_valid:
                current_ppg_ecg_any_abp_sequence_length += 1
                if abp_valid:
                    subject_stats['subject_valid_ppg_ecg_abp_count'] += 1
                    current_supervised_sequence_length += 1
                    if current_unlabeled_sequence_length > 0:
                        subject_stats['unlabeled_sequence_lengths'].append(current_unlabeled_sequence_length)
                        current_unlabeled_sequence_length = 0
                else:
                    subject_stats['subject_valid_ppg_ecg_invalid_abp_count'] += 1
                    current_unlabeled_sequence_length += 1
                    if current_supervised_sequence_length > 0:
                        subject_stats['supervised_sequence_lengths'].append(current_supervised_sequence_length)
                        current_supervised_sequence_length = 0
            else: # Either PPG or ECG or both are invalid -> discard for model input
                subject_stats['subject_invalid_ppg_ecg_count'] += 1
                subject_stats['local_discarded_windows_count'] += 1
                # End any active sequences
                if current_supervised_sequence_length > 0:
                    subject_stats['supervised_sequence_lengths'].append(current_supervised_sequence_length)
                    current_supervised_sequence_length = 0
                if current_unlabeled_sequence_length > 0:
                    subject_stats['unlabeled_sequence_lengths'].append(current_unlabeled_sequence_length)
                    current_unlabeled_sequence_length = 0
                if current_ppg_ecg_any_abp_sequence_length > 0:
                    subject_stats['ppg_ecg_any_abp_sequence_lengths'].append(current_ppg_ecg_any_abp_sequence_length)
                    current_ppg_ecg_any_abp_sequence_length = 0
                continue # Skip saving to LMDB for invalid input windows

            # Use a composite key to ensure uniqueness across subjects
            lmdb_key = f"{subject_id}_{i}" # lmdb_key is now directly the composite string

            # Save data to LMDB
            if window_data['ppg'] is not None:
                txn.put(key=f"{lmdb_key}-ppg".encode(), value=window_data['ppg'].tobytes())
            if window_data['vpg'] is not None:
                txn.put(key=f"{lmdb_key}-vpg".encode(), value=window_data['vpg'].tobytes())
            if window_data['apg'] is not None:
                txn.put(key=f"{lmdb_key}-apg".encode(), value=window_data['apg'].tobytes())
            if window_data['ecg'] is not None:
                txn.put(key=f"{lmdb_key}-ecg".encode(), value=window_data['ecg'].tobytes())
            if window_data['resp'] is not None:
                txn.put(key=f"{lmdb_key}-resp".encode(), value=window_data['resp'].tobytes())

            if window_data['ppg_emd_imfs'] is not None:
                txn.put(key=f"{lmdb_key}-imfs".encode(), value=window_data['ppg_emd_imfs'].tobytes())
            if window_data['ppg_scalogram_freqs'] is not None:
                txn.put(key=f"{lmdb_key}-ppg_freqs".encode(), value=window_data['ppg_scalogram_freqs'].tobytes())

            # Always save raw ABP as it's the ground truth for some cases (e.g., plotting, analysis)
            txn.put(key=f"{lmdb_key}-abp_raw".encode(), value=window_data['abp_raw'].tobytes())

            # Add processed ABP if it was valid (i.e., abp_valid is True)
            if window_data['abp'] is not None:
                txn.put(key=f"{lmdb_key}-abp".encode(), value=window_data['abp'].tobytes())

            txn.put(key=f"{lmdb_key}-sbp".encode(), value=np.array([window_data['sbp']]).tobytes())
            txn.put(key=f"{lmdb_key}-dbp".encode(), value=np.array([window_data['dbp']]).tobytes())

            # Store validity flags as boolean (converted to int8 for storage)
            txn.put(key=f"{lmdb_key}-ppg_valid".encode(), value=np.array([window_data['ppg_valid']]).astype(np.int8).tobytes())
            txn.put(key=f"{lmdb_key}-ecg_valid".encode(), value=np.array([window_data['ecg_valid']]).astype(np.int8).tobytes())
            txn.put(key=f"{lmdb_key}-abp_valid".encode(), value=np.array([window_data['abp_valid']]).astype(np.int8).tobytes())
            txn.put(key=f"{lmdb_key}-is_sig2sig".encode(), value=np.array([window_data['is_sig2sig']]).astype(np.int8).tobytes())

            subject_stats['lmdb_keys_saved'].append(lmdb_key) # Store the actual LMDB key string
            subject_stats['local_lmdb_windows_saved_count'] += 1

    local_lmdb_env.close() # Close LMDB environment for this worker process after its transaction

    # Finalize sequence lengths for this subject
    if current_supervised_sequence_length > 0:
        subject_stats['supervised_sequence_lengths'].append(current_supervised_sequence_length)
    if current_unlabeled_sequence_length > 0:
        subject_stats['unlabeled_sequence_lengths'].append(current_unlabeled_sequence_length)
    if current_ppg_ecg_any_abp_sequence_length > 0:
        subject_stats['ppg_ecg_any_abp_sequence_lengths'].append(current_ppg_ecg_any_abp_sequence_length)

    # Plot subject validity if explicitly requested for this worker process (e.g., for single-subject plot mode)
    if plot_subject:
        # This calls the plot_subject_validity_over_time from `dataset.py` via import
        # (It expects `subject_data_list` as generated by `process_windows`).
        # This will momentarily load the entire subject_data_list into memory for plotting,
        # but only for the `args.plot = True` case which exits afterwards.
        plot_subject_validity_over_time(subject_id, subject_data_list, args.window_length, args.fs, savepath)

    print(f'Completed {subject_id}')
    return subject_stats # Return the collected stats and LMDB keys for the main process


def preprocess_dataset(args):
    subjects_labels_path = [
        f.path for f in os.scandir(os.path.join(args.input_folder, 'abp'))
        if f.is_file() and f.name.endswith('.npy')
    ]

    subjects_ids = []
    for file in subjects_labels_path:
        subjects_ids.append(file.split('/')[-1].split('_')[0])

    figs_savepath = os.path.join(args.figs_folder, args.name)
    os.makedirs(figs_savepath, exist_ok=True)

    lmdb_output_path = os.path.join(args.output_folder, args.name)
    os.makedirs(lmdb_output_path, exist_ok=True)

    if args.plot:
        # If plot is true, process only the first subject, plot its details, and exit.
        # This calls process_subject with `plot_subject=True`.
        subject_id_to_plot = subjects_ids[0]
        print(f"Processing and plotting subject {subject_id_to_plot}")
        process_subject(subject_id_to_plot, args, figs_savepath, lmdb_output_path, plot_subject=True)
        print("Plotting for single subject complete. Exiting as --plot was True.")
        exit()

    # --- Parallel Processing and Aggregation ---
    num_threads = int(args.num_threads)

    # `Parallel` will execute `process_subject` for each ID and return its results
    all_subjects_returned_stats = Parallel(n_jobs=num_threads)(
        delayed(process_subject)(subject_id, args, figs_savepath, lmdb_output_path) for subject_id in subjects_ids
    )

    # --- Aggregate Statistics and Build Final LMDB Index Files ---
    print("\n--- Aggregating Statistics and Building LMDB Index Files ---", flush=True)

    overall_valid_ppg_ecg_abp_counts = 0
    overall_valid_ppg_ecg_invalid_abp_counts = 0
    overall_invalid_ppg_ecg_counts = 0
    total_discarded_windows_count = 0
    total_lmdb_windows_saved = 0

    overall_supervised_sequence_lengths = []
    overall_unlabeled_sequence_lengths = []
    overall_ppg_ecg_any_abp_sequence_lengths = []

    # These are the only structures we need to save for OnlinePhysioDataset
    final_subject_list = [] # List of subject IDs that have data in LMDB
    subject_to_lmdb_keys_map = {} # {subject_id: [list of LMDB keys for that subject, in temporal order]}

    for subject_stats in all_subjects_returned_stats:
        subject_id = subject_stats['subject_id']

        overall_valid_ppg_ecg_abp_counts += subject_stats['subject_valid_ppg_ecg_abp_count']
        overall_valid_ppg_ecg_invalid_abp_counts += subject_stats['subject_valid_ppg_ecg_invalid_abp_count']
        overall_invalid_ppg_ecg_counts += subject_stats['subject_invalid_ppg_ecg_count']
        total_discarded_windows_count += subject_stats['local_discarded_windows_count']
        total_lmdb_windows_saved += subject_stats['local_lmdb_windows_saved_count']

        overall_supervised_sequence_lengths.extend(subject_stats['supervised_sequence_lengths'])
        overall_unlabeled_sequence_lengths.extend(subject_stats['unlabeled_sequence_lengths'])
        overall_ppg_ecg_any_abp_sequence_lengths.extend(subject_stats['ppg_ecg_any_abp_sequence_lengths'])

        # Build LMDB index structures relevant for OnlinePhysioDataset
        if subject_stats['local_lmdb_windows_saved_count'] > 0:
            final_subject_list.append(subject_id)
            # Store the sorted list of LMDB keys for this subject
            subject_to_lmdb_keys_map[subject_id] = sorted(subject_stats['lmdb_keys_saved'], key=lambda k: int(k.split('_')[-1]))


    # Print overall statistics
    print("\n--- Overall Data Distribution ---")
    total_original_windows = overall_valid_ppg_ecg_abp_counts + overall_valid_ppg_ecg_invalid_abp_counts + overall_invalid_ppg_ecg_counts
    if total_original_windows > 0:
        print(f"Total Original Windows (before discarding invalid input): {total_original_windows}")
        print(f"Overall Supervised (PPG, ECG, ABP Valid): {overall_valid_ppg_ecg_abp_counts} ({overall_valid_ppg_ecg_abp_counts / total_original_windows:.2%})")
        print(f"Overall Unlabeled (PPG, ECG Valid, ABP Invalid): {overall_valid_ppg_ecg_invalid_abp_counts} ({overall_valid_ppg_ecg_invalid_abp_counts / total_original_windows:.2%})")
        print(f"Overall Invalid (PPG or ECG Invalid, and thus discarded from LMDB): {overall_invalid_ppg_ecg_counts} ({overall_invalid_ppg_ecg_counts / total_original_windows:.2%})")
    else:
        print("No windows were processed across all subjects.")

    print("\n--- Consecutive Valid Window Lengths ---")
    if overall_supervised_sequence_lengths:
        print(f"Supervised Sequences (PPG, ECG, ABP Valid):")
        print(f"  Min Length: {min(overall_supervised_sequence_lengths)} windows ({min(overall_supervised_sequence_lengths) * args.window_length} seconds)")
        print(f"  Max Length: {max(overall_supervised_sequence_lengths)} windows ({max(overall_supervised_sequence_lengths) * args.window_length} seconds)")
        print(f"  Mean Length: {np.mean(overall_supervised_sequence_lengths):.2f} windows ({np.mean(overall_supervised_sequence_lengths) * args.window_length:.2f} seconds)")
    else:
        print("No supervised sequences found.")

    if overall_unlabeled_sequence_lengths:
        print(f"Unlabeled Sequences (PPG, ECG Valid, ABP Invalid):")
        print(f"  Min Length: {min(overall_unlabeled_sequence_lengths)} windows ({min(overall_unlabeled_sequence_lengths) * args.window_length} seconds)")
        print(f"  Max Length: {max(overall_unlabeled_sequence_lengths)} windows ({max(overall_unlabeled_sequence_lengths) * args.window_length} seconds)")
        print(f"  Mean Length: {np.mean(overall_unlabeled_sequence_lengths):.2f} windows ({np.mean(overall_unlabeled_sequence_lengths) * args.window_length:.2f} seconds)")
    else:
        print("No unlabeled sequences found.")

    if overall_ppg_ecg_any_abp_sequence_lengths:
        print(f"PPG and ECG Valid (regardless of ABP) Sequences:")
        print(f"  Min Length: {min(overall_ppg_ecg_any_abp_sequence_lengths)} windows ({min(overall_ppg_ecg_any_abp_sequence_lengths) * args.window_length} seconds)")
        print(f"  Max Length: {max(overall_ppg_ecg_any_abp_sequence_lengths)} windows ({max(overall_ppg_ecg_any_abp_sequence_lengths) * args.window_length} seconds)")
        print(f"  Mean Length: {np.mean(overall_ppg_ecg_any_abp_sequence_lengths):.2f} windows ({np.mean(overall_ppg_ecg_any_abp_sequence_lengths) * args.window_length:.2f} seconds)")
    else:
        print("No sequences found where PPG and ECG were valid.")

    print(f"\n--- LMDB Storage Summary ---")
    print(f"Windows discarded due to invalid PPG/ECG: {total_discarded_windows_count}")
    print(f"Windows saved to LMDB: {total_lmdb_windows_saved}")

    # Final LMDB write for index files (this needs to be done once by the main process)
    LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000
    final_lmdb_env = lmdb.open(lmdb_output_path, map_size=LMDB_MAP_SIZE, subdir=True)
    with final_lmdb_env.begin(write=True) as txn:
        # Save only the necessary index structures for OnlinePhysioDataset/ContinualLearningDataset
        txn.put(key="subject_list".encode(), value=pickle.dumps(final_subject_list))
        txn.put(key="index_by_subject_id".encode(), value=pickle.dumps(subject_to_lmdb_keys_map))
    final_lmdb_env.close()

    print('Preprocessing completed successfully')


def parseargs():
    parser = argparse.ArgumentParser(description="VitalDB Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_vital_db', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--name', default='vital_db', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=multiprocessing.cpu_count(), type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--fs', default=125, type=int, help='the sampling frequency')
    parser.add_argument('--window_length', default=5, type=int, help='analysis window length in seconds')
    parser.add_argument('--window_overlap', default=3.0, type=float, help='window overlapping in seconds')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load resp or not')
    parser.add_argument('--resample', default='False', type=lambda x: bool(strtobool(x)), help='whether to resample the PPG or not')
    parser.add_argument('--target_resample_fs', default=50, type=float, help='target resampling frequency')
    parser.add_argument('--butterworth_filter', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth PPG with the Butterworth Filter or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to calcualte empirical mode decompoistion for PPG (4 channels) or not')
    parser.add_argument('--scalogram', default='False', type=lambda x: bool(strtobool(x)), help='whether to calcualte the scalogram for PPG or not')
    parser.add_argument('--sig2sig', default='True', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--plot', default='False', type=lambda x: bool(strtobool(x)), help='whether to plot intermediate preprocessing steps or not')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()

    print("Parsed Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()

    if args.ppg_emd and args.scalogram:
        raise ValueError('Returning both EMD of PPG and Scalogram of PPG is not possible (yet)')

    preprocess_dataset(args)