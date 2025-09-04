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


def process_windows(abp, ppg, ecg, subject_id, fs, window_length, window_overlap, args, savepath):
    # Number of measurements in the annotation signal
    sample_n_in_current_subject = len(abp)
    

    # Divide signals in windows for analysis
    win_start, win_stop = create_windows(window_length, fs, sample_n_in_current_subject, window_overlap)
    n_win = len(win_start)
    
    subject_data = []

    # Cycle over windows
    for j in range(0, n_win):

        idx_start = win_start[j]
        idx_stop = win_stop[j]

        window_abp = abp[idx_start: idx_stop + 1]
        window_ppg = ppg[idx_start: idx_stop + 1]
        window_ecg = ecg[idx_start: idx_stop + 1]
        
        sbp = -1.0
        dbp = -1.0
        map = -1.0

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
                
                # Apply all filtering criteria and proceed with further processing
                if bp_within_extreme_bounds and sbp_above_minimum and dbp_below_maximum and pulse_pressure_valid:
                    
                    sbp = temp_sbp
                    dbp = temp_dbp
                    map = (2 * dbp + sbp) / 3
                    window_abp = window_abp.astype(np.float32)
                
                    # 2. PPG and ECG Validity Check using Autocorrelation Filter
                    ppg_is_valid = autocorrelation_filter(window_ppg)
                    ecg_is_valid = autocorrelation_filter(window_ecg)

                    # --- Conditional Preprocessing based on signal validity ---
                    if ppg_is_valid and ecg_is_valid:
                        if args.butterworth_filter:
                            window_ppg = butterworth_filtering(signal=window_ppg, fs=fs, level=4, low_freq=0.5, high_freq=12.5, plot=args.plot, title='PPG-FIR', savepath=savepath)
                            
                        if args.resample:
                            # Resampling to target fs: note that for ECG at least 50 Hz are required; TODO look for references on the target fs for RESP
                            window_ppg = resample_signal(window_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='PPG-Resampling', savepath=savepath)
                            window_ecg = resample_signal(window_ecg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ECG-Resampling', savepath=savepath)
                            window_abp = resample_signal(window_abp, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ABP-Resampling', savepath=savepath)
                            
                        # Normalization: Rescale to unit, Zero-mean standardization (Z-score) or EMA Z-score 
                        if args.ema_std:
                            window_ppg = ema_normalization(window_ppg, plot=args.plot, title='PPG-EMA-Z-Score', savepath=savepath)
                            window_ecg = ema_normalization(window_ecg, plot=args.plot, title='ECG-EMA-Z-Score', savepath=savepath)
                        elif args.rescale_to_unit:
                            window_ppg = rescale_to_unit(window_ppg, plot=args.plot, title='PPG-Rescale', savepath=savepath)
                            window_ecg = rescale_to_unit(window_ecg, plot=args.plot, title='ECG-Rescale', savepath=savepath)
                        elif args.percentile:
                            window_ppg = percentile_normalize(window_ppg, plot=args.plot, title='PPG-Percentile', savepath=savepath)
                            window_ecg = percentile_normalize(window_ecg, plot=args.plot, title='ECG-Percentile', savepath=savepath)
                        else:
                            window_ppg = standardize(window_ppg, plot=args.plot, title='PPG-Z-Score', savepath=savepath)
                            window_ecg = standardize(window_ecg, plot=args.plot, title='ECG-Z-Score', savepath=savepath)
                            
                        # Ensure signals ar in floating point format
                        window_abp = window_abp.astype(np.float32)
                        sbp = sbp.astype(np.float32)
                        dbp = dbp.astype(np.float32)
                        map = map.astype(np.float32)
                        
                        window_ppg = window_ppg.astype(np.float32)
                        window_ecg = window_ecg.astype(np.float32)
                        
                        # Store the window data along with its original chronological index (j)
                        subject_data.append((window_ppg, window_ecg, window_abp, sbp, dbp, map, j))
                                                
                        if args.plot:
                            plot_signals(
                                [window_ppg, window_ecg, window_abp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'ECG', 'ABP'], 
                                title=f'Subejct {subject_id} Sample Input Signals ~ SBP {sbp:.2f} - DBP {dbp:.2f} - MAP {map:.2f}', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'mV', 'mmHg']
                                )
                                        
                            # No need to do the preprocessing of all subjects when plot true
                            return
                    
    if len(subject_data) > 0:
        return subject_data
    else: 
        None
        
            
def process_subject(subject_id, args, savepath, result_queue=None):
    """
    Processes a single subject's data, writes valid windows directly to LMDB,
    and returns aggregated statistics and LMDB keys for the main process.
    """
    # Load annotation
    subject_abps = np.load(os.path.join(args.input_folder, 'abp', f'{subject_id}_abp.npy'))
    
    # Load input signals
    subject_ppgs = np.load(os.path.join(args.input_folder, 'ppg', f'{subject_id}_ppg.npy'))
    subject_ecgs = np.load(os.path.join(args.input_folder, 'ecg', f'{subject_id}_ecg.npy'))

    fs = args.fs
    window_length = args.window_length
    window_overlap = args.window_overlap

    # Process windows
    subject_data = process_windows(
        abp=subject_abps, 
        ppg=subject_ppgs, 
        ecg=subject_ecgs, 
        subject_id=subject_id,
        fs=fs, 
        window_length=window_length, 
        window_overlap=window_overlap,
        args=args, 
        savepath=savepath
    )
    
    if args.plot:
        return

    print(f'Completed {subject_id}')
    result_queue.put((int(subject_id), subject_data))


def preprocess_dataset(args):
    subjects_labels_path = [
        f.path for f in os.scandir(os.path.join(args.input_folder, 'abp'))
        if f.is_file() and f.name.endswith('.npy')
    ]

    subjects_ids = []
    for file in subjects_labels_path:
        subjects_ids.append(file.split('/')[-1].split('_')[0])

    if args.plot:
        savepath = os.path.join(args.figs_folder, args.name)
        os.makedirs(savepath, exist_ok=True)
    else:
        savepath = args.figs_folder
        
    if args.plot:
        # Choose a random subject to plot
        subject_id = np.random.choice(subjects_ids)
        print(f"Plotting subject {subject_id}")
        process_subject(subject_id, args, savepath)
        
        # No need to do the preprocessing of all subjects when plot true
        return
        
    # Dataset preprocessing
    LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000  # 1T

    lmdbenv = lmdb.open(os.path.join(args.output_folder, args.name), map_size=LMDB_MAP_SIZE)
        
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    num_threads = int(args.num_threads)  # Set the desired number of threads here

    # Use joblib for parallel processing
    Parallel(n_jobs=num_threads)(
        delayed(process_subject)(subject_id, args, savepath, result_queue) for subject_id in subjects_ids
    )
    
    index_by_sample_id = list()
    index_by_subject_id = dict()
    subject_adjacent_samples = dict()
    subject_id_list = list()
    sample_id = 0

    with lmdbenv.begin(write=True) as txn:
        while not result_queue.empty():
            subject_id, subject_data = result_queue.get()
            
            if subject_data is None:
                print(f"Skipping {subject_id} as no valid data found")
                continue
            
            subject_id_list.append(subject_id)
            index_by_subject_id[subject_id] = []
            subject_adjacent_samples[subject_id] = []
            subject_n_recording = 0
            
            for window_ppg, window_ecg, window_abp, sbp, dbp, map, window_idx in subject_data:
                txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                txn.put(key="{}-abp".format(sample_id).encode(), value=window_abp.tobytes())
                txn.put(key="{}-sbp".format(sample_id).encode(), value=np.array([sbp]).tobytes())
                txn.put(key="{}-dbp".format(sample_id).encode(), value=np.array([dbp]).tobytes())
                txn.put(key="{}-map".format(sample_id).encode(), value=np.array([map]).tobytes())

                index_by_sample_id.append((subject_id, subject_n_recording))
                index_by_subject_id[subject_id].append(sample_id)
                subject_adjacent_samples[subject_id].append(window_idx)
                sample_id += 1
                subject_n_recording += 1
            
        txn.put(key="index_by_sample_id".encode(), value=pickle.dumps(index_by_sample_id))
        txn.put(key="index_by_subject_id".encode(), value=pickle.dumps(index_by_subject_id))
        txn.put(key="subject_adjacent_samples".encode(), value=pickle.dumps(subject_adjacent_samples))
        txn.put(key="subject_list".encode(), value=pickle.dumps(subject_id_list))

    print('Preprocessing completed successfully')



def parseargs():
    parser = argparse.ArgumentParser(description="VitalDB Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_vital_db', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--name', default='vital_db', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--fs', default=125, type=int, help='the sampling frequency')
    parser.add_argument('--window_length', default=5, type=int, help='analysis window length in seconds')
    parser.add_argument('--window_overlap', default=3.0, type=float, help='window overlapping in seconds')
    parser.add_argument('--ema_std', default='False', type=lambda x: bool(strtobool(x)), help='whether to filter input signals with an EMA Z-score')
    parser.add_argument('--percentile', default='False', type=lambda x: bool(strtobool(x)), help='whether to normalize the signals with percentile normalization')
    parser.add_argument('--rescale_to_unit', default='False', type=lambda x: bool(strtobool(x)), help='whether to rescale the input signals to the range [0,1] or not')
    parser.add_argument('--resample', default='False', type=lambda x: bool(strtobool(x)), help='whether to resample the PPG or not')
    parser.add_argument('--target_resample_fs', default=50, type=float, help='target resampling frequency')
    parser.add_argument('--butterworth_filter', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth PPG with the Butterworth Filter or not')
    parser.add_argument('--fir_bp_filtering', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth BP with a bandpass FIR filter or not')
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

    preprocess_dataset(args)