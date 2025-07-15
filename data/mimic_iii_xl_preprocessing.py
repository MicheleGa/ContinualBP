import os
import argparse
from distutils.util import strtobool
import glob
import lmdb
import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from pyampd.ampd import find_peaks
from preprocessing_utils.signal_processing import *
from preprocessing_utils.data_visualization import plot_signals, plot_abp


def process_windows(abp, ppg, ecg, resp, subject_id, subject_data, trial_folder, fs, window_length, window_overlap, args, savepath):
    
    # Number of measurements in the annotation signal
    sample_n_in_current_abp = len(abp)

    # Divide signals in windows for analysis
    win_start, win_stop = create_windows(window_length, fs, sample_n_in_current_abp, window_overlap)
    n_win = len(win_start)
    
    # Cycle over windows        
    for j in range(0, n_win):
        
        # Extract windows data
        idx_start = win_start[j]
        idx_stop = win_stop[j]
        
        window_abp = abp[idx_start: idx_stop + 1]
        window_ppg = ppg[idx_start: idx_stop + 1]
        
        if args.ecg:
            window_ecg = ecg[idx_start: idx_stop + 1]
        
        if args.resp:
            window_resp = resp[idx_start: idx_stop + 1]
            
        # Initialize validity flags
        abp_valid = False
        ppg_valid = False
        if args.ecg:
            ecg_valid = False
        if args.resp:
            resp_valid = False
            
        # SBP/DBP calculation
        sbp = -1.0
        dbp = -1.0
        
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
                    window_abp = window_abp.astype(np.float32)
                else:
                    abp_valid = False
            else:
                abp_valid = False
        else:
            abp_valid = False
        
        if abp_valid:
            # 2. PPG (and ECG/RESP) Validity Check using Autocorrelation Filter
            ppg_is_valid = autocorrelation_filter(window_ppg)
            if ppg_is_valid:
                ppg_valid = True

                if args.ecg:
                    ecg_is_valid = autocorrelation_filter(window_ecg)
                    if ecg_is_valid:
                        ecg_valid = True
                
                    if args.resp:
                        resp_is_valid = autocorrelation_filter(window_resp)
                        if resp_is_valid:
                            resp_valid = True
        
        if abp_valid and ppg_valid and (not args.ecg) and (not args.resp):
            
            # Preprocess PPG window before saving it in the subject data
            if args.butterworth_filter:
                # 4-th order butterworth filtering
                window_ppg = butterworth_filtering(window_ppg, plot=args.plot, title='PPG-Butter', savepath=savepath)
           
            if args.resample:
                # Resampling to target fs: note that for PPG at least 25 Hz are required
                window_ppg = resample_signal(window_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='PPG-Resampling', savepath=savepath)
                
            # Zero-mean standardization (Z-score)
            window_ppg = standardize(window_ppg, plot=args.plot, title='PPG-Z-Score', savepath=savepath)

            # Ensure FP32 type for PPG
            window_ppg = window_ppg.astype(np.float32)
            
            if args.sig2sig:
                subject_data.append((window_ppg, None, None, window_abp))
            else:
                subject_data.append((window_ppg, None, None, sbp, dbp))
                
            if args.plot:
                plot_abp(window_abp, fs=fs, peaks=peaks, valleys=valleys, title=f'ABP [SBP {sbp:.2f} - DBP {dbp:.2f}]', save_path=savepath)
                
                if args.sig2sig:
                    plot_signals(
                        [window_ppg, window_abp], 
                        labels=['PPG', 'ABP'], 
                        title=f'Sample Input Signals', 
                        savepath=savepath, 
                        ylabels=['a.u.', 'mmHg']
                        )
                else:
                    plot_signals(
                        [window_ppg], 
                        labels=['PPG'], 
                        title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                        savepath=savepath, 
                        ylabels=['a.u.']
                        )
                # No need to do the preprocessing of all subjects when plot true
                exit()
    
        elif abp_valid and ppg_valid and args.ecg and ecg_valid and (not args.resp):
            
            # Preprocess PPG/ECG windows before saving it in the subject data
            if args.butterworth_filter:
                # 4-th order butterworth filtering
                window_ppg = butterworth_filtering(window_ppg, plot=args.plot, title='PPG-Butter', savepath=savepath)
            
            # TO-DO: ECG filtering (DWT maybe)
            
            if args.resample:
                # Resampling to target fs: note that for ECG at least 50 Hz are required
                window_ppg = resample_signal(window_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='PPG-Resampling', savepath=savepath)
                window_ecg = resample_signal(window_ecg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ECG-Resampling', savepath=savepath)
            
            # Zero-mean standardization (Z-score)
            window_ppg = standardize(window_ppg, plot=args.plot, title='PPG-Z-Score', savepath=savepath)
            window_ecg = standardize(window_ecg, plot=args.plot, title='ECG-Z-Score', savepath=savepath)
            
            # Ensure FP32 type for PPG
            window_ppg = window_ppg.astype(np.float32)
            window_ecg = window_ecg.astype(np.float32)
            
            if args.sig2sig:
                subject_data.append((window_ppg, window_ecg, None, window_abp))
            else:
                subject_data.append((window_ppg, window_ecg, None, sbp, dbp))
                
            if args.plot:
                plot_abp(window_abp, fs=fs, peaks=peaks, valleys=valleys, title=f'ABP [SBP {sbp:.2f} - DBP {dbp:.2f}]', save_path=savepath)
                if args.sig2sig:
                    plot_signals(
                        [window_ppg, window_ecg, window_abp], 
                        labels=['PPG', 'ECG', 'ABP'], 
                        title=f'Sample Input Signals', 
                        savepath=savepath, 
                        ylabels=['a.u.', 'mV', 'mmHg']
                        )
                else:
                    plot_signals(
                        [window_ppg, window_ecg], 
                        labels=['PPG', 'ECG'], 
                        title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                        savepath=savepath, 
                        ylabels=['a.u.', 'mV']
                        )
                # No need to do the preprocessing of all subjects when plot true
                exit()
        
        elif abp_valid and ppg_valid and args.ecg and ecg_valid and args.resp and resp_valid:
            
            # Preprocess PPG/ECG/RESP windows before saving it in the subject data
            if args.butterworth_filter:
                # 4-th order butterworth filtering
                window_ppg = butterworth_filtering(window_ppg, plot=args.plot, title='PPG-Butter', savepath=savepath)
            
            # TO-DO: ECG/RESP filtering (DWT maybe)
            
            if args.resample:
                # Resampling to target fs: note that for ECG at least 50 Hz are required, what about RESP?
                window_ppg = resample_signal(window_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='PPG-Resampling', savepath=savepath)
                window_ecg = resample_signal(window_ecg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ECG-Resampling', savepath=savepath)
                window_resp = resample_signal(window_resp, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='RESP-Resampling', savepath=savepath)
        
            # Zero-mean standardization (Z-score)
            window_ppg = standardize(window_ppg, plot=args.plot, title='PPG-Z-Score', savepath=savepath)
            window_ecg = standardize(window_ecg, plot=args.plot, title='ECG-Z-Score', savepath=savepath)
            window_resp = standardize(window_resp, plot=args.plot, title='RESP-Z-Score', savepath=savepath)
        
            # Ensure FP32 type for PPG
            window_ppg = window_ppg.astype(np.float32)
            window_ecg = window_ecg.astype(np.float32)
            window_resp = window_resp.astype(np.float32)
            
            if args.sig2sig:
                subject_data.append((window_ppg, window_ecg, window_resp, window_abp))
            else:
                subject_data.append((window_ppg, window_ecg, window_resp, sbp, dbp))
            
            if args.plot:
                plot_abp(window_abp, fs=fs, peaks=peaks, valleys=valleys, title=f'ABP [SBP {sbp:.2f} - DBP {dbp:.2f}]', save_path=savepath)
                if args.sig2sig:
                    plot_signals(
                        [window_ppg, window_ecg, window_resp, window_abp], 
                        labels=['PPG', 'ECG', 'RESP', 'ABP'], 
                        title=f'Sample Input Signals', 
                        savepath=savepath, 
                        ylabels=['a.u.', 'mV', 'pm', 'mmHg']
                        )
                else:
                    plot_signals(
                        [window_ppg, window_ecg, window_resp], 
                        labels=['PPG', 'ECG', 'RESP'], 
                        title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                        savepath=savepath, 
                        ylabels=['a.u.', 'mV', 'pm']
                        )
                # No need to do the preprocessing of all subjects when plot true
                exit()
    

def process_subject(subject_id, args, savepath, result_queue=None):
    
    subject_data = []
    fs = args.fs
    window_length = args.window_length # seconds
    window_overlap = args.window_overlap # overlap in seconds
    
    # Find all trial folders for this subject - [0] is because a single list element 
    # is returned by glob as the patient id is unique, hence [0] is important for the second glob function
    subject_folder = glob.glob(os.path.join(args.input_folder, f"p*/{subject_id}"))[0]
    trial_folders = sorted(glob.glob(os.path.join(subject_folder, "*_*")))
    
    for trial_folder in trial_folders:
        abp_path = os.path.join(trial_folder, "abp.npy")
        ppg_path = os.path.join(trial_folder, "ppg.npy")
        ecg_path = os.path.join(trial_folder, "ecg.npy")
        resp_path = os.path.join(trial_folder, "resp.npy")
        
        # Always require abp and ppg
        if not (os.path.isfile(abp_path) and os.path.isfile(ppg_path)):
            continue
        
        # If ecg is required but not present for the trial, skip it
        if args.ecg and not os.path.isfile(ecg_path):
            print(f"Skipping {subject_id} in {trial_folder} as ECG is required but not present.")
            continue
            
        # If resp is required but not present for the trial, skip it
        if args.resp and not os.path.isfile(resp_path):
            print(f"Skipping {subject_id} in {trial_folder} as RESP is required but not present.")
            continue
        
        abp = np.load(abp_path)
        ppg = np.load(ppg_path)
        
        ecg = np.load(ecg_path) if args.ecg else None
        resp = np.load(resp_path) if args.resp else None
            
        process_windows(
            abp=abp,
            ppg=ppg,
            ecg=ecg,
            resp=resp,
            subject_id=subject_id,
            subject_data=subject_data,
            trial_folder=trial_folder,
            fs=fs,
            window_length=window_length,
            window_overlap=window_overlap,
            args=args,
            savepath=savepath
        )

    print(f'Completed {subject_id}')
    if result_queue is not None:
        result_queue.put((int(subject_id[1:]), subject_data)) # subject id is integer
    else:    
        print(f'No result queue provided for {subject_id}, retry with another subject id ...')

def preprocess_dataset(args):
    
    subjects_ids = set()
    
    # Pattern for patient folders: p******
    patient_folders = glob.glob(os.path.join(args.input_folder, "p[0-9]*/p[0-9]*****"))
    
    for patient_folder in patient_folders:
        subject_id = os.path.basename(patient_folder)
        subjects_ids.add(subject_id)
    
    subjects_ids = list(subjects_ids)
    print(f"{len(subjects_ids)} subjects found in the dataset {args.input_folder}")
    
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
    subject_id_list = list()
    sample_id = 0

    with lmdbenv.begin(write=True) as txn:
        while not result_queue.empty():
            subject_id, subject_data = result_queue.get()
            
            if len(subject_data) == 0:
                print(f"Skipping {subject_id} as no valid data found")
                continue
            
            subject_id_list.append(subject_id)
            index_by_subject_id[subject_id] = []
            subject_n_recording = 0
            
            if args.sig2sig:
                for window_ppg, window_ecg, window_resp, window_abp in subject_data:
                    txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                    if window_ecg is not None:
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                    if window_resp is not None:
                        txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes())
                    txn.put(key="{}-abp".format(sample_id).encode(), value=window_abp.tobytes())
                    
                    index_by_sample_id.append((subject_id, subject_n_recording))
                    index_by_subject_id[subject_id].append(sample_id)
                    sample_id += 1
                    subject_n_recording += 1
            else:
                for window_ppg, window_ecg, window_resp, sbp, dbp in subject_data:
                    txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                    if window_ecg is not None:
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                    if window_resp is not None:
                        txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes())
                    txn.put(key="{}-sbp".format(sample_id).encode(), value=np.array([sbp]).tobytes())
                    txn.put(key="{}-dbp".format(sample_id).encode(), value=np.array([dbp]).tobytes())

                    index_by_sample_id.append((subject_id, subject_n_recording))
                    index_by_subject_id[subject_id].append(sample_id)
                    sample_id += 1
                    subject_n_recording += 1                        

        txn.put(key="index_by_sample_id".encode(), value=pickle.dumps(index_by_sample_id))
        txn.put(key="index_by_subject_id".encode(), value=pickle.dumps(index_by_subject_id))
        txn.put(key="subject_list".encode(), value=pickle.dumps(subject_id_list))

    print('Preprocessing completed successfully')


def check_invalid_window_abp(args):
    LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000 # 1T
    lmdb_folder = os.path.join(args.output_folder, args.name) 
    lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE)
    with lmdbenv.begin(write=True) as txn:    
        subject_list:list = pickle.loads(txn.get("subject_list".encode()))
        index_by_subject_id:dict = pickle.loads(txn.get("index_by_subject_id".encode()))
        index_by_sample_id = pickle.loads(txn.get("index_by_sample_id".encode()))
        
        print(len(subject_list), "subjects in the dataset")
        
        for subject_id in subject_list:
            sample_list = index_by_subject_id[subject_id]
            invalid_sample_ids = []
            valid_sample_ids = []
            for sample_id in sample_list:
                window_abp = np.squeeze(np.frombuffer(txn.get("{}-abp".format(sample_id).encode()), dtype="float32"))
                sbp, dbp, peaks, valleys = compute_sp_dp(window_abp, fs=args.fs)
                if sbp < 0 or dbp < 0 or len(peaks) == 0 or len(valleys) == 0:
                    print(f'Error during peaks/valleys processing: no peaks or valleys found in the signal {sample_id} of subject {subject_id}.')
                    invalid_sample_ids.append(sample_id)
                else:
                    valid_sample_ids.append(sample_id)
            if len(invalid_sample_ids) > 0:
                print(f'Invalid samples for subject {subject_id}: {invalid_sample_ids}')

                # Update index_by_subject_id
                index_by_subject_id[subject_id] = valid_sample_ids

                # Remove invalid samples from index_by_sample_id
                for invalid_sample_id in invalid_sample_ids:
                    if invalid_sample_id in index_by_sample_id:
                        del index_by_sample_id[invalid_sample_id]

                print(f'Removed the following samples from the dataset: {invalid_sample_ids}')

        # Write updated indices back to LMDB
        txn.put("index_by_subject_id".encode(), pickle.dumps(index_by_subject_id))
        txn.put("index_by_sample_id".encode(), pickle.dumps(index_by_sample_id))

        print("LMDB dataset updated successfully.")

 
def parseargs():
    parser = argparse.ArgumentParser(description="MIMIC III XL Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_mimic_iii_xl', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--fs', default=125, type=int, help='the sampling frequency')
    parser.add_argument('--window_length', default=5, type=int, help='analysis window length in seconds')
    parser.add_argument('--window_overlap', default=3.0, type=float, help='window overlapping in seconds')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load resp or not')
    parser.add_argument('--resample', default='False', type=lambda x: bool(strtobool(x)), help='whether to resample the PPG or not')
    parser.add_argument('--target_resample_fs', default=50, type=float, help='target resampling frequency')
    parser.add_argument('--butterworth_filter', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth PPG with the Butterworth Filter or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
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