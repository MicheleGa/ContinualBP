import os
import argparse
from distutils.util import strtobool
import lmdb
import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from pyampd.ampd import find_peaks
from preprocessing_utils.signal_processing import *
from preprocessing_utils.data_visualization import plot_signals, plot_abp


def process_windows(abp, ppg, ecg, resp, subject_id, subject_data, segment, fs, window_length, window_overlap, args, savepath):
    # Number of measurements in the annotation signal
    sample_n_in_current_subject = len(abp)

    # Divide signals in windows for analysis
    win_start, win_stop = create_windows(window_length, fs, sample_n_in_current_subject, window_overlap)
    n_win = len(win_start)
    
    # Cycle over windows        
    for j in range(0, n_win):
        
        idx_start = win_start[j]
        idx_stop = win_stop[j]
        
        # Process and check annotation first
        window_abp = abp[idx_start: idx_stop + 1]
        
        if args.fir_bp_filtering:
            # Apply bandpass FIR filter to ABP
            window_abp = high_freq_butterworth(signal=window_abp, fs=fs, cutoff_freq=35, plot=args.plot, title='ABP-LP', savepath=savepath)
            
        # Compute SBP, DBP, peaks and valleys# SBP/DBP calculation
        sbp, dbp, peaks, valleys = compute_sp_dp(sig=window_abp, fs=fs) 
                    
        if sbp < 0 or dbp < 0 or len(peaks) == 0 or len(valleys) == 0:
            print(f"\t{subject_id}|{j + 1} of {n_win} - Peaks/Valleys not found for {subject_id}, in segment {segment}, window {j} [{idx_start}:{idx_stop}]")
        else:
            
            # Plot BP before resampling, otherwise displayed peaks/valleys will be inconsistent
            if args.plot:
                plot_abp(window_abp, fs=fs, peaks=peaks, valleys=valleys, title=f'ABP [SBP {sbp:.2f} - DBP {dbp:.2f}]', save_path=savepath)
            
            # Process also input signals        
            window_ppg = ppg[idx_start: idx_stop + 1]
            
            if args.ecg:
                window_ecg = ecg[idx_start: idx_stop + 1]
            
            if args.resp:
                window_resp = resp[idx_start: idx_stop + 1]
            
            if args.butterworth_filter:
                window_ppg = butterworth_filtering(signal=window_ppg, fs=fs, level=4, low_freq=0.5, high_freq=12.5, plot=args.plot, title='PPG-FIR', savepath=savepath)
                
            if args.resample:
                # Resampling to target fs: note that for ECG at least 50 Hz are required; TODO look for references on the target fs for RESP
                window_ppg = resample_signal(window_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='PPG-Resampling', savepath=savepath)
                if args.ecg:
                    window_ecg = resample_signal(window_ecg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ECG-Resampling', savepath=savepath)
                if args.resp:
                    window_resp = resample_signal(window_resp, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='RESP-Resampling', savepath=savepath)
                window_abp = resample_signal(window_abp, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ABP-Resampling', savepath=savepath)
                
            # Normalization: Rescale to unit, Zero-mean standardization (Z-score) or EMA Z-score 
            if args.ema_std:
                window_ppg = ema_normalization(window_ppg, plot=args.plot, title='PPG-EMA-Z-Score', savepath=savepath)
                if args.ecg:
                    window_ecg = ema_normalization(window_ecg, plot=args.plot, title='ECG-EMA-Z-Score', savepath=savepath)
                if args.resp:
                    window_resp = ema_normalization(window_resp, plot=args.plot, title='RESP-EMA-Z-Score', savepath=savepath)
            elif args.rescale_to_unit:
                window_ppg = rescale_to_unit(window_ppg, plot=args.plot, title='PPG-Rescale', savepath=savepath)
                if args.ecg:
                    window_ecg = rescale_to_unit(window_ecg, plot=args.plot, title='ECG-Rescale', savepath=savepath)
                if args.resp:
                    window_resp = rescale_to_unit(window_resp, plot=args.plot, title='RESP-Rescale', savepath=savepath)
            elif args.percentile:
                window_ppg = percentile_normalize(window_ppg, plot=args.plot, title='PPG-Percentile', savepath=savepath)
                if args.ecg:
                    window_ecg = percentile_normalize(window_ecg, plot=args.plot, title='ECG-Percentile', savepath=savepath)
                if args.resp:
                    window_resp = percentile_normalize(window_resp, plot=args.plot, title='RESP-Percentile', savepath=savepath)
            else:
                window_ppg = standardize(window_ppg, plot=args.plot, title='PPG-Z-Score', savepath=savepath)
                if args.ecg:
                    window_ecg = standardize(window_ecg, plot=args.plot, title='ECG-Z-Score', savepath=savepath)
                if args.resp:
                    window_resp = standardize(window_resp, plot=args.plot, title='RESP-Z-Score', savepath=savepath)
            
            # Ensure signals ar in floating point format
            window_abp = window_abp.astype(np.float32)
            sbp = sbp.astype(np.float32)
            dbp = dbp.astype(np.float32)
            
            window_ppg = window_ppg.astype(np.float32)
            if args.ecg:
                window_ecg = window_ecg.astype(np.float32)
            if args.resp:
                window_resp = window_resp.astype(np.float32)
                
            if args.ecg:
                if args.resp:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_ecg, window_resp, window_abp))
                    else:
                        subject_data.append((window_ppg, window_ecg, window_resp, sbp, dbp))
                else:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_ecg, None, window_abp))
                    else:
                        subject_data.append((window_ppg, window_ecg, None, sbp, dbp))
            else:
                if args.resp:
                    if args.sig2sig:
                        subject_data.append((window_ppg, None, window_resp, window_abp))
                    else:
                        subject_data.append((window_ppg, None, window_resp, sbp, dbp))
                else:
                    if args.sig2sig:
                        subject_data.append((window_ppg, None, None, window_abp))
                    else:
                        subject_data.append((window_ppg, None, None, sbp, dbp))
                    
            if args.plot:
                if args.ecg:
                    if args.resp:
                        if args.sig2sig:
                            plot_signals(
                                [window_ppg, window_ecg, window_resp, window_abp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'ECG', 'RESP', 'ABP'], 
                                title=f'Sample Input Signals', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'mV', 'pm', 'mmHg']
                                )
                        else:
                            plot_signals(
                                [window_ppg, window_ecg, window_resp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'ECG', 'RESP'], 
                                title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'mV', 'pm']
                                )
                    else:
                        if args.sig2sig:
                            plot_signals(
                                [window_ppg, window_ecg, window_abp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'ECG', 'ABP'], 
                                title=f'Sample Input Signals', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'mV', 'mmHg']
                                )
                        else:
                            plot_signals(
                                [window_ppg, window_ecg], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'ECG'], 
                                title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'mV']
                                )
                else:
                    if args.resp:
                        if args.sig2sig:
                            plot_signals(
                                [window_ppg, window_resp, window_abp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'RESP', 'ABP'], 
                                title=f'Sample Input Signals', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'pm', 'mmHg']
                                )
                        else:
                            plot_signals(
                                [window_ppg, window_resp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'RESP'], 
                                title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'pm']
                                )
                    else:
                        if args.sig2sig:
                            plot_signals(
                                [window_ppg, window_abp], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG', 'ABP'], 
                                title=f'Sample Input Signals', 
                                savepath=savepath, 
                                ylabels=['a.u.', 'mmHg']
                                )
                        else:
                            plot_signals(
                                [window_ppg], 
                                fs=fs if not args.resample else args.target_resample_fs,
                                labels=['PPG'], 
                                title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                                savepath=savepath, 
                                ylabels=['a.u.']
                                )
                            
                # No need to do the preprocessing of all subjects when plot true
                exit()

def process_subject(subject_id, args, savepath, result_queue=None):
    
    # Load annotation
    subject_abps = np.load(os.path.join(args.input_folder, 'abp', f'{subject_id}_abp.npy'))
    
    # Load input signals
    subject_ppgs = np.load(os.path.join(args.input_folder, 'ppg', f'{subject_id}_ppg.npy'))
    if args.ecg:
        subject_ecgs = np.load(os.path.join(args.input_folder, 'ecg', f'{subject_id}_ecg.npy'))
    if args.resp:
        subject_resps = np.load(os.path.join(args.input_folder, 'resp', f'{subject_id}_resp.npy'))
    
    subject_data = []
    fs = args.fs
    window_length = args.window_length # seconds
    window_overlap = args.window_overlap # overlap in seconds
    
    # Each subject has 30 segments with the required signals
    for segment in range(subject_abps.shape[0]):
        
        # Get annotation segment
        abp = subject_abps[segment, :]
        
        # Get input signal segments
        ppg = subject_ppgs[segment, :]
        if args.ecg:
            ecg = subject_ecgs[segment, :]
        if args.resp:
            resp = subject_resps[segment, :]

        process_windows(
            abp=abp,
            ppg=ppg,
            ecg=ecg if args.ecg else None,
            resp=resp if args.resp else None,
            subject_id=subject_id,
            subject_data=subject_data,
            segment=segment,
            fs=fs,
            window_length=window_length,
            window_overlap=window_overlap,
            args=args,
            savepath=savepath
        )
    
    print(f'Completed {subject_id}')
    result_queue.put((int(subject_id[1:]), subject_data)) # subject id is integer
    

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
        # p056287
        print(f"Plotting subject {subject_id}")
        #process_subject(subject_id, args, savepath)
        process_subject("p056287", args, savepath)
        
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


def parseargs():
    parser = argparse.ArgumentParser(description="MIMIC III Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_mimic_iii', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--fs', default=125, type=int, help='the sampling frequency')
    parser.add_argument('--window_length', default=5, type=int, help='analysis window length in seconds')
    parser.add_argument('--window_overlap', default=3.0, type=float, help='window overlapping in seconds')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load resp or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load resp or not')
    parser.add_argument('--ema_std', default='False', type=lambda x: bool(strtobool(x)), help='whether to filter input signals with an EMA Z-score')
    parser.add_argument('--percentile', default='False', type=lambda x: bool(strtobool(x)), help='whether to normalize the signals with percentile normalization')
    parser.add_argument('--rescale_to_unit', default='False', type=lambda x: bool(strtobool(x)), help='whether to rescale the input signals to the range [0,1] or not')
    parser.add_argument('--resample', default='False', type=lambda x: bool(strtobool(x)), help='whether to resample the PPG or not')
    parser.add_argument('--target_resample_fs', default=50, type=float, help='target resampling frequency')
    parser.add_argument('--butterworth_filter', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth PPG with the Butterworth Filter or not')
    parser.add_argument('--fir_bp_filtering', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth BP with a bandpass FIR filter or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--plot', default='False', type=lambda x: bool(strtobool(x)), help='whether to plot intermediate preprocessing steps or not')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    
    print("Parsed Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()
    
    preprocess_dataset(args)
    