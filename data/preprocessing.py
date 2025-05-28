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
        
        window_abp = abp[idx_start: idx_stop + 1]
        window_ppg = ppg[idx_start: idx_stop + 1]
        window_ecg = ecg[idx_start: idx_stop + 1]
        
        if args.resp:
            window_resp = resp[idx_start: idx_stop + 1]
        
        if args.butterworth_filter:
            # 4-th order butterworth filtering
            window_ppg = butterworth_filtering(window_ppg, plot=args.plot, title='PPG-Butter', savepath=savepath)
            
        if args.resample:
            # Resampling to target fs: note that for ECG at least 50 Hz are required
            window_ppg = resample_signal(window_ppg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='PPG-Resampling', savepath=savepath)
            window_ecg = resample_signal(window_ecg, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='ECG-Resampling', savepath=savepath)
            if args.resp:
                window_resp = resample_signal(window_resp, original_fs=fs, target_fs=args.target_resample_fs, plot=args.plot, title='RESP-Resampling', savepath=savepath)
        
        # Zero-mean standardization (Z-score)
        window_ppg = standardize(window_ppg, plot=args.plot, title='PPG-Z-Score', savepath=savepath)
        window_ecg = standardize(window_ecg, plot=args.plot, title='ECG-Z-Score', savepath=savepath)
        if args.resp:
            window_resp = standardize(window_resp, plot=args.plot, title='RESP-Z-Score', savepath=savepath)
        
        # VPG/APG calculation
        window_vpg, window_apg = calculate_differences(window_ppg, plot=args.plot, savepath=savepath)
        
        # Z-score on VPG and APG
        window_vpg = standardize(window_vpg, plot=args.plot, title='VPG-Z-Score', savepath=savepath)
        window_apg = standardize(window_apg, plot=args.plot, title='APG-Z-Score', savepath=savepath)
                
        if args.ppg_emd:
            # Empirical Mode Decomposition
            imfs = get_emd_imfs(window_ppg, plot=args.plot, savepath=savepath)
            
        if args.scalogram:
            # Continuous Wavelet Transform to get the scalogram (note that the high/low frequencies are the one suggest from CardioID)
            ppg_freqs = scalogram(window_ppg, fs=fs, high_freq=12.5, low_freq=0.5, num_scales=16, plot=args.plot, savepath=savepath)
            
        # SBP/DBP calculation
        sbp, dbp, peaks, valleys = compute_sp_dp(sig=window_abp, fs=fs) 
                    
        if sbp < 0 or dbp < 0 or len(peaks) == 0 or len(valleys) == 0:
            print(f"\t{subject_id}|{j + 1} of {n_win} - Peaks/Valleys not found for {subject_id}, in segment {segment}, window {j} [{idx_start}:{idx_stop}]")
        else:
            window_abp = window_abp.astype(np.float32)
            sbp = sbp.astype(np.float32)
            dbp = dbp.astype(np.float32)
            
            window_ppg = window_ppg.astype(np.float32)
            window_vpg = window_vpg.astype(np.float32)
            window_apg = window_apg.astype(np.float32)
            
            window_ecg = window_ecg.astype(np.float32)
            if args.resp:
                window_resp = window_resp.astype(np.float32)
            
            # N.B.: we calculate EMD and Scalogram only after the z-score 
            if args.ppg_emd:
                imfs = imfs.astype(np.float32)
                
            if args.scalogram:
                ppg_freqs = ppg_freqs.astype(np.float32)
                
            # Save into subject data
            if args.ppg_emd and not args.scalogram:
                if args.resp:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_vpg, window_apg, imfs, window_ecg, window_resp, window_abp))
                    else:
                        subject_data.append((window_ppg, window_vpg, window_apg, imfs, window_ecg, window_resp, sbp, dbp))
                else:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_vpg, window_apg, imfs, window_ecg, None, window_abp))
                    else:
                        subject_data.append((window_ppg, window_vpg, window_apg, imfs, window_ecg, None, sbp, dbp))
            elif not args.ppg_emd and args.scalogram:
                if args.resp:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_vpg, window_apg, ppg_freqs, window_ecg, window_resp, window_abp))
                    else:
                        subject_data.append((window_ppg, window_vpg, window_apg, ppg_freqs, window_ecg, window_resp, sbp, dbp))
                else:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_vpg, window_apg, ppg_freqs, window_ecg, None, window_abp))
                    else:
                        subject_data.append((window_ppg, window_vpg, window_apg, ppg_freqs, window_ecg, None, sbp, dbp))
            else:
                if args.resp:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_vpg, window_apg, window_ecg, window_resp, window_abp))
                    else:
                        subject_data.append((window_ppg, window_vpg, window_apg, window_ecg, window_resp, sbp, dbp))
                else:
                    if args.sig2sig:
                        subject_data.append((window_ppg, window_vpg, window_apg, window_ecg, None, window_abp))
                    else:
                        subject_data.append((window_ppg, window_vpg, window_apg, window_ecg, None, sbp, dbp))
                    
            if args.plot:
                plot_abp(window_abp, fs=fs, peaks=peaks, valleys=valleys, title=f'ABP [SBP {sbp:.2f} - DBP {dbp:.2f}]', save_path=savepath)
                if args.resp:
                    if args.sig2sig:
                        plot_signals(
                            [window_ppg, window_vpg, window_apg, window_ecg, window_resp, window_abp], 
                            labels=['PPG', 'VPG', 'APG', 'ECG', 'RESP', 'ABP'], 
                            title=f'Sample Input Signals', 
                            savepath=savepath, 
                            ylabels=['a.u.', 'a.u.', 'a.u.', 'mV', 'pm', 'mmHg']
                            )
                    else:
                        plot_signals(
                            [window_ppg, window_vpg, window_apg, window_ecg, window_resp], 
                            labels=['PPG', 'VPG', 'APG', 'ECG', 'RESP'], 
                            title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                            savepath=savepath, 
                            ylabels=['a.u.', 'a.u.', 'a.u.', 'mV', 'pm']
                            )
                else:
                    if args.sig2sig:
                        plot_signals(
                            [window_ppg, window_vpg, window_apg, window_ecg, window_abp], 
                            labels=['PPG', 'VPG', 'APG', 'ECG', 'ABP'], 
                            title=f'Sample Input Signals', 
                            savepath=savepath, 
                            ylabels=['a.u.', 'a.u.', 'a.u.', 'mV', 'mmHg']
                            )
                    else:
                        plot_signals(
                            [window_ppg, window_vpg, window_apg, window_ecg], 
                            labels=['PPG', 'VPG', 'APG', 'ECG'], 
                            title=f'Sample Input Signals ~ SBP {sbp} - DBP {dbp}', 
                            savepath=savepath, 
                            ylabels=['a.u.', 'a.u.', 'a.u.', 'mV']
                            )
                # No need to do the preprocessing of all subjects when plot true
                exit()

def process_subject(subject_id, args, savepath, result_queue=None):
    
    # Load annotation
    subject_abps = np.load(os.path.join(args.input_folder, 'abp', f'{subject_id}_abp.npy'))
    
    # Load input signals
    subject_ppgs = np.load(os.path.join(args.input_folder, 'ppg', f'{subject_id}_ppg.npy'))
    subject_ecgs = np.load(os.path.join(args.input_folder, 'ecg', f'{subject_id}_ecg.npy'))
    if args.resp:
        subject_resps = np.load(os.path.join(args.input_folder, 'resp', f'{subject_id}_resp.npy'))
    
    subject_data = []
    fs = args.fs
    window_length = args.window_length # seconds
    window_overlap = args.window_overlap # overlap in seconds
    
    if len(subject_abps.shape) == 2:
        # Each subject has 30 segments with the required signals
        for segment in range(subject_abps.shape[0]):
            
            # Get annotation segment
            abp = subject_abps[segment, :]
            
            # Get input signal segments
            ppg = subject_ppgs[segment, :]
            ecg = subject_ecgs[segment, :]
            if args.resp:
                resp = subject_resps[segment, :]

            process_windows(
                abp=abp,
                ppg=ppg,
                ecg=ecg,
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
    elif len(subject_abps.shape) == 1:
        # Each subject has 1 segment with the required signals
        process_windows(
                abp=subject_abps,
                ppg=subject_ppgs,
                ecg=subject_ecgs,
                resp=subject_resps if args.resp else None,
                subject_id=subject_id,
                subject_data=subject_data,
                segment=0,
                fs=fs,
                window_length=window_length,
                window_overlap=window_overlap,
                args=args,
                savepath=savepath
            )
    else:
        raise ValueError(f"Invalid number of segments for subject {subject_id}: {len(subject_abps)}")
    
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
            
            if args.ppg_emd and not args.scalogram:
                
                if args.sig2sig:
                    for window_ppg, window_vpg, window_apg, imfs, window_ecg, window_resp, window_abp in subject_data:
                        txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                        txn.put(key="{}-vpg".format(sample_id).encode(), value=window_vpg.tobytes())
                        txn.put(key="{}-apg".format(sample_id).encode(), value=window_apg.tobytes())
                        txn.put(key="{}-imfs".format(sample_id).encode(), value=imfs.tobytes())
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                        if window_resp is not None:
                            txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes())
                        txn.put(key="{}-abp".format(sample_id).encode(), value=window_abp.tobytes())
                        
                        index_by_sample_id.append((subject_id, subject_n_recording))
                        index_by_subject_id[subject_id].append(sample_id)
                        sample_id += 1
                        subject_n_recording += 1
                else:
                    for window_ppg, window_vpg, window_apg, imfs, window_ecg, window_resp, sbp, dbp in subject_data:
                        txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                        txn.put(key="{}-vpg".format(sample_id).encode(), value=window_vpg.tobytes())
                        txn.put(key="{}-apg".format(sample_id).encode(), value=window_apg.tobytes())
                        txn.put(key="{}-imfs".format(sample_id).encode(), value=imfs.tobytes())
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                        if window_resp is not None:
                            txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes())
                        txn.put(key="{}-sbp".format(sample_id).encode(), value=np.array([sbp]).tobytes())
                        txn.put(key="{}-dbp".format(sample_id).encode(), value=np.array([dbp]).tobytes())

                        index_by_sample_id.append((subject_id, subject_n_recording))
                        index_by_subject_id[subject_id].append(sample_id)
                        sample_id += 1
                        subject_n_recording += 1
                        
            elif not args.ppg_emd and args.scalogram:
                
                if args.sisg2sig:
                    for window_ppg, window_vpg, window_apg, ppg_freqs, window_ecg, window_resp, window_abp in subject_data:
                        txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                        txn.put(key="{}-vpg".format(sample_id).encode(), value=window_vpg.tobytes())
                        txn.put(key="{}-apg".format(sample_id).encode(), value=window_apg.tobytes())
                        txn.put(key="{}-ppg_freqs".format(sample_id).encode(), value=ppg_freqs.tobytes())
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                        if window_resp is not None:
                            txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes())         
                        txn.put(key="{}-abp".format(sample_id).encode(), value=window_abp.tobytes())

                        index_by_sample_id.append((subject_id, subject_n_recording))
                        index_by_subject_id[subject_id].append(sample_id)
                        sample_id += 1
                        subject_n_recording += 1
                else:
                    for window_ppg, window_vpg, window_apg, ppg_freqs, window_ecg, window_resp, sbp, dbp in subject_data:
                        txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                        txn.put(key="{}-vpg".format(sample_id).encode(), value=window_vpg.tobytes())
                        txn.put(key="{}-apg".format(sample_id).encode(), value=window_apg.tobytes())
                        txn.put(key="{}-ppg_freqs".format(sample_id).encode(), value=ppg_freqs.tobytes())
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                        if window_resp is not None:
                            txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes())         
                        txn.put(key="{}-sbp".format(sample_id).encode(), value=np.array([sbp]).tobytes())
                        txn.put(key="{}-dbp".format(sample_id).encode(), value=np.array([dbp]).tobytes())

                        index_by_sample_id.append((subject_id, subject_n_recording))
                        index_by_subject_id[subject_id].append(sample_id)
                        sample_id += 1
                        subject_n_recording += 1
                        
            else:
                if args.sig2sig:
                    for window_ppg, window_vpg, window_apg, window_ecg, window_resp, window_abp in subject_data:
                        txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                        txn.put(key="{}-vpg".format(sample_id).encode(), value=window_vpg.tobytes())
                        txn.put(key="{}-apg".format(sample_id).encode(), value=window_apg.tobytes())
                        txn.put(key="{}-ecg".format(sample_id).encode(), value=window_ecg.tobytes())
                        if window_resp is not None:
                            txn.put(key="{}-resp".format(sample_id).encode(), value=window_resp.tobytes()) 
                        txn.put(key="{}-abp".format(sample_id).encode(), value=window_abp.tobytes())
                        
                        index_by_sample_id.append((subject_id, subject_n_recording))
                        index_by_subject_id[subject_id].append(sample_id)
                        sample_id += 1
                        subject_n_recording += 1
                else:
                    for window_ppg, window_vpg, window_apg, window_ecg, window_resp, sbp, dbp in subject_data:
                        txn.put(key="{}-ppg".format(sample_id).encode(), value=window_ppg.tobytes())
                        txn.put(key="{}-vpg".format(sample_id).encode(), value=window_vpg.tobytes())
                        txn.put(key="{}-apg".format(sample_id).encode(), value=window_apg.tobytes())
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
    parser = argparse.ArgumentParser(description="MIMIC III Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_mimic_iii', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--fs', default=125, type=int, help='the sampling frequency')
    parser.add_argument('--window_length', default=5, type=int, help='analysis window length in seconds')
    parser.add_argument('--window_overlap', default=3.0, type=float, help='window overlapping in seconds')
    parser.add_argument('--resp', default='True', type=lambda x: bool(strtobool(x)), help='whether to load resp or not')
    parser.add_argument('--resample', default='False', type=lambda x: bool(strtobool(x)), help='whether to resample the PPG or not')
    parser.add_argument('--target_resample_fs', default=50, type=float, help='target resampling frequency')
    parser.add_argument('--butterworth_filter', default='False', type=lambda x: bool(strtobool(x)), help='whether to smooth PPG with the Butterworth Filter or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to calcualte empirical mode decompoistion for PPG (4 channels) or not')
    parser.add_argument('--scalogram', default='False', type=lambda x: bool(strtobool(x)), help='whether to calcualte the scalogram for PPG or not')
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
    
    # Check that the preprocessing pipeline is used in the proper mannner
    if args.ppg_emd and args.scalogram:
        raise ValueError('Returning both EMD of PPG and Scalogram of PPG is not possible (yet)')
    
    preprocess_dataset(args)
    
    #if args.sig2sig:
    #   print("Checking for invalid windows in the dataset...")
    #    check_invalid_window_abp(args)
    #    print("Preprocessing and wabp checking completed successfully.")
