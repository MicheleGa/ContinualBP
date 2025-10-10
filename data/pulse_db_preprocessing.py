import os
import argparse
import pickle
import lmdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from distutils.util import strtobool
from preprocessing_utils.data_visualization import plot_signals
from preprocessing_utils.signal_processing import percentile_normalize


def process_subject(subject_info, args, result_queue=None, savepath=''):
    subject_id = subject_info['subject_id']
    n_segments = subject_info['n_segments']
    
    subject_data = []
    npz_path = os.path.join(args.input_folder, f"{subject_id}.npz")
    
    if not os.path.exists(npz_path):
        print(f"Skipping {subject_id} as the NPZ file was not found.")
        if result_queue:
            result_queue.put(None)
        return
    
    data = np.load(npz_path, allow_pickle=True)
    for segment_idx in range(n_segments):
        try:
            ecg = data["signals"][segment_idx, 0, :]  # ECG
            ppg = data["signals"][segment_idx, 1, :]  # PPG
            
            ecg = percentile_normalize(ecg, title=f'ECG Normalization for {subject_id} Segment {segment_idx}', plot=args.plot, savepath=savepath)
            ppg = percentile_normalize(ppg, title=f'PPG Normalization for {subject_id} Segment {segment_idx}', plot=args.plot, savepath=savepath)
            
            sig = np.concatenate([ecg[np.newaxis, :], ppg[np.newaxis, :]], axis=0)
            abp = data["signals"][segment_idx, 2]  # ABP
            sbp = data["sbp"][segment_idx]
            dbp = data["dbp"][segment_idx]
            map = dbp + (sbp - dbp) / 3
            timestamp = data["timestamps"][segment_idx]

            # Ensure signals are in floating point format
            sig = sig.astype(np.float32)
            abp = abp.astype(np.float32)
            sbp = np.array([sbp]).astype(np.float32)
            dbp = np.array([dbp]).astype(np.float32)
            map = np.array([map]).astype(np.float32)
            timestamp = timestamp.astype(np.float32)
            
            # The pulse_db data is already windowed
            subject_data.append((sig, abp, sbp, dbp, map, timestamp)) 
            
            if args.plot:
                plot_signals(
                    [sig[0], sig[1], abp], 
                    fs=args.fs,
                    labels=['ECG', 'PPG', 'ABP'], 
                    title=f'Sample Signals for {subject_id} Segment {segment_idx} with SBP {sbp}, DBP {dbp}, and MAP {map}', 
                    savepath=savepath, 
                    ylabels=['mV', 'a.u.', 'mmHg']
                )
                exit()

        except Exception as e:
            print(f"Error processing {subject_id} segment {segment_idx}: {e}")
            continue

    print(f'Completed {subject_id}', flush=True)
    if result_queue:
        result_queue.put((subject_id, subject_data))


def preprocess_dataset(args):
    
    # Load the subject index from the CSV file
    if not os.path.exists(args.index_file_name):
        print(f"Index file not found at {args.index_file_name}")
        return
        
    df = pd.read_csv(args.index_file_name)
    subjects_info = df.to_dict(orient="records")
    
    print(f"Found {len(subjects_info)} valid subjects to preprocess.")
    
    if args.plot:
        savepath = os.path.join(args.figs_folder, args.name)
        os.makedirs(savepath, exist_ok=True)
    else:
        savepath = args.figs_folder
        
    if args.plot:
        # Choose a random subject to plot
        subject_info = np.random.choice(subjects_info)
        
        print(f"Plotting subject {subject_info['subject_id']}")
        process_subject(subject_info, args, result_queue=None, savepath=savepath)
        
    # Dataset preprocessing
    LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000  # 1T

    lmdbenv = lmdb.open(os.path.join(args.output_folder, args.name), map_size=LMDB_MAP_SIZE)
        
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    num_threads = int(args.num_threads)
    
    # Use joblib for parallel processing
    Parallel(n_jobs=num_threads)(
        delayed(process_subject)(subject_info, args, result_queue) for subject_info in subjects_info
    )
    
    index_by_sample_id = list()
    index_by_subject_id = dict()
    subject_id_list = list()
    sample_id = 0

    with lmdbenv.begin(write=True) as txn:
        while not result_queue.empty():
            result = result_queue.get()
            if result is None:
                continue
            
            subject_id, subject_data = result
            
            if len(subject_data) == 0:
                print(f"Skipping {subject_id} as no valid data found")
                continue
            
            subject_id_list.append(subject_id)
            index_by_subject_id[subject_id] = []
            subject_n_recording = 0

            for window_sig, window_abp, window_sbp, window_dbp, window_map, window_timestamp in subject_data:
                txn.put(key="{}-ecg".format(sample_id).encode(), value=window_sig[0].tobytes()) # ECG is the first channel
                txn.put(key="{}-ppg".format(sample_id).encode(), value=window_sig[1].tobytes()) # PPG is the second channel
                txn.put(key="{}-abp".format(sample_id).encode(), value=window_abp.tobytes())
                txn.put(key="{}-sbp".format(sample_id).encode(), value=window_sbp.tobytes())
                txn.put(key="{}-dbp".format(sample_id).encode(), value=window_dbp.tobytes())
                txn.put(key="{}-map".format(sample_id).encode(), value=window_map.tobytes())
                txn.put(key="{}-timestamp".format(sample_id).encode(), value=window_timestamp.tobytes())
                
                index_by_sample_id.append((subject_id, subject_n_recording))
                index_by_subject_id[subject_id].append(sample_id)
                sample_id += 1
                subject_n_recording += 1
            
        txn.put(key="index_by_sample_id".encode(), value=pickle.dumps(index_by_sample_id))
        txn.put(key="index_by_subject_id".encode(), value=pickle.dumps(index_by_subject_id))
        txn.put(key="subject_list".encode(), value=pickle.dumps(subject_id_list))

    print('Preprocessing completed successfully', flush=True)
    
    
def parseargs():
    parser = argparse.ArgumentParser(description="PulseDB Preprocessing Pipeline")
    
    parser.add_argument('--input_folder', default='./pulse_db/mimic_iii_npz', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--index_file_name', default='./pulse_db/pulse_db_index.csv', type=str, help='name of the dataset index file')
    parser.add_argument('--name', default='mimic_iii_pulse_db', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--plot', default='False', type=lambda x: bool(strtobool(x)), help='whether to plot intermediate preprocessing steps or not')
    parser.add_argument('--fs', default=125, type=int, help='the sampling frequency')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    
    print("Parsed Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()
    
    preprocess_dataset(args)