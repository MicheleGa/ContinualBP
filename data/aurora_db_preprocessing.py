import os
import argparse
import pickle
import lmdb
import numpy as np
import pandas as pd
import neurokit2 as nk
from joblib import Parallel, delayed
import multiprocessing
from preprocessing_utils.signal_processing  import (
    filter_ecg, filter_ppg, resample_signals, 
    extract_central_window, quantile_normalization,
    extract_ecg_ppg_features
)
from preprocessing_utils.data_visualization import plot_signals



def process_subject(subject_info, args, result_queue=None, savepath=''):
    r"""
    Processes all data segments for a single subject, performing signal normalization, 
    blood pressure metric calculation, and data type conversion.

    This function serves as the worker unit for the parallel preprocessing pipeline. 
    It transforms raw data loaded from NPZ files into standardized tensors ready 
    for storage in the LMDB database.

    Processing Steps:
    1.  **Signal Extraction**: Separates ECG and PPG channels from the signal matrix.
    2.  **Normalization**: Applies the specified scaling method (Z-score, Min-max, 
        or Percentile) to ensure signals are within a comparable range across subjects.
    3.  **BP Metric Derivation**: Extracts Systolic (SBP) and Diastolic (DBP) values 
        and calculates the Mean Arterial Pressure (MAP) using the standard formula.
    4.  **Type Casting**: Converts all signals to `float32` to optimize storage 
        efficiency and GPU compatibility.

    Parameters
    ------------
    subject_info (dict): 
        Metadata for the subject, including 'subject_id' and 'n_segments'.
        
    args (Namespace): 
        Configuration arguments containing normalization type, sampling rate (fs), 
        and input/output paths.
        
    result_queue (multiprocessing.Queue, optional): 
        A thread-safe queue used to return the processed `subject_data` to the 
        main process.
        
    savepath (str, optional): 
        Directory path used for saving diagnostic plots if `args.plot` is enabled.

    Returns
    ------------
    None: 
        The result is placed into the `result_queue` if provided.
    """
    subject_data = []
    subject_id = subject_info['pid']
    if subject_id.startswith('a'):
        meas_df = pd.read_csv(os.path.join(args.input_folder, 'measurements_auscultatory.tsv'), delimiter='\t')
    elif subject_id.startswith('o'):
        meas_df = pd.read_csv(os.path.join(args.input_folder, 'measurements_oscillometric.tsv'), delimiter='\t')
    else:
        raise ValueError('Undefined protocol phase: shubject id should starts with a/o (auscultatory/oscillometric)')
    
    # Get a view of the participant's measurement metadata
    ppt_meas_df = meas_df.loc[meas_df.pid==subject_id]
    ppt_meas_df['date_time'] = pd.to_datetime(ppt_meas_df['date_time'])
    ppt_meas_df = ppt_meas_df.sort_values(by='date_time')
    ppt_meas_df = ppt_meas_df.reset_index(drop=True)

    for index, row in ppt_meas_df.iterrows():
        # Extracting values by column name
        date_time = row['date_time']
        sbp = row['sbp']
        dbp = row['dbp']
        
        if pd.isna(sbp) or pd.isna(dbp) or sbp < 20 or sbp > 300 or dbp < 20 or dbp > 300:
            print(f"Skipping {subject_id} as the SBP/DBP {date_time} are not valid.")
            continue
        
        waveform_file_path = row['waveform_file_path']
        
        if pd.isna(waveform_file_path) or not os.path.exists(os.path.join(args.input_folder, waveform_file_path)):
            print(f"Skipping {subject_id} as the waveform file {date_time} was not found.")
            continue
    
        waveform_df = pd.read_csv(os.path.join(args.input_folder, waveform_file_path), delimiter='\t')
        
        # Get ECG, pressure, and optical waveforms. 
        # 1 - Note that ECG waveform will be inverted or not depending on electrode placement
        # 2 - Note that waveforms maybe longer than the target duration: 10s
        # 3 - Note that waveforms have a different sampling rate (500 Hz) than the target fs: 125Hz
        timestamp = waveform_df.t.to_numpy()
        ecg = waveform_df.ekg.to_numpy()
        ppg = waveform_df.optical.to_numpy()
        
        # Before cleaning, ECG may be inverted, check this with neurokit
        ecg, _ = nk.ecg_invert(ecg, args.fs)
        
        try:                
            # Cleaning & Filtering according to Aurora paper 
            #clean_ppg = filter_ppg(ppg, args.fs)
            #clean_ecg = filter_ecg(ecg, args.fs)
            
            # Cleaning & Filtering w/ Neurokit
            # Neurokit2 cleaning & filtering
            # -> bio_process handles cleaning (filtering) and peak detection internally
            bio_signals_seg, bio_info_seg = nk.bio_process(
                ecg=ecg, ppg=ppg, sampling_rate=args.fs
            )

            clean_ppg = bio_signals_seg['PPG_Clean'].values.copy()
            clean_ecg = bio_signals_seg['ECG_Clean'].values.copy()
                        
            # Resample after cleaning
            timestamp_rs, ecg_rs, ppg_rs = resample_signals(
                timestamp, clean_ecg, clean_ppg,
                orig_fs=args.fs, target_fs=args.target_fs
            )
            
            # Extract 10s windows from the middle of the signal
            timestamp, ecg, ppg = extract_central_window(timestamp_rs, ecg_rs, ppg_rs, args.target_fs)
            
            # Value normalization
            ecg = quantile_normalization(ecg)
            ppg = quantile_normalization(ppg)
            
            # Feature extraction
            _, handcrafted_feats, _ = extract_ecg_ppg_features(ecg, ppg, args.target_fs, args.plot, filename=f'{subject_id}_{date_time}_ecg_ppg_fiducials', save_path=savepath)
            
        except Exception as e:
            print(f"Processing failed for subject {subject_id} record {date_time}: {e}")
            continue
        
        sig = np.concatenate([ecg[np.newaxis, :], ppg[np.newaxis, :]], axis=0)
        map = dbp + (sbp - dbp) / 3
        # Ensure signals are in floating point format
        sig = sig.astype(np.float32)
        handcrafted_feats = handcrafted_feats.astype(np.float32)
        sbp = np.array([sbp]).astype(np.float32)
        dbp = np.array([dbp]).astype(np.float32)
        map = np.array([map]).astype(np.float32)
        timestamp = timestamp.astype(np.float32)
        
        subject_data.append((sig, handcrafted_feats, sbp, dbp, map, timestamp)) 
        
        if args.plot:
            plot_signals(
                [sig[0], sig[1]], 
                fs=args.target_fs,
                labels=['ECG', 'PPG'], 
                title=f'Sample Signals for {subject_id} Segment of {date_time} with SBP {sbp}, DBP {dbp}, and MAP {map}', 
                savepath=savepath, 
                ylabels=['mV', 'a.u.']
            )
            exit()

    print(f'Completed {subject_id}', flush=True)
    if result_queue:
        result_queue.put((subject_id, subject_data))


def preprocess_dataset(args):
    r"""
    Orchestrates the large-scale preprocessing of physiological signals (ECG, PPG, BP) 
    and serializes them into a high-performance LMDB database. 
    
    The pipeline follows three main stages:
    1.  **Parallel Processing**: Uses `joblib` to distribute subject-level signal 
        processing (filtering, windowing, feature extraction) across multiple threads.
    2.  **LMDB Serialization**: Collects processed windows and writes them to an 
        LMDB environment, ensuring data is stored in a memory-mapped format for 
        fast I/O.
    3.  **Indexing**: Generates and stores lookup tables (`index_by_sample_id`, 
        `index_by_subject_id`) to allow the PyTorch DataLoader to access specific 
        samples or subjects in constant time.

    Parameters
    ------------
    args (Namespace): 
        A configuration object containing:
        - `index_file_name`: CSV path containing subject metadata.
        - `output_folder`: Destination for the LMDB database.
        - `num_threads`: Number of parallel workers for processing.
        - `plot`: Boolean to trigger a sample visualization for quality control.
        - `figs_folder`: Directory to save diagnostic plots.

    Returns
    ------------
    None: 
        Results are written directly to disk in the form of an LMDB database file.
    """
    if not os.path.exists(args.index_file_name):
        # Load the subject data as dataframes, and join features/participants files on pid field.
        ppt_df = pd.read_csv(os.path.join(args.input_folder, 'participants.tsv'), delimiter='\t')
        feat_df = pd.read_csv(os.path.join(args.input_folder, 'features.tsv'), delimiter='\t')
        comb_df = ppt_df.merge(feat_df, how='left', left_on='pid', right_on='pid')
        
        # Restrict to ambulatory phase as ambulatory measurements refer to continuous or repeated monitoring 
        # while the patient goes about their daily life, typically over 24–48 hours using a wearable device
        # -> should I restrict to other phases? Need clinical experts here
        #comb_df = comb_df.loc[comb_df['phase'] == 'ambulatory']
        
        # Set up features
        indep_features = ['pid', 'age', 'gender', 'height', 'weight']
        comb_df = comb_df[indep_features]
        
        # Conversion factors
        INCHES_TO_METERS = 0.0254
        LBS_TO_KG = 0.45359237

        # Apply conversions
        comb_df['height'] = comb_df['height'] * INCHES_TO_METERS
        comb_df['weight'] = comb_df['weight'] * LBS_TO_KG
        comb_df['height'] = comb_df['height'].round(2)
        comb_df['weight'] = comb_df['weight'].round(2)
        comb_df['gender'] = (comb_df['gender'] == 'M').astype(int)
        
        # Subset dataframe to contain only the ambulatory measurements by restricting 'phase'
        comb_df.dropna(inplace=True)
        comb_df.drop_duplicates(inplace=True)
        comb_df.to_csv(args.index_file_name)
    
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
        
        print(f"Plotting subject {subject_info['pid']}")
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
    
    # Create a mapping from string PID to unique Integer ID
    # This ensures consistency: the same 'o***'/'a***' always becomes the same 'int'
    # Note: it was not possible to doing it before because o/a identifies which measurement tsv to read
    unique_pids = sorted(df['pid'].unique())
    pid_to_int = {pid: i for i, pid in enumerate(unique_pids)}
    
    # Change for pid col for consistency with PulseDB
    df['pid'] = df['pid'].map(pid_to_int)
    df = df.rename(columns={'pid': 'subject_id'})
    df.to_csv(args.index_file_name)

    # Save to LMDB
    index_by_sample_id = list()
    index_by_subject_id = dict()
    subject_id_list = list()
    sample_id = 0

    with lmdbenv.begin(write=True) as txn:
        while not result_queue.empty():
            result = result_queue.get()
            if result is None:
                continue
            
            original_subject_id, subject_data = result
            
            # Convert to integer ID for storage
            subject_id = pid_to_int[original_subject_id]
            
            if len(subject_data) == 0:
                print(f"Skipping {subject_id} as no valid data found")
                continue
            
            subject_id_list.append(subject_id)
            index_by_subject_id[subject_id] = []
            subject_n_recording = 0

            for window_sig, window_handcrafted_feats, window_sbp, window_dbp, window_map, window_timestamp in subject_data:
                txn.put(key="{}-ecg".format(sample_id).encode(), value=window_sig[0].tobytes()) # ECG is the first channel
                txn.put(key="{}-ppg".format(sample_id).encode(), value=window_sig[1].tobytes()) # PPG is the second channel
                txn.put(key="{}-handcrafted".format(sample_id).encode(), value=window_handcrafted_feats.tobytes())
                txn.put(key="{}-sbp".format(sample_id).encode(), value=window_sbp.tobytes())
                txn.put(key="{}-dbp".format(sample_id).encode(), value=window_dbp.tobytes())
                txn.put(key="{}-map".format(sample_id).encode(), value=window_map.tobytes())
                txn.put(key="{}-timestamp".format(sample_id).encode(), value=window_timestamp.tobytes())
                
                index_by_sample_id.append((subject_id, subject_n_recording))
                index_by_subject_id[subject_id].append(sample_id)
                sample_id += 1
                subject_n_recording += 1
        
        # Store the PID mapping itself in LMDB 
        # This allows you to reverse-lookup 'int -> string' later if needed
        int_to_pid = {i: pid for pid, i in pid_to_int.items()}
        txn.put(key="pid_mapping".encode(), value=pickle.dumps(int_to_pid))
        txn.put(key="index_by_sample_id".encode(), value=pickle.dumps(index_by_sample_id))
        txn.put(key="index_by_subject_id".encode(), value=pickle.dumps(index_by_subject_id))
        txn.put(key="subject_list".encode(), value=pickle.dumps(subject_id_list))

    print('Preprocessing completed successfully', flush=True)
    
    
def parseargs():
    parser = argparse.ArgumentParser(description="AuroraDB Preprocessing Pipeline")
    
    parser.add_argument('--input_folder', default='./aurora_db', type=str, help='path to raw dataset')
    parser.add_argument('--output_folder', default='./lmdb', type=str, help='path to cleaned datasets')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='path to the figures folder')
    parser.add_argument('--index_file_name', default='./aurora_db/aurora_db_index.csv', type=str, help='name of the dataset index file')
    parser.add_argument('--name', default='aurora_db_pulse_db', type=str, help='name of the processed dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='whether to plot intermediate preprocessing steps or not')
    parser.add_argument('--fs', default=500, type=int, help='the sampling frequency of the raw data')
    parser.add_argument('--target_fs', default=125, type=int, help='the target sampling frequency of the cleaned data')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    
    print("Parsed Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()
    
    preprocess_dataset(args)