import os
import argparse
from distutils.util import strtobool
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import lmdb
import numpy as np
from scipy.signal import resample_poly
from pyampd.ampd import find_peaks
import wfdb
from signal_processing import *


def process_subject(subject_path, subject_id, args, savepath, result_queue, required_signals=['II'], optional_signals=['RESP'], min_signal_duration=8, nans_th=0.05, flat_th=0.05):
    """Processes a single subject and returns the preprocessed data."""

    case_id = int(subject_id[1:])
    subject_data = []
    
    for recording_idx, recording in enumerate(os.scandir(subject_path)):
        
        recording_path_splitted = recording.path.split('/')
        parent_folder, patient_id, seg_id = recording_path_splitted[-3], recording_path_splitted[-2], recording_path_splitted[-1]
        seg_id = seg_id.strip() # remove the initial/endline whitespaces
        
        segment_header = wfdb.rdheader(record_name=seg_id, pn_dir=f'mimic3wdb-matched/1.0/{parent_folder}/{patient_id}')
        required_signals = set(required_signals)
        optional_signals = set(optional_signals)
        
        if required_signals.issubset(set(segment_header.sig_name)):
            sig_len_min = segment_header.sig_len / (segment_header.fs * 60)
            
            if sig_len_min >= min_signal_duration: # 8 minutes
                
                record_data = wfdb.rdrecord(record_name=seg_id, pn_dir=f'mimic3wdb-matched/1.0/{parent_folder}/{patient_id}')
                
                ecg_idx = record_data.sig_name.index('II')
                ecg = record_data.p_signal[:, ecg_idx] 
                
                if nans_percentage(ecg) <= nans_th: # no more than 5% of nans
                    ecg = interpolate_nan_pchip(ecg)
                
                    if flat_lines_detection(ecg) <= flat_th: # no more than 5% of flat linesL
                        np.save(os.path.join(recording.path, 'ecg.npy'), ecg)
                        subject_data.append(os.path.join(recording.path, 'ecg.npy'))
                        print(f'ECG saved in {recording.path}')
                    else:
                        print(f'Flat lines > 5% detected in ECG of {recording.path}')
                        continue
                else:
                    print(f'Nans > 5% detected in ECG of {recording.path}')
                    continue
                
                # Check if the optional signals are present in the recording
                if optional_signals.issubset(set(segment_header.sig_name)):
                    
                    resp_idx = record_data.sig_name.index('RESP')
                    resp = record_data.p_signal[:, resp_idx] 
                    
                    if nans_percentage(resp) <= nans_th: # no more than 5% of nans
                        resp = interpolate_nan_pchip(resp)
                        
                        if flat_lines_detection(resp) <= flat_th: # no more than 5% of flat linesL
                            np.save(os.path.join(recording.path, 'resp.npy'), resp)
                            subject_data.append(os.path.join(recording.path, 'ecg.npy'))
                            print(f'RESP saved in {recording.path}')
                        else:
                            print(f'Flat lines > 5% detected in RESP of {recording.path}')
                            continue
                    else:    
                        print(f'Nans > 5% detected in RESP of {recording.path}')
                        continue
                    
            else:
                print(f'Signal duration is less than 8 minutes in {recording.path}')
                continue
            
        else:
            print(f'Missing required signals in {recording.path}')
            continue
                
    print(f'Completed {case_id}')
    result_queue.put((case_id, subject_data))

def preprocess_dataset(args):

    subjects_path = []
    subject_ids = []
    for directory in os.scandir(args.input_folder):
        if directory.is_dir():
            for patient in os.scandir(directory):
                subjects_path.append(patient.path)
                subject_ids.append(patient.path.split('/')[-1])

    if args.plot:
        os.makedirs(f'./figs/{args.output_folder.split("_")[-1]}', exist_ok=True)
        savepath = f'./figs/{args.output_folder.split("_")[-1]}'
    else:
        savepath = './figs'

    result_queue = queue.Queue()
    
    num_threads = int(args.num_threads)  # Set the desired number of threads here

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for subject_path, subject_id in zip(subjects_path, subject_ids):
            future = executor.submit(process_subject, subject_path, subject_id, args, savepath, result_queue)
            futures.append(future)

        # Wait for all futures to complete (you can remove this if you want to process results as they come in)
        for future in futures:
            future.result()

    # Open the two .txt files for writing
    with open('patients_with_ecg.txt', 'w') as file1, open('patients_with_ecg_and_resp.txt', 'w') as file2:
        
        while not result_queue.empty():
            _, subject_data = result_queue.get()
            
            for path in subject_data:

                # Write results to the files
                if 'ecg.npy' in path:
                    file1.write(path + '\n')
                elif 'resp.npy' in path:
                    file2.write(path + '\n')
                else:
                    raise ValueError('Unexpected file type')
                
    print('Preprocessing completed successfully')
    
    
def parseargs():
    parser = argparse.ArgumentParser(description="MIMIC III Preprocessing Pipeline")
    parser.add_argument('--input_folder', default='./raw_mimic_iii', type=str, help='path to raw dataset')
    parser.add_argument('--num_threads', default=5, type=int, help='number of parallel threads to use for processing')
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
