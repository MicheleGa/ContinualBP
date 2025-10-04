# preprocess_pulse_db_parallel.py
import os
import argparse
import numpy as np
import pandas as pd
from mat73 import loadmat
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

def check_record_continuity(timestamps, tolerance_factor=2.0):
    """
    Check if patient signal records are contiguous based on timestamps.
    
    Parameters:
    npz_file_path: Path to the .npz file
    tolerance_factor: Factor to multiply normal sampling interval for gap detection
    
    Returns:
    Dictionary with continuity analysis results
    """
    
    # timestamps shape: (677, 1, 1250)
    
    print(f"Loaded data with {timestamps.shape[0]} records")
    print(f"Each record has {timestamps.shape[2]} samples")
    
    # Calculate intra-record sampling interval (should be consistent)
    first_record_intervals = np.diff(timestamps[0, 0, :])
    sampling_interval = np.mean(first_record_intervals)
    sampling_std = np.std(first_record_intervals)
    
    print(f"Intra-record sampling interval: {sampling_interval:.6f} ± {sampling_std:.6f} seconds")
    
    # Check gaps between consecutive records
    gaps = []
    discontinuities = []
    
    for i in range(len(timestamps) - 1):
        # Last timestamp of current record
        last_timestamp_current = timestamps[i, 0, -1]
        # First timestamp of next record
        first_timestamp_next = timestamps[i+1, 0, 0]
        
        # Calculate the gap
        gap = first_timestamp_next - last_timestamp_current
        gaps.append(gap)
        
        # Check if gap is significantly larger than normal sampling interval
        if gap > sampling_interval * tolerance_factor:
            discontinuities.append({
                'record_pair': (i, i+1),
                'gap': gap,
                'expected_gap': sampling_interval,
                'gap_ratio': gap / sampling_interval,
                'last_timestamp_record1': last_timestamp_current,
                'first_timestamp_record2': first_timestamp_next
            })
    
    gaps = np.array(gaps)
    
    # Statistics
    results = {
        'total_record_pairs': len(gaps),
        'discontinuities_found': len(discontinuities),
        'sampling_interval_mean': sampling_interval,
        'sampling_interval_std': sampling_std,
        'inter_record_gaps': {
            'mean': np.mean(gaps),
            'std': np.std(gaps),
            'min': np.min(gaps),
            'max': np.max(gaps),
            'median': np.median(gaps)
        },
        'discontinuities': discontinuities
    }
    
    # Print summary
    print(f"\n=== CONTINUITY ANALYSIS ===")
    print(f"Total record pairs analyzed: {len(gaps)}")
    print(f"Discontinuities found: {len(discontinuities)}")
    print(f"Inter-record gap statistics:")
    print(f"  Mean: {np.mean(gaps):.6f} seconds")
    print(f"  Std:  {np.std(gaps):.6f} seconds")
    print(f"  Min:  {np.min(gaps):.6f} seconds")
    print(f"  Max:  {np.max(gaps):.6f} seconds")
    
    if discontinuities:
        print(f"\n=== DISCONTINUITIES DETECTED ===")
        for disc in discontinuities:
            print(f"Between records {disc['record_pair'][0]} and {disc['record_pair'][1]}:")
            print(f"  Gap: {disc['gap']:.6f} seconds ({disc['gap_ratio']:.2f}x normal)")
            print(f"  Last timestamp record {disc['record_pair'][0]}: {disc['last_timestamp_record1']:.6f}")
            print(f"  First timestamp record {disc['record_pair'][1]}: {disc['first_timestamp_record2']:.6f}")
    else:
        print("✓ All records appear to be contiguous!")
    
    return results

def visualize_gaps(results, show_plot=True):
    """
    Visualize the gaps between consecutive records
    """
    gaps = []
    record_indices = []
    
    # Extract gap data
    for i in range(results['total_record_pairs']):
        if i < len(results['discontinuities']):
            # Find the gap for this record pair
            found = False
            for disc in results['discontinuities']:
                if disc['record_pair'][0] == i:
                    gaps.append(disc['gap'])
                    found = True
                    break
            if not found:
                gaps.append(results['sampling_interval_mean'])  # Approximate normal gap
        else:
            gaps.append(results['sampling_interval_mean'])  # Approximate normal gap
    
    if show_plot:
        plt.figure(figsize=(12, 6))
        
        # Plot gaps
        plt.subplot(1, 2, 1)
        plt.plot(gaps, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(y=results['sampling_interval_mean'], color='r', linestyle='--', 
                   label=f'Normal interval ({results["sampling_interval_mean"]:.6f}s)')
        plt.xlabel('Record Pair Index')
        plt.ylabel('Gap (seconds)')
        plt.title('Inter-Record Gaps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Histogram of gaps
        plt.subplot(1, 2, 2)
        plt.hist(gaps, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=results['sampling_interval_mean'], color='r', linestyle='--',
                   label=f'Normal interval')
        plt.xlabel('Gap (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Inter-Record Gaps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./analysis_results.png')


def preprocess_subject(mat_path, out_folder):
    """Convert one subject file (pXXX.mat) into compressed NumPy format (.npz)."""
    try:
        data = loadmat(mat_path)
        segs = data['Subj_Wins']
        subject_id = os.path.basename(mat_path).replace(".mat", "")
        
        abp = np.array(segs['ABP_Raw'])
        n_segments = abp.shape[0]
        
        if len(abp.shape) < 3:
            print(f"⚠️ Failed processing {mat_path}: expected more than a single segment since the ABP data is only [time series] and not [num_segments, time series].")
            return None 
        
        # Extract signals
        signals = np.concatenate([
            np.array(segs['ECG_F']),
            np.array(segs['PPG_F']),
            abp
        ], axis=1)  # (n_segments, 3, 1250)
        
        sbp = np.array(segs['SegSBP']).astype(np.float32).reshape(-1)
        dbp = np.array(segs['SegDBP']).astype(np.float32).reshape(-1)
        map = (2 * dbp + sbp) / 3

        # Signals timestamp
        timestamps = np.array(segs['T']).astype(np.float32)
        
        # Demographics
        age_data = np.array(segs['Age'])
        age = int(age_data.flatten()[0])
        gender_data = np.array(segs['Gender'])
        gender_str = gender_data.flatten()[0]
        gender = int(gender_str == 'M')
        
        # Debugging print
        #print('Out file:', out_file)
        #print('Signals shape:', signals.shape)
        #print('Timestamp shape:', timestamps.shape)
        #print('SBP shape:', sbp.shape)
        #print('DBP shape:', dbp.shape)
        #print('MAP shape:', map.shape)
        #print('Age shape:', age)
        #print('Gender shape:', gender)
        
        #results = check_record_continuity(timestamps=timestamps)
        #visualize_gaps(results)
        
        #print(f"\n=== DETAILED ANALYSIS ===")
        #if results['discontinuities_found'] == 0:
        #    print("Data appears to be fully contiguous.")
        #    continuity_percentage = 100.0
        #else:
        #    continuity_percentage = (results['total_record_pairs'] - results['discontinuities_found']) / results['total_record_pairs'] * 100
        #    print(f"Continuity: {continuity_percentage:.2f}%")
        #    print(f"Missing/discontinuous segments: {results['discontinuities_found']}")
        #    
        #    # Estimate total missing time
        #    total_missing_time = sum([disc['gap'] - results['sampling_interval_mean'] 
        #                            for disc in results['discontinuities']])
        #    print(f"Estimated total missing time: {total_missing_time:.6f} seconds")
        
        out_file = os.path.join(out_folder, subject_id + ".npz")
        
        np.savez_compressed(
            out_file,
            signals=signals.astype(np.float32),
            timestamps=timestamps,
            sbp=sbp, 
            dbp=dbp, 
            map=map.astype(np.float32),
            age=age, 
            gender=gender
        )

        return subject_id, n_segments, age, gender
    except Exception as e:
        print(f"⚠️ Failed processing {mat_path}: {e}")
        return None


def preprocess_dataset(in_folder, out_folder, index_file, n_jobs=8):
    os.makedirs(out_folder, exist_ok=True)
    subject_files = [os.path.join(in_folder, f) for f in os.listdir(in_folder) if f.endswith(".mat")]

    results = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_subject)(f, out_folder) for f in tqdm(subject_files, desc="Preprocessing")
    )

    # Drop failed
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results, columns=["subject_id", "n_segments", "age", "gender"])
    df.to_csv(index_file, index=False)
    print(f"✅ Done. {len(results)} subjects processed into {out_folder}")
    print(f"📑 Index saved at {index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="pulse_db_index.csv")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    preprocess_dataset(args.in_folder, args.out_folder, args.index_file, n_jobs=args.n_jobs)
