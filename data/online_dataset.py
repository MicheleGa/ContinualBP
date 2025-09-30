import os
import argparse
from distutils.util import strtobool
import pickle
import pprint
import random
import numpy as np
import lmdb
import torch
from torch.utils.data import Dataset
from preprocessing_utils.data_visualization import plot_subject_sample_distribution, plot_consecutive_runs_all, plot_subject_annotation_runs, plot_run_length_statistics


class OnlineDatasetBase(Dataset): # No inheritance from PhysioDataset to avoid openinng the LMDB environment two times (would raise errors)
    def __init__(self,
                 seed, 
                 lmdb_folder, 
                 fs=125, 
                 input_seq_len_s=5, 
                 ecg=False, 
                 sig2sig=False,
                 min_subject_sample_number=0, 
                 plot=False, 
                 savepath='./figs'):
        super(OnlineDatasetBase, self).__init__()
        
        # Generic arguments
        LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000 # 1T

        self.seed = seed
        self.dataset_folder = lmdb_folder
        self.min_subject_sample_number = min_subject_sample_number
        self.lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE)
        self.lmdbtxn = self.lmdbenv.begin()

        # Subject/Sample lists/dicts
        # NOTE: this expects the lmdb to contain pickled objects under these keys
        self.subjects_for_personalization:list = pickle.loads(self.lmdbtxn.get("subject_list".encode()))
        self.index_by_subject_id:dict = pickle.loads(self.lmdbtxn.get("index_by_subject_id".encode()))
        self.index_by_sample_id = pickle.loads(self.lmdbtxn.get("index_by_sample_id".encode()))
        self.check_subjects_list(min_subject_sample_number=min_subject_sample_number)
                
        # Which input data to load (PPG or PPG + ECG), PPG is always loaded
        self.ecg = ecg
        self.sig2sig = sig2sig
        self.fs = fs
        self.input_seq_len_s = input_seq_len_s
        
        # Plot arguments
        self.plot = plot
        self.savepath = savepath

        # Dataset split
        self.total_subject_n = len(self.subjects_for_personalization)
        
        if self.plot:
            if self.min_subject_sample_number > 0:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_personalization, savepath=os.path.join(savepath, f'personalization_subject_sample_distribution_min_sample_{self.min_subject_sample_number}.jpg'))
            else:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_personalization, savepath=os.path.join(savepath, 'personalization_subject_sample_distribution.jpg'))
        
        print("{:s} initialized with following configuration:".format(self.__class__.__name__))
        pprint.pprint(
            {
                "Total Subjects": len(self.subjects_for_personalization),
                "Total Samples": len(self.index_by_sample_id)
            }
        )

    def check_subjects_list(self, min_subject_sample_number=0):
        # Considering preprocessing in the mimic_iii, when a subject has no valid samples,
        # its ID is in the self.index_by_subject_id but not in the self.index_by_sample_id as the for loop inside
        # with lmdbenv.begin(write=True) as txn: deos not make this check
        invalid_subjects = list()
        # Fix: use the correct attribute self.subjects_for_personalization
        for subject in list(self.subjects_for_personalization):
            if subject not in self.index_by_subject_id or len(self.index_by_subject_id.get(subject, [])) <= min_subject_sample_number:
                invalid_subjects.append(subject)
        
        if len(invalid_subjects) > 0:
            print("Invalid subjects found in the dataset, removing them ...")
            for subject in invalid_subjects:
                if subject in self.subjects_for_personalization:
                    self.subjects_for_personalization.remove(subject)
                if subject in self.index_by_subject_id:
                    del self.index_by_subject_id[subject]

        # Shorten each subject list to min_subject_sample_number
        if min_subject_sample_number > 0:
            for subject in list(self.subjects_for_personalization):
                if len(self.index_by_subject_id.get(subject, [])) > min_subject_sample_number:
                    self.index_by_subject_id[subject] = self.index_by_subject_id[subject][:min_subject_sample_number]

    def __len__(self):
        """Returns the total number of samples across all subjects available in the dataset."""
        return len(self.index_by_sample_id)

    def __getitem__(self, index):
        
        # ---------------------------------------------------------
        # === Load raw input signals (PPG/ECG) ===
        # ---------------------------------------------------------
        ppg = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-ppg".encode()), dtype="float32"))
        ecg = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-ecg".encode()), dtype="float32"))

        # ---------------------------------------------------------
        # === Load annotations ===
        #   - Full ABP waveform
        #   - SBP, DBP, MAP values
        # ---------------------------------------------------------
        abp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-abp".encode()), dtype="float32"))
        sbp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-sbp".encode()), dtype="float32"))
        dbp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-dbp".encode()), dtype="float32"))
        map = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-map".encode()), dtype="float32"))
        
        # ---------------------------------------------------------
        # === Load timestamps ===
        # ---------------------------------------------------------
        timestamp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-timestamp".encode()), dtype="float32"))
        
        if self.ecg:
            # Shape: [time, 2]  (PPG, ECG)
            sig = np.stack((ppg, ecg), axis=-1)
        else:
            # Shape: [time, 1]  (PPG)
            sig = ppg

        # Make arrays writable
        sig = np.require(sig, requirements=['O', 'W'])
        sig.setflags(write=1)
        
        # Cast to torch tensor
        signals = torch.tensor(sig)
            
        if self.sig2sig:
            # Make arrays writable
            abp = np.require(abp, requirements=['O', 'W'])
            abp.setflags(write=1)
            
            # Cast to torch tensor
            annotation = torch.tensor(abp)
        else:
            # Make arrays writable
            sbp = np.require(sbp, requirements=['O', 'W'])
            sbp.setflags(write=1)
            dbp = np.require(dbp, requirements=['O', 'W'])
            dbp.setflags(write=1)
            map = np.require(map, requirements=['O', 'W'])
            map.setflags(write=1)
            
            # Cast to torch tensor
            annotation = torch.stack([
                torch.tensor(sbp), 
                torch.tensor(dbp), 
                torch.tensor(map)
            ], dim=-1)

        return signals, annotation, timestamp
        

class OnlineSubjectDataset(OnlineDatasetBase):
    def __init__(self, min_run_length, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_run_length = min_run_length        
        self.filter_subjects_by_min_run_length(self.min_run_length, kwargs['input_seq_len_s'])

    def find_consecutive_runs(self, sample_list, window_length):
        """
        Find runs of chronologically consecutive windows for a list of sample ids.

        The function will:
          - fetch timestamps for each sample id from the LMDB
          - sort windows by their start timestamp
          - compute time gaps between consecutive window starts
          - group windows into runs when the gap <= threshold

        Returns a list of dictionaries, each with the following keys:
          - start_idx: start index in the SORTED list
          - end_idx: end index in the SORTED list
          - length: number of windows in the run
          - start_time: float (start time of run)
          - end_time: float (end time of run)
          - sample_ids: list of sample ids (ordered chronologically) in the run
        """
        if sample_list is None:
            return []

        # Ensure a list copy
        samples = list(sample_list)
        n = len(samples)
        if n == 0:
            return []

        starts = np.zeros(n, dtype=float)
        ends = np.zeros(n, dtype=float)
        valid_samples = []
        
        # Read timestamps for each sample. Keep those with valid timestamps.
        for i, sid in enumerate(samples):
            key = f"{sid}-timestamp".encode()
            raw = self.lmdbtxn.get(key)
            if raw is None:
                # Skip samples without timestamps
                raise ValueError("Found a sample without timestamp. Exiting ...")
            ts = np.squeeze(np.frombuffer(raw, dtype="float32"))
            if ts.size == 0:
                # Skip samples with empty timestamps
                raise ValueError("Found a sample with an empty timestamp. Exiting ...")
            
            valid_samples.append((sid, float(ts[0]), float(ts[-1])))
        
        if len(valid_samples) == 0:
            return []

        # Unpack and sort by start time
        sample_ids = [x[0] for x in valid_samples]
        starts = np.array([x[1] for x in valid_samples], dtype=float)
        ends = np.array([x[2] for x in valid_samples], dtype=float)

        order = np.argsort(starts)
        sorted_ids = [sample_ids[i] for i in order]
        sorted_starts = starts[order]
        sorted_ends = ends[order]
        
        # Compute diffs between consecutive START times
        if len(sorted_starts) <= 1:
            # Single window -> a single run
            return [{
                "start_idx": 0,
                "end_idx": 0,
                "length": 1,
                "start_time": float(sorted_starts[0]),
                "end_time": float(sorted_ends[0]),
                "sample_ids": [sorted_ids[0]]
            }]

        diffs = np.diff(sorted_starts)
        # Consider only strictly positive diffs for median (just in case of duplicated timestamps)
        
        # Estimate a reasonable threshold to decide when a gap separates runs:
        # the gap exists between two consecutive windows if diff is greater then the window length plus a small delta (0.5)
        # if the end of a window and the ebginning of the next one is more then half-a second delta, then there is a gap 
        threshold = float(window_length) + 0.5

        # Group into runs: whenever gap > threshold we cut the run
        runs = []
        run_start = 0
        for i, gap in enumerate(diffs):
            if gap > threshold:
                run_end = i
                runs.append({
                    "start_idx": run_start,
                    "end_idx": run_end,
                    "length": run_end - run_start + 1,
                    "start_time": float(sorted_starts[run_start]),
                    "end_time": float(sorted_ends[run_end]),
                    "sample_ids": sorted_ids[run_start:run_end+1]
                })
                run_start = i + 1

        # Add last run
        if run_start <= len(sorted_ids) - 1:
            runs.append({
                "start_idx": run_start,
                "end_idx": len(sorted_ids) - 1,
                "length": len(sorted_ids) - run_start,
                "start_time": float(sorted_starts[run_start]),
                "end_time": float(sorted_ends[-1]),
                "sample_ids": sorted_ids[run_start:]
            })

        return runs

    def filter_subjects_by_min_run_length(self, min_run_length, window_length):
        """
        Filter subjects so that only runs of length >= min_run_length are kept.
        Subjects with fewer than 2 runs meeting the minimum length are removed entirely.

        Updates:
            - self.subject_adjacent_samples
            - self.index_by_subject_id
            - self.subjects_for_personalization

        Parameters
        ----------
        min_run_length : int
            Minimum run length to keep.
        """
        subjects_to_remove = []

        for subj in list(self.subjects_for_personalization):
            # Get all samples of a subject
            index_list = list(self.index_by_subject_id[subj])
            if len(index_list) <= min_run_length:
                subjects_to_remove.append(subj)
                continue

            # Find consecutive runs and their lengths (runs contain actual sample ids)
            runs = self.find_consecutive_runs(index_list, window_length)

            # Collect sample ids of runs that satisfy the length constraint
            keep_samples = []
            valid_runs = []
            for r in runs:
                if r["length"] >= min_run_length:
                    keep_samples.extend(r["sample_ids"])
                    valid_runs.append(r)

            # Require at least two valid runs to keep the subject
            if len(valid_runs) >= 2:
                # Update the index_by_subject_id mapping to only contain the kept (chronological) samples
                self.index_by_subject_id[subj] = keep_samples
            else:
                subjects_to_remove.append(subj)

        # Remove subjects with insufficient valid runs
        for subj in subjects_to_remove:
            if subj in self.index_by_subject_id:
                del self.index_by_subject_id[subj]
            if subj in self.subjects_for_personalization:
                self.subjects_for_personalization.remove(subj)

        print(f"Filtered dataset: {len(self.subjects_for_personalization)} subjects remain "
            f"(removed {len(subjects_to_remove)} subjects with fewer than 2 runs ≥ {min_run_length})")


    def get_subject_runs(self, subject_id, window_length, adapt_size: int = 32, val_size: int = 32, min_block_length: int = 128):
        r"""
        Build subject runs using fixed interleaved adaptation/validation windows.

        Parameters
        ----------
        subject_id : int
            Subject identifier from the dataset.
        adapt_size : int, optional
            Number of windows per adaptation (train) block. Default=32.
        val_size : int, optional
            Number of windows per validation (test) block. Default=32.
        min_block_length : int, optional
            Minimum number of windows in a run to be considered. Default=200.

        Returns
        -------
        runs : list of dict
            Each dict has:
            - "train": list of sample IDs for adaptation
            - "test": list of sample IDs for validation
            - "all": list of all sample IDs in the block (train + test)
        """
        # Get samples of a subject
        index_list = list(self.index_by_subject_id[subject_id])

        # Collect valid run indices
        runs = self.find_consecutive_runs(index_list, window_length)
        subject_runs = []

        for r_idx, r in enumerate(runs):
            # r['sample_ids'] is already chronologically ordered
            run_samples = r['sample_ids']
            run_len = len(run_samples)

            # Skip runs shorter than required minimum (should not happen as get_subject_runs should be called after the initial filtering of subjects)
            if run_len < min_block_length:
                raise ValueError(f"Run length {run_len} is shorter than minimum required {min_block_length}")

            # Segment into interleaved adaptation/testing blocks
            block_size = adapt_size + val_size
            n_blocks = run_len // block_size

            for b in range(n_blocks):
                start = b * block_size
                train_segment = run_samples[start : start + adapt_size]
                test_segment = run_samples[start + adapt_size : start + block_size]

                if len(train_segment) == adapt_size and len(test_segment) == val_size:
                    subject_runs.append({
                        "r_idx": r_idx,
                        "b_idx": b,
                        "train": train_segment,
                        "test": test_segment,
                        "all": train_segment + test_segment
                    })

        return subject_runs


def parseargs():
    parser = argparse.ArgumentParser(description="Dataset overview")

    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--save_path', default='./data_figs', type=str, help='where to save graphs from dataset analysis')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    parser.add_argument('--min_run_length', default=128, type=int, help='minimum number of samples per subject to consider it valid for doing online learning, 0 means no limit')
    parser.add_argument('--plot', default='False', type=lambda x: bool(strtobool(x)), help='plot dataset overview or not (# subjects per pretraining/personalization steps, # samples in pretraining splits)')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of loader workers')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseargs()

    root_figs_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)

    # Instantiate OnlinePhysioDataset (the base for continual learning)
    online_physio_dataset = OnlineSubjectDataset(
        seed=args.seed,
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        min_run_length=args.min_run_length,
        ecg=args.ecg,
        sig2sig=args.sig2sig,
        savepath=root_figs_folder
    )
    
    # Sucjet id notixe that subjects with insufficient runs have been removed
    subject_id = random.choice(list(online_physio_dataset.index_by_subject_id))
    print(f"Subject selected {subject_id}")
    
    # Compute runs first
    all_runs = []
    for subj in online_physio_dataset.subjects_for_personalization:
        index_list = list(online_physio_dataset.index_by_subject_id[subj])
        runs = online_physio_dataset.find_consecutive_runs(index_list, args.input_seq_len_s)
        all_runs.append(runs)

    # Plot distribution across all subjects
    plot_consecutive_runs_all(all_runs, savepath=os.path.join(root_figs_folder, 'all_subjects_run_lengths.jpg' if args.min_run_length == 0 else f'all_subjects_run_lengths_{args.min_run_length}.jpg'))

    # Plot the Annotation statistics for the runs
    runs = online_physio_dataset.get_subject_runs(subject_id, window_length=args.input_seq_len_s, adapt_size=args.batch_size, val_size=args.batch_size, min_block_length=args.min_run_length)
    
    plot_subject_annotation_runs(online_physio_dataset, subject_id, runs, savepath=os.path.join(root_figs_folder, f"subject_{subject_id}_annotation_runs.png"), show_bp_plot=args.plot)
    
    # Plot run length statistics across subjects
    plot_run_length_statistics(online_physio_dataset, savepath=root_figs_folder, keep_longest=False)
