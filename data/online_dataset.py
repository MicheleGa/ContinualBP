import os
import argparse
from distutils.util import strtobool
import pickle
import pprint
import numpy as np
import random
import sys
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader # Keep Dataset import for clarity and potential future base datasets
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
        self.subjects_for_personalization:list = pickle.loads(self.lmdbtxn.get("subject_list".encode()))
        self.index_by_subject_id:dict = pickle.loads(self.lmdbtxn.get("index_by_subject_id".encode()))
        self.index_by_sample_id = pickle.loads(self.lmdbtxn.get("index_by_sample_id".encode()))
        self.subject_adjacent_samples:dict = pickle.loads(self.lmdbtxn.get("subject_adjacent_samples".encode()))
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

    def check_subjects_list(self, min_subject_sample_number):
        # Considering preprocessing in the mimic_iii, when a subject has no valid samples,
        # its ID is in the self.index_by_subject_id but not in the self.index_by_sample_id as the for loop inside
        # with lmdbenv.begin(write=True) as txn: deos not make this check
        invalid_subjects = list()
        for subject in self.subjects_for_personalization:
            if len(self.index_by_subject_id[subject]) <= min_subject_sample_number:
                invalid_subjects.append(subject)
        
        if len(invalid_subjects) > 0:
            print("Invalid subjects found in the dataset, removing them ...")
            for subject in invalid_subjects:
                self.subjects_for_personalization.remove(subject)
                del self.index_by_subject_id[subject]
                del self.subject_adjacent_samples[subject]

    def before_pickle(self):
        self.lmdbenv = None
        self.lmdbtxn = None

    def __len__(self):
        """Returns the total number of samples across all subjects available in the dataset."""
        return len(self.index_by_sample_id)

    def __getitem__(self, index):
        sample = dict()

        # Input data        
        if self.ecg:
            sample['ppg'] = np.frombuffer(self.lmdbtxn.get("{}-ppg".format(index).encode()), dtype="float32")
            sample['ecg'] = np.frombuffer(self.lmdbtxn.get("{}-ecg".format(index).encode()), dtype="float32")
            
            sample['sig'] = np.concatenate(
                (
                    np.expand_dims(sample['ppg'], axis=-1), 
                    np.expand_dims(sample['ecg'], axis=-1)
                ), 
                axis=-1)
        else:
            sample['ppg'] = np.frombuffer(self.lmdbtxn.get("{}-ppg".format(index).encode()), dtype="float32")
            
            sample['sig'] = sample['ppg']
            
        # Annotation
        if self.sig2sig:
            sample['abp'] = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-abp".encode()), dtype="float32"))
            
            # Ensure arrays are writeable
            for k in sample:
                sample[k] = np.require(sample[k], requirements=['O', 'W'])
                sample[k].setflags(write=1)
            
            # Cast to torch tensor
            signals = torch.tensor(sample['sig'])
            abp = torch.tensor(sample['abp'])
            
            return signals, abp
        
        else:
            # Regression mode: return SBP/DBP/MAP
            sample['sbp'] = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-sbp".encode()), dtype="float32"))
            sample['dbp'] = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-dbp".encode()), dtype="float32"))
            sample['map'] = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-map".encode()), dtype="float32"))
            
            for k in sample:
                sample[k] = np.require(sample[k], requirements=['O', 'W'])
                sample[k].setflags(write=1)
            
            signals = torch.tensor(sample['sig'])
            sbp_val = torch.tensor(sample['sbp'])
            dbp_val = torch.tensor(sample['dbp'])
            map_val = torch.tensor(sample['map'])
            
            return signals, torch.stack([sbp_val, dbp_val, map_val], dim=-1)
        

class OnlineSubjectDataset(OnlineDatasetBase):
    def __init__(self, min_run_length, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Build mapping between "internal index" and "real subject_id"
        self.active_subject = None
        self.active_subject_samples = []
        self.sample_pointer = 0
        
        self.min_run_length = min_run_length
        
        self.filter_subjects_by_min_run_length(self.min_run_length)

    def set_active_subject(self, subject_id):
        
        self.active_subject = subject_id
        self.active_subject_samples = self.subject_adjacent_samples[subject_id]
        
        self.sample_pointer = 0

    def __len__(self):
        if self.active_subject is None:
            return 0
        return len(self.active_subject_samples)

    def find_consecutive_runs(self, sample_list):
        """
        Find runs of consecutive values in a list of sample indices.
        """
        if not sample_list:
            return []

        runs = []
        start_idx = 0

        for i in range(1, len(sample_list)):
            if sample_list[i] != sample_list[i - 1] + 1:
                run_values = sample_list[start_idx:i]
                runs.append({
                    "start_idx": start_idx,
                    "end_idx": i - 1,
                    "values": run_values,
                    "length": len(run_values)
                })
                start_idx = i

        run_values = sample_list[start_idx:]
        runs.append({
            "start_idx": start_idx,
            "end_idx": len(sample_list) - 1,
            "values": run_values,
            "length": len(run_values)
        })

        return runs

    def filter_subjects_by_min_run_length(self, min_run_length: int):
        """
        Filter subjects so that only runs of length >= min_run_length are kept.
        Subjects without any valid runs are removed entirely.

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
            sample_list = self.subject_adjacent_samples[subj]
            index_list = self.index_by_subject_id[subj]

            runs = self.find_consecutive_runs(sample_list)

            # Collect indices of valid runs
            keep_positions = []
            for r in runs:
                if r["length"] >= min_run_length:
                    keep_positions.extend(range(r["start_idx"], r["end_idx"] + 1))

            if keep_positions:
                # Keep only valid samples for this subject
                self.subject_adjacent_samples[subj] = [sample_list[i] for i in keep_positions]
                self.index_by_subject_id[subj] = [index_list[i] for i in keep_positions]
            else:
                # Mark subject for removal
                subjects_to_remove.append(subj)

        # Remove subjects with no valid runs
        for subj in subjects_to_remove:
            del self.subject_adjacent_samples[subj]
            del self.index_by_subject_id[subj]
            self.subjects_for_personalization.remove(subj)

        print(f"Filtered dataset: {len(self.subjects_for_personalization)} subjects remain "
            f"(removed {len(subjects_to_remove)} subjects with no runs ≥ {min_run_length})")


    def get_subject_runs(self, subject_id: int, adapt_size: int = 32, val_size: int = 32, min_block_length: int = 200):
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
        sample_list = self.subject_adjacent_samples[subject_id]
        index_list = self.index_by_subject_id[subject_id]

        runs = self.find_consecutive_runs(sample_list)
        subject_runs = []

        for r_idx, r in enumerate(runs):
            run_positions = range(r["start_idx"], r["end_idx"] + 1)
            run_samples = [index_list[i] for i in run_positions]
            run_len = len(run_samples)

            # Skip runs shorter than required minimum
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
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--mix_pretraining_subject_samples', default='True', type=lambda x: bool(strtobool(x)), help='whether to mix pretraining subject samples among train/val/test or not')
    parser.add_argument('--min_run_length', default=0, type=int, help='minimum number of samples per subject to consider it valid for doing online learning, 0 means no limit')
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
    
    # Sucjet id
    subject_id = 570 # alternatively also 2100 has enough windows
    print(f"Subject selected {subject_id}")
    online_physio_dataset.set_active_subject(subject_id)
    
    # Compute runs first
    all_runs = []
    for subj in online_physio_dataset.subjects_for_personalization:
        runs = online_physio_dataset.find_consecutive_runs(online_physio_dataset.subject_adjacent_samples[subj])
        all_runs.append(runs)

    # Plot distribution across all subjects
    plot_consecutive_runs_all(all_runs, savepath=os.path.join(root_figs_folder, 'all_subjects_run_lengths.jpg' if args.min_run_length == 0 else f'all_subjects_run_lengths_{args.min_run_length}.jpg'))

    # Plot the Annotation statistics for the runs
    runs = online_physio_dataset.get_subject_runs(subject_id, adapt_size=args.batch_size, val_size=args.batch_size, min_block_length=args.min_run_length)
    
    plot_subject_annotation_runs(online_physio_dataset, subject_id, runs, savepath=os.path.join(root_figs_folder, f"subject_{subject_id}_annotation_runs.png"), show_bp_plot=args.plot)
    
    # Plot run length statistics across subjects
    plot_run_length_statistics(online_physio_dataset, savepath=root_figs_folder, keep_longest=False)