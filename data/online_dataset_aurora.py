import os
import argparse
import pickle
import pprint
import numpy as np
import pandas as pd
import lmdb
import torch
from torch.utils.data import Dataset
from preprocessing_utils.data_visualization import (
    plot_subject_sample_distribution, plot_consecutive_block_all, 
    plot_subject_annotation_blocks, plot_block_length_statistics,
    plot_pareto_frontier
)
from online_dataset import OnlineDatasetBase
        

class AuroraOnlineSubjectDataset(OnlineDatasetBase):
    def __init__(self, batch_size, num_batches, num_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_subjects_by_min_block_length(
            batch_size=batch_size,
            num_batches=num_batches,
            num_blocks=num_blocks
        )

    def filter_subjects_by_min_block_length(self, batch_size, num_batches, num_blocks):
        r"""
        Filters subjects based on:
        1. Block length sufficient to contain `num_batches * batch_size` windows.
        2. At least `num_blocks` of such blocks per subject.
        3. Each block is *trimmed* so that only the first
            `num_batches * batch_size` windows are kept.

        Parameters
        ----------
        window_length : int
            Length of each input window in seconds.
        batch_size : int
            Number of windows in each batch.
        num_batches : int
            Number of batches required for a block of a subject to be kept.
        num_blocks : int
            Minimum number of blocks required for a subject to be kept.
        """
        subjects_to_remove = []

        print(f"[Filtering] Required windows per batch: {batch_size}")
        print(f"[Filtering] Required batches per block: {num_batches}")
        print(f"[Filtering] Required blocks per subject: {num_blocks}")

        for subj in list(self.subjects_for_personalization):

            sample_ids = list(self.index_by_subject_id[subj])
            
            # Remove subjects with insufficient windows for a batch
            if len(sample_ids) < batch_size:
                subjects_to_remove.append(subj)
                continue

            # 1) AuroraDB does not have contiguous windows, thereby we load samples a single block
            # -> we assume each 10s-window is not timestamp-contiguous with the next one
            # -> we assume that preprocessing collects sample and their IDs in chronological order
            # thereby the sample_ids reflect the chronological order

            valid_blocks = []
            kept_sample_ids = []
            
            # 2) Keep only blocks long enough AND trim them
            # We iterate from 0 to the end of the list, jumping by batch_size
            for i in range(0, len(sample_ids), batch_size):
                # Calculate the slice
                batch_slice = sample_ids[i : i + batch_size]
                
                # Check if the slice is a full batch (trimming the remainder)
                if len(batch_slice) == batch_size:
                    valid_blocks.append({
                        "block_idx": len(valid_blocks), # Sequential index starting from 0
                        "sample_ids": batch_slice
                    })
                    kept_sample_ids.extend(batch_slice)
                
                if len(valid_blocks) >= num_blocks:
                    break
                    
            # 3) Check if enough valid blocks exist for this subject
            if len(valid_blocks) < num_blocks:
                subjects_to_remove.append(subj)
                continue

            # 4) Keep only chronologically ordered subset of valid samples
            kept_sample_ids = sorted(set(kept_sample_ids))
            self.index_by_subject_id[subj] = kept_sample_ids

        # Remove subjects without enough valid blocks
        for subj in subjects_to_remove:
            if subj in self.index_by_subject_id:
                del self.index_by_subject_id[subj]
            if subj in self.subjects_for_personalization:
                self.subjects_for_personalization.remove(subj)

        print(f"[Filtering] Subjects remaining: {len(self.subjects_for_personalization)} (removed {len(subjects_to_remove)})")


    def get_subject_blocks(self, subject_id, batch_size: int = 16, num_batches: int = 3, num_blocks: int = 3):
        r"""
        Build subject blocks using fixed interleaved adaptation/validation windows.

        Parameters
        ----------
        subject_id : int
            Subject identifier from the dataset.
        batch_size : int, optional
            Number of windows per batch.
        num_batches : int, optional
            Number of batches per block.
        num_blocks : int, optional
            Number of blocks per subject.

        Returns
        -------
        blocks : list of dict
        """
        
        # Get samples of a subject
        index_list = list(self.index_by_subject_id[subject_id])

        # Collect valid block indices
        blocks = []
        for i in range(0, len(index_list), batch_size):
            batch_slice = index_list[i : i + batch_size]
            
            if len(batch_slice) == batch_size:
                blocks.append({
                    "block_idx": len(blocks), # Sequential index starting from 0
                    "sample_ids": batch_slice
                })
                
        # Skip blocks shorter than required minimum (should not happen as get_subject_blocks should be called after the initial filtering of subjects)
        if len(blocks) < num_blocks:
            raise ValueError(f"Subject {subject_id} has only {len(blocks)} valid blocks, but {num_blocks} are required.")
        
        for block in blocks:
            # block['sample_ids'] is already chronologically ordered
            block_samples = block['sample_ids']
            block_len = len(block_samples)

            # Skip blocks shorter than required minimum (should not happen)
            if block_len < batch_size:
                raise ValueError(f"Block length {block_len} is shorter than minimum required {num_batches * batch_size}")

            if len(block_samples) < batch_size:
                raise ValueError(f"Batch {block['block_idx']} (len {len(block_samples)}) is shorter than minimum required {batch_size}")

            yield {
                "subject_id": subject_id,
                "block_idx": block['block_idx'],
                "sample_ids": block_samples
            }
    
    def __del__(self):
        """
        Magic method called when the object is destroyed.
        Explicitly close the LMDB environment and transaction.
        """
        # Check if attributes exist to avoid errors during partial initialization
        if hasattr(self, 'lmdbtxn'):
            self.lmdbtxn = None 
        if hasattr(self, 'lmdbenv') and self.lmdbenv is not None:
            self.lmdbenv.close()
            self.lmdbenv = None


def pareto_frontier(df):
    r"""
    Identifies the Pareto frontier (non-dominated set) from a DataFrame based on 
    three maximization objectives.

    A row is considered "dominated" if there exists another row that is at least 
    as good in all objectives and strictly better in at least one. This function 
    is essential for multi-objective decision making, such as finding the best 
    performing models across multiple clinical metrics.

    Objectives (all maximized):
    1.  **N_patients**: Data volume or subject count.
    2.  **A**: First performance metric (e.g., SBP Accuracy).
    3.  **B**: Second performance metric (e.g., DBP Accuracy).

    Parameters
    ------------
    df (pd.DataFrame): 
        Input data containing at least the columns "N_patients", "A", and "B".

    Returns
    ------------
    output param 1 (pd.DataFrame):
        A DataFrame containing only the points that lie on the Pareto frontier.
    """
    pareto_points = []

    for _, row in df.iterrows():
        dominated = False

        for _, other in df.iterrows():
            if (
                other["N_patients"] >= row["N_patients"] and
                other["A"] >= row["A"] and
                other["B"] >= row["B"] and
                (
                    other["N_patients"] > row["N_patients"] or
                    other["A"] > row["A"] or
                    other["B"] > row["B"]
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto_points.append(row)

    return pd.DataFrame(pareto_points)


def parseargs():
    parser = argparse.ArgumentParser(description="OnlineSubjectDataset")

    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--save_path', default='./data_figs', type=str, help='where to save graphs from dataset analysis')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='plot dataset overview or not (# subjects per pretraining/personalization steps, # samples in pretraining splits)')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='whether to load only ecg or not')
    parser.add_argument('--save_run', action=argparse.BooleanOptionalAction, default=False, help='whether to save a specific subject run to a pickle dict or not')
    parser.add_argument('--personalization_batch_size', default=16, type=int, help='batch size during personalization (number of windows in a batch)')
    parser.add_argument('--num_batches', default=2, type=int, help='required number of batches with timestamp-contiguous windows')
    parser.add_argument('--num_blocks', default=2, type=int, help='required number of blocks with timestamp-contiguous windows per subject')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of loader workers')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parseargs()

    root_figs_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)
    
    # Instantiate AuroraOnlineSubjectDataset (the base for continual learning)
    online_physio_dataset = AuroraOnlineSubjectDataset(
        seed=args.seed,
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        savepath=root_figs_folder,
        batch_size=args.personalization_batch_size,
        num_batches=args.num_batches,
        num_blocks=args.num_blocks
    )
    
    # Notice that at this point subjects with insufficient runs have been removed
    subject_id = online_physio_dataset.subjects_for_personalization[1] # random.choice(list(online_physio_dataset.index_by_subject_id))
    print(f"Subject selected {subject_id}")
    
    if args.save_run:
        import pickle
        blocks = online_physio_dataset.get_subject_blocks(
            subject_id, 
            window_length=args.input_seq_len_s, 
            batch_size=args.personalization_batch_size,
            num_batches=args.num_batches,
            num_blocks=args.num_blocks
        )
    
        # Save the list of dictionaries, namely runs
        with open(f"../notebooks/data/{subject_id}_blocks.pkl", "wb") as f:
            pickle.dump(blocks, f)
            
        print(f"Saved {subject_id} runs to ../notebooks/data/{subject_id}_blocks.pkl, exiting")
        exit()

    # Compute runs first
    all_blocks = []
    for subj in online_physio_dataset.subjects_for_personalization:
        blocks = online_physio_dataset.get_subject_blocks(
            subject_id=subj,
            batch_size=args.personalization_batch_size,
            num_batches=args.num_batches,
            num_blocks=args.num_blocks
        )
        all_blocks.append(blocks)

    # Plot the Annotation statistics for the runs
    blocks = online_physio_dataset.get_subject_blocks(
        subject_id, 
        batch_size=args.personalization_batch_size,
        num_batches=args.num_batches,
        num_blocks=args.num_blocks
    )
    
    for block in blocks:
            
        block_idx = block['block_idx']
        sample_ids = block['sample_ids']
        print(f"Block {block_idx}")
        print(f"Sample IDs {sample_ids} ({len(sample_ids)} samples)")
        
        for sample_id in sample_ids:
            signals, targets_,  _ = online_physio_dataset[sample_id]
            
            print(f"\tSample ID: {sample_id}")
            print(f"\tSignals shape: {signals.shape}, Targets shape: {targets_.shape} - SBP: {targets_[0]}, DBP: {targets_[1]}, MAP: {targets_[2]}")