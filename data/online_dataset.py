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


class OnlineDatasetBase(Dataset): 
    '''No inheritance from PhysioDataset to avoid openinng the LMDB environment two times (would raise errors)'''
    def __init__(self,
                 seed, 
                 lmdb_folder, 
                 fs=125, 
                 input_seq_len_s=10, 
                 ecg=False, 
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
        r"""
        Performs data integrity checks and balancing on the subject list to avoid
        run-time errors during pre-training data loading.
        
        Parameters
        ------------
        min_subject_sample_number (int, optional): 
            The minimum required number of signal windows a subject must have to remain 
            in the dataset. If greater than 0, subjects exceeding this number will 
            be truncated to this length for data balancing. Defaults to 0.

        Returns
        ------------
        None: 
            The method modifies the instance attributes `self.subjects_for_pretraining` 
            and `self.index_by_subject_id` in-place.
        """
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
        return len(self.index_by_sample_id)

    def __getitem__(self, index):
        
        # ---------------------------------------------------------
        # === Load raw input signals (PPG/ECG) ===
        # ---------------------------------------------------------
        ppg = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-ppg".encode()), dtype="float32"))
        ecg = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-ecg".encode()), dtype="float32"))

        # ---------------------------------------------------------
        # === Load annotations ===
        #   - SBP, DBP, MAP values
        # ---------------------------------------------------------
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
    def __init__(self, batch_size, num_batches, num_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_subjects_by_min_block_length(
            window_length=kwargs['input_seq_len_s'], 
            batch_size=batch_size,
            num_batches=num_batches,
            num_blocks=num_blocks
        )

    def find_consecutive_blocks(self, sample_list, window_length):
        r"""
        Identifies and groups segments of samples that form a continuous, 
        uninterrupted chronological sequence.

        Parameters
        ------------
        sample_list (list): 
            A collection of sample IDs (integers) to be checked for continuity.
            
        window_length (float): 
            The expected temporal duration of a single data window in seconds.

        Returns
        ------------
        output param 1 (list):
            A list of dictionaries, where each dictionary represents a continuous block. 
            Keys include 'start_idx', 'end_idx', 'length', 'start_time', 'end_time', 
            and the ordered 'sample_ids'.
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

        # Group into blocks: whenever gap > threshold we cut the block
        blocks = []
        block_start = 0
        for i, gap in enumerate(diffs):
            if gap > threshold:
                block_end = i
                blocks.append({
                    "start_idx": block_start,
                    "end_idx": block_end,
                    "length": block_end - block_start + 1,
                    "start_time": float(sorted_starts[block_start]),
                    "end_time": float(sorted_ends[block_end]),
                    "sample_ids": sorted_ids[block_start:block_end+1]
                })
                block_start = i + 1

        # Add last block if it exists
        if block_start <= len(sorted_ids) - 1:
            blocks.append({
                "start_idx": block_start,
                "end_idx": len(sorted_ids) - 1,
                "length": len(sorted_ids) - block_start,
                "start_time": float(sorted_starts[block_start]),
                "end_time": float(sorted_ends[-1]),
                "sample_ids": sorted_ids[block_start:]
            })

        return blocks

    def filter_subjects_by_min_block_length(self, window_length, batch_size, num_batches, num_blocks):
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

            # 1) Identify raw blocks of consecutive windows for this subject (without trimming yet)
            blocks = self.find_consecutive_blocks(sample_ids, window_length)

            valid_blocks = []
            kept_sample_ids = []

            # 2) Keep only blocks long enough AND trim them
            for b in blocks:
                if b["length"] >= batch_size:

                    # --- trim block to first num_batches * batch_size windows ---
                    trimmed_samples = b["sample_ids"][:num_batches * batch_size]
                    
                    if len(trimmed_samples) < (num_batches * batch_size):
                        continue

                    valid_blocks.append(trimmed_samples)
                    kept_sample_ids.extend(trimmed_samples)

                    # Maintain only required number of valid blocks
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


    def get_subject_blocks(self, subject_id, window_length, batch_size: int = 16, num_batches: int = 3, num_blocks: int = 3):
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
        blocks = self.find_consecutive_blocks(index_list, window_length)
        
        # Skip blocks shorter than required minimum (should not happen as get_subject_blocks should be called after the initial filtering of subjects)
        if len(blocks) < num_blocks:
            raise ValueError(f"Subject {subject_id} has only {len(blocks)} valid blocks, but {num_blocks} are required.")
        
        for block_idx, block in enumerate(blocks):
            # b['sample_ids'] is already chronologically ordered
            block_samples = block['sample_ids']
            block_len = len(block_samples)

            # Skip blocks shorter than required minimum (should not happen as get_subject_blocks should be called after the initial filtering of subjects)
            if block_len < (num_batches * batch_size):
                raise ValueError(f"Block length {block_len} is shorter than minimum required {num_batches * batch_size}")

            # No worries on the //, see filter_subjects_by_min_block_length where blocks are already trimmed to the required length
            n_batches = block_len // batch_size 

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = start + batch_size

                batch = block_samples[start:end]

                if len(batch) < batch_size:
                    raise ValueError(f"Batch {batch_idx} (len {len(batch)}) is shorter than minimum required {batch_size}")

                yield {
                    "subject_id": subject_id,
                    "block_idx": block_idx,
                    "batch_idx": batch_idx,
                    "sample_ids": batch
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
    
    # Instantiate OnlinePhysioDataset (the base for continual learning)
    online_physio_dataset = OnlineSubjectDataset(
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
        index_list = list(online_physio_dataset.index_by_subject_id[subj])
        blocks = online_physio_dataset.find_consecutive_blocks(index_list, args.input_seq_len_s)
        all_blocks.append(blocks)

    # Plot distribution across all subjects
    plot_consecutive_block_all(all_blocks, savepath=os.path.join(root_figs_folder, f'all_subjects_block_lengths_{args.personalization_batch_size}_{args.num_batches}_{args.num_blocks}.jpg'))

    # Plot the Annotation statistics for the runs
    blocks = online_physio_dataset.get_subject_blocks(
        subject_id, 
        window_length=args.input_seq_len_s, 
        batch_size=args.personalization_batch_size,
        num_batches=args.num_batches,
        num_blocks=args.num_blocks
    )
    
    plot_subject_annotation_blocks(online_physio_dataset, subject_id, blocks, savepath=os.path.join(root_figs_folder, f"subject_{subject_id}_annotation_blocks.png"), show_bp_plot=args.plot)
    
    # Plot run length statistics across subjects
    plot_block_length_statistics(online_physio_dataset, savepath=root_figs_folder, keep_longest=False)
    
    # Trigger __del__
    del online_physio_dataset
    
    # Choose a suitable value for the number of runs and number of train/val per run so that the number of aptietns is maximized
    # -> find the pareto-frontier
    A_values = range(1, 25)      # num blocks (abrupt shifts)
    B_values = range(1, 21)      # num batches (consecutive batches)

    results = []
    
    for A in A_values:
        for B in B_values:

            args.num_blocks = A
            args.num_batches = B

            block_length = args.num_batches * args.personalization_batch_size
            
            online_physio_dataset = OnlineSubjectDataset(
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

            N = len(online_physio_dataset.subjects_for_personalization)

            results.append({
                "A": A,
                "B": B,
                "N_patients": N
            })
            
            # Trigger __del__
            del online_physio_dataset

    
    # Cvt ot dataframe
    df = pd.DataFrame(results)
    
    # Extract the apreto frontier
    pareto_df = pareto_frontier(df)
    
    feasible_df = df[df["N_patients"] >= 85]
    pareto_feasible = pareto_df[pareto_df["N_patients"] >= 85].copy()
    
    pareto_feasible["balance"] = abs(
        pareto_feasible["A"] - pareto_feasible["B"]
    )
    
    # Select config (balance A and B as much as possible, then max A)   
    selected = (
        pareto_feasible
        .sort_values(["balance", "A"], ascending=[True, False])
        .iloc[0]
    )
    
    # Abrupt-shift-stressed: maximize A, then minimize B
    abrupt_shift_stressed = (
        pareto_feasible
        .sort_values(["A", "B"], ascending=[False, True])
        .iloc[0]
    )

    # Gradual-shift-stressed: maximize B, then minimize A
    gradual_shift_stressed = (
        pareto_feasible
        .sort_values(["B", "A"], ascending=[False, True])
        .iloc[0]
    )
    
    # Plot the pareto frontier
    plot_pareto_frontier(
        df,
        pareto_feasible,
        selected,
        abrupt_shift_stressed,
        gradual_shift_stressed,
        savepath=os.path.join(
            root_figs_folder,
            "pareto_frontier_num_blocks_vs_num_batches.png"
        )
    )
    
    print("[Pareto] Selected configurations with at least 85 patients:")
    print("Mixed-Shifts Set:")
    print(selected)
    print("Abrupt-Shifts Set:")
    print(abrupt_shift_stressed)
    print("Gradual-Shifts Set:")
    print(gradual_shift_stressed)