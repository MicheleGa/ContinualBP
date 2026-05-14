import os 
import argparse
import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import PhysioDataset  
from preprocessing_utils.data_visualization import plot_age_gender_distribution, calculate_dataloaders_mean_std, plot_meta_dataset_run_distribution


def _stack_batch(batch):
    r"""
    Collates a list of individual samples into standardized batch tensors for 
    the deep learning model.

    Parameters
    ------------
    batch (list): 
        A list of tuples $(X, Y)$ of length $B$ (batch size), where $X$ is the 
        physiological signal and $Y$ represents the blood pressure targets.

    Returns
    ------------
    output param 1 (torch.Tensor):
        X tensor of shape $(B, C, T)$ where $B$ is batch size, $C$ is the number 
        of channels (e.g., ECG, PPG), and $T$ is the number of time steps.
        
    output param 2 (torch.Tensor):
        Y tensor of shape $(B, 3)$ representing the target Systolic, Diastolic, 
        and Mean Arterial Pressure.
    """
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    
    # Handle input X
    # If PhysioDataset gives (T, C), permute -> (C, T)
    x_tensors = []
    for x in xs:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2 and x.shape[0] == 1250:  # (T, C)
            x = x.permute(1, 0)                  # -> (C, T)
        x_tensors.append(x)
    X = torch.stack(x_tensors, dim=0)  # (B, C, T)
    
    y_tensors = []
    for y in ys:
        y = torch.as_tensor(y, dtype=torch.float32)
        if y.ndim == 2 and y.shape[0] == 1:  # (1, T)
            y = y.squeeze(0)                 # -> (T,)
        y_tensors.append(y)
    Y = torch.stack(y_tensors, dim=0)       # (B, T)
    
    return X, Y


class MetaTaskDataset(Dataset):
    """
    Each __getitem__ returns ONE TASK for meta-learning:
        - Pick a patient (domain)
        - Sample K support windows and K_query query windows chronologically
          from one contiguous run of that patient
        - Return ((X_s, Y_s), (X_q, Y_q), patient_id)
    Implementation steps:
      1) Remove patients with fewer than k_support + k_query samples
      2) Precompute chronological blocks per patient using timestamps in LMDB
      3) Remove blocks shorter than k_support + k_query
      4) Remove patients without any valid blocks
      5) At sampling time: choose a run and a chronological slice
    """
    def __init__(
        self,
        base_dataset: PhysioDataset,
        patient_ids: List[str],
        k_support: int = 8,
        k_query: int = 8,
        window_length: int = 10,   # seconds (used to determine contiguous blocks)
    ):
        super().__init__()
        self.ds = base_dataset
        self.orig_patient_ids = list(patient_ids)
        self.k_support = int(k_support)
        self.k_query = int(k_query)
        self.window_length = float(window_length)
        self.total_needed = self.k_support + self.k_query

        # Reproducible RNG (seeded from base dataset)
        # Note: when using multiple DataLoader workers you may want to reseed per worker.
        self.rng = np.random.default_rng(getattr(base_dataset, "seed"))

        # Patient → list of sample_ids (as provided by PhysioDataset)
        self.index_by_subject_id = self.ds.index_by_subject_id

        print(f"[MetaTaskDataset] Dataset initialized with {len(self.orig_patient_ids)} patients.")

        # 1) Filter patients with insufficient total samples
        filtered_patients = []
        for pid in self.orig_patient_ids:
            sample_ids = list(self.index_by_subject_id.get(pid))
            if len(sample_ids) >= self.total_needed:
                filtered_patients.append(pid)
            # else: excluded
        print(f"[MetaTaskDataset] {len(filtered_patients)} patients remain after filtering by total samples >= {self.total_needed}.")

        # 2) Precompute blocks for each remaining patient
        blocks_by_patient = {}
        for pid in filtered_patients:
            sample_ids = list(self.index_by_subject_id.get(pid))
            blocks = self.find_consecutive_blocks(sample_ids)
            
            # Filter blocks by minimum length requirement
            valid_blocks = [r for r in blocks if r["length"] >= self.total_needed]
            if len(valid_blocks) > 0:
                blocks_by_patient[pid] = valid_blocks
            # else: patient will be excluded (no blocks long enough)
        print(f"[MetaTaskDataset] {len(blocks_by_patient)} patients remain after filtering blocks by length >= {self.total_needed}.")

        # 3) Remove patients without valid blocks
        self.patient_ids = sorted([pid for pid, patient_blocks in blocks_by_patient.items() if len(patient_blocks) > 0])
        self.blocks_by_patient = {pid: blocks_by_patient[pid] for pid in self.patient_ids}
        print(f"[MetaTaskDataset] {len(self.patient_ids)} patients remain after final filtering by the number of blocks per patient.")
        
        if len(self.patient_ids) == 0:
            raise ValueError(
                f"No patients left after filtering. Need at least one patient with >= {self.total_needed} samples "
                f"and at least one run of length >= {self.total_needed}."
            )

        # For debug/visibility
        print(f"[MetaTaskDataset] Initialized with {len(self.patient_ids)} patients (k_support={self.k_support}, k_query={self.k_query}).")

    def find_consecutive_blocks(self, sample_list):
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
        sample_ids = list(sample_list)
        if len(sample_ids) == 0:
            return []

        starts = []
        ends = []

        # Load timestamps preserving sample_id order
        lmdbtxn = getattr(self.ds, "lmdbtxn")
        for sid in sample_ids:

            raw = lmdbtxn.get(f"{sid}-timestamp".encode())
            if raw is None:
                raise ValueError(f"Missing timestamp for sample {sid}")

            ts = np.squeeze(np.frombuffer(raw, dtype="float32"))
            if ts.size == 0:
                raise ValueError(f"Empty timestamp for sample {sid}")

            starts.append(float(ts[0]))
            ends.append(float(ts[-1]))

        if len(starts) == 0 or len(ends) == 0:
            return []
        
        # Preserve original ordering from sample_list
        # some subjects exhibit subsets of sample dis with different sample ids 
        # but same start time
        starts = np.array(starts)
        ends = np.array(ends)

        blocks = []

        block_start = 0

        # Remove window_length from signature, or make it drive the threshold:
        gap_threshold = 0.5  # seconds; end-to-start gap above this implies a removed-artifact boundary
        for i in range(len(sample_ids) - 1):

            current_end = ends[i]
            next_start = starts[i + 1]

            # --------------------------------------------------
            # CASE 1:
            # timestamp reset -> new monitoring session
            # --------------------------------------------------
            if next_start <= starts[i]:

                block_end = i
                blocks.append({
                    "start_idx": block_start,
                    "end_idx": block_end,
                    "length": block_end - block_start + 1,
                    "start_time": starts[block_start],
                    "end_time": ends[block_end],
                    "sample_ids": sample_ids[block_start:block_end + 1]
                })
                block_start = i + 1
                continue

            # --------------------------------------------------
            # CASE 2:
            # gap between end of current window and start of next
            # exceeds threshold → artifact removal created a discontinuity
            # --------------------------------------------------
            gap = next_start - current_end

            if gap > gap_threshold:

                block_end = i
                blocks.append({
                    "start_idx": block_start,
                    "end_idx": block_end,
                    "length": block_end - block_start + 1,
                    "start_time": starts[block_start],
                    "end_time": ends[block_end],
                    "sample_ids": sample_ids[block_start:block_end + 1]
                })
                block_start = i + 1

        # Final block
        blocks.append({
            "start_idx": block_start,
            "end_idx": len(sample_ids) - 1,
            "length": len(sample_ids) - block_start,
            "start_time": starts[block_start],
            "end_time": ends[-1],
            "sample_ids": sample_ids[block_start:]
        })

        return blocks

    def _sample_indices_for_patient(self, pid: str) -> Tuple[List[int], List[int]]:
        r"""
        Extracts a continuous chronological slice of data from a patient's record 
        to create a meta-learning task.

        The method selects a random "run" (a continuous sequence of data) from the 
        specified patient and then extracts a fixed-length window. This window is 
        partitioned into:
        1.  **Support Set**: The first $k_{support}$ samples used for rapid adaptation.
        2.  **Query Set**: The subsequent $k_{query}$ samples used to calculate the 
            meta-gradient and update the model's initial weights.

        Parameters
        ------------
        pid (str): 
            The unique patient identifier from which to sample.

        Returns
        ------------
        output param 1 (List[int]):
            support_ids: A list of sample IDs representing the "past" calibration 
            data for the task.
            
        output param 2 (List[int]):
            query_ids: A list of sample IDs representing the "future" evaluation 
            data for the task.
        """
        blocks = self.blocks_by_patient.get(pid)
        if not blocks:
            raise ValueError(f"No valid blocks for patient {pid} at sampling time.")

        # choose a run randomly among valid blocks (reproducible via self.rng)
        run = self.rng.choice(blocks)
        sample_ids = run["sample_ids"]
        run_len = len(sample_ids)
        total_needed = self.total_needed

        if run_len < total_needed:
            # This should not happen because we filtered blocks by length >= total_needed at init,
            # but double-check to be safe.
            raise ValueError(f"Chosen run is too short for patient {pid} (run_len={run_len} < needed={total_needed})")
        
        # Choose a chronological slice inside the run: start index in [0, run_len - total_needed]
        max_start = run_len - total_needed
        if max_start == 0:
            start_idx = 0
        else:
            start_idx = int(self.rng.integers(0, max_start + 1))  # inclusive of max_start

        support_ids = sample_ids[start_idx:start_idx + self.k_support]
        query_ids = sample_ids[start_idx + self.k_support:start_idx + total_needed]

        return support_ids, query_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # Map idx to a patient (wrap-around if needed)
        pid = self.patient_ids[idx % len(self.patient_ids)]
        support_ids, query_ids = self._sample_indices_for_patient(pid)

        support_batch = [self.ds[s_id] for s_id in support_ids]
        query_batch = [self.ds[q_id] for q_id in query_ids]

        Xs, Ys = _stack_batch(support_batch)  # [k_support, C, T]
        Xq, Yq = _stack_batch(query_batch)    # [k_query, C, T]

        return (Xs, Ys), (Xq, Yq), pid


def build_meta_splits_and_loaders(
    lmdb_folder: str,
    root_figs_folder: str = './',
    seed: int = 42,
    fs: int = 125,
    input_seq_len_s: int = 10,
    ecg: bool = False,
    pretraining_split_ratio=(0.7, 0.1, 0.2),
    meta_split_ratio: float = 0.2,
    drift_aware: bool = False,
    min_subject_sample_number: int = 0,
    loader_workers: int = 4,
    # meta/task params
    k_support: int = 16,
    k_query: int = 16,
    meta_batch_size: int = 4,
    plot: bool = False,
    index_file_name: str = ''
):
    r"""
    Orchestrates the creation of meta-learning data loaders by partitioning 
    subjects and wrapping them into task-based datasets.

    The function executes four primary steps:
    1.  **Base Initialization**: Creates a standard `PhysioDataset` to manage 
        the raw LMDB connections.
    2.  **Subject Partitioning**: Splits the subject pool into Train, Val, 
        and Test sets at the subject level (preventing subject leakage).
    3.  **Task Wrapping**: Wraps these subject sets into `MetaTaskDataset` 
        objects which manage the $k$-shot support/query sampling.
    4.  **Loader Construction**: Returns PyTorch DataLoaders where each 
        batch contains multiple "tasks" (one task per patient).

    Parameters
    ------------
    lmdb_folder (str): 
        Path to the LMDB database containing the signal windows.
    root_figs_folder (str): 
        Directory where diagnostic plots and distributions will be saved.
    k_support (int): 
        Number of samples in the support set (calibration) for each meta-task.
    k_query (int): 
        Number of samples in the query set (evaluation) for each meta-task.
    meta_batch_size (int): 
        Number of tasks (patients) to include in a single meta-update batch.

    Returns
    ------------
    output param 1 (PhysioDataset):
        The underlying base dataset instance.
    output param 2 (DataLoader):
        The meta-training loader providing batches of tasks.
    output param 3 (DataLoader):
        The meta-validation loader (typically batch size 1).
    output param 4 (DataLoader):
        The meta-test loader (typically batch size 1).
    output param 5 (dict):
        A dictionary containing the subject IDs assigned to each split.
    """

    # 1) Instantiate existing PhysioDataset (reusing all its behaviors)
    base_ds = PhysioDataset(
        seed=seed,
        lmdb_folder=lmdb_folder,
        pretraining_split_ratio=list(pretraining_split_ratio),
        meta_split_ratio=meta_split_ratio,
        drift_aware=drift_aware,
        fs=fs,
        input_seq_len_s=input_seq_len_s,
        ecg=ecg,
        min_subject_sample_number=min_subject_sample_number,
        plot=False,
        savepath="./figs"
    )

    # 2) Use its split function to partition subjects (domain split). This fills:
    #    supervised train & meta learning subjects / val / test 
    
    _ = base_ds.get_pretraining_samplers()

    _ = base_ds.supervised_pretrain_subjects
    meta_train_ids = base_ds.meta_learning_subjects
    val_ids = base_ds.pretraining_val_subjects
    test_ids = base_ds.pretraining_test_subjects

    # 3) Create MetaTaskDatasets for each split
    print(f"[MetaTaskDataset] Building meta train dataset ...")
    meta_train_ds = MetaTaskDataset(
        base_dataset=base_ds,
        patient_ids=meta_train_ids,
        k_support=k_support,
        k_query=k_query,
        window_length=input_seq_len_s
    )
    print(f"[MetaTaskDataset] Meta train dataset has {len(meta_train_ds)} patients.")
    
    print(f"[MetaTaskDataset] Building meta val dataset ...")
    meta_val_ds = MetaTaskDataset(
        base_dataset=base_ds,
        patient_ids=val_ids,
        k_support=k_support,
        k_query=k_query,
        window_length=input_seq_len_s
    )
    print(f"[MetaTaskDataset] Meta val dataset has {len(meta_val_ds)} patients.")
    
    print(f"[MetaTaskDataset] Building meta test dataset ...")
    meta_test_ds = MetaTaskDataset(
        base_dataset=base_ds,
        patient_ids=test_ids,
        k_support=k_support,
        k_query=k_query,
        window_length=input_seq_len_s
    )
    print(f"[MetaTaskDataset] Meta test dataset has {len(meta_test_ds)} patients.")

    # 4) DataLoaders: each batch = many tasks (meta-batch).
    #    Set shuffle=True so patient tasks are shuffled across epochs; each patient
    #    contributes equally (MetaTaskDataset has one entry per patient).
    meta_train_loader = DataLoader(meta_train_ds, batch_size=meta_batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)
    meta_val_loader = DataLoader(meta_val_ds, batch_size=1, num_workers=loader_workers, pin_memory=True)  # keep batch_size=1 for adaptation
    meta_test_loader = DataLoader(meta_test_ds, batch_size=1, num_workers=loader_workers, pin_memory=True)  # keep batch_size=1 for adaptation

    split_ids = {
        "train_ids": meta_train_ds.patient_ids,
        "val_ids": meta_val_ds.patient_ids,
        "test_ids": meta_test_ds.patient_ids
    }
    
    if plot:     
        if index_file_name != '':
            # Meta-train ds
            print(f'[MetaTaskDataset] Plot age, gender, and run length distributions of subjects from meta train ds')
            plot_age_gender_distribution(meta_train_ds.patient_ids, index_file_name, savepath=root_figs_folder, filename='meta_train_ds_')
            plot_meta_dataset_run_distribution(meta_train_ds, dataset_name="meta_train_ds", savepath=root_figs_folder)
            
            # Meta-val ds
            print(f'[MetaTaskDataset] Plot age, gender, and run length distributions of subjects from meta val ds')
            plot_age_gender_distribution(meta_val_ds.patient_ids, index_file_name, savepath=root_figs_folder, filename='meta_val_ds_')
            plot_meta_dataset_run_distribution(meta_val_ds, dataset_name="meta_val_ds", savepath=root_figs_folder)
            
            # Meta-test ds
            print(f'[MetaTaskDataset] Plot age, gender, and run length distributions of subjects from meta test ds')
            plot_age_gender_distribution(meta_test_ds.patient_ids, index_file_name, savepath=root_figs_folder, filename='meta_test_ds_')
            plot_meta_dataset_run_distribution(meta_test_ds, dataset_name="meta_test_ds", savepath=root_figs_folder)
        
            calculate_dataloaders_mean_std(
                dataloaders=[meta_train_loader, meta_val_loader, meta_test_loader], 
                dataloaders_names=['Meta-Pretraining-Train', 'Meta-Pretraining-Val', 'Meta-Pretraining-Test'], 
                savepath=root_figs_folder,
                meta_dataloader=True
            ) 
        else:
            print('No index file provided...')
    
    return base_ds, meta_train_loader, meta_val_loader, meta_test_loader, split_ids


def parseargs():
    parser = argparse.ArgumentParser(description="MetaTaskDataset")
    
    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='where to save graphs from dataset analysis')
    parser.add_argument('--dataset_name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--fs', type=int, default=125, help='signals frequency')
    parser.add_argument('--input_seq_len_s', type=int, default=10, help='single window duration')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='whether to load only ecg or not')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.15,0.15', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--meta_train_split_ratio', default=0.2, type=float, help='percentage of subjects to extract from the training subjects for meta-learning')
    parser.add_argument('--min_subject_sample_number', default=0, type=int, help='minimum number of samples per subject to consider it valid, 0 means no limit; given the support and query sample number, it is set to 15')
    parser.add_argument('--drift_aware', action=argparse.BooleanOptionalAction, default=False, help='sample training subjects according to their SBP drift over time or nots')
    parser.add_argument('--k_support', type=int, default=16, help='meta-learning support set size')
    parser.add_argument('--k_query', type=int, default=16, help='meta-learning query set size')
    parser.add_argument('--meta_batch_size', type=int, default=4, help='meta batch size')
    parser.add_argument('--workers', type=int, default=2, help='parallel data loaders')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='plot dataset overview or not')
    parser.add_argument('--index_file_name', default='', type=str, help='name of the dataset index file')
    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    
    args = parseargs()

    root_figs_folder = os.path.join(args.figs_folder, args.dataset_name) 
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)
    
    base_ds, train_loader, val_loader, test_loader, splits = build_meta_splits_and_loaders(
        lmdb_folder=os.path.join(args.dataset_folder, args.dataset_name),
        root_figs_folder=root_figs_folder,
        seed=args.seed,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        pretraining_split_ratio=tuple(map(float, args.pretraining_tr_val_tt_split_ratio.split(','))),
        meta_split_ratio=args.meta_train_split_ratio,
        drift_aware=args.drift_aware,
        k_support=args.k_support,
        k_query=args.k_query,
        meta_batch_size=args.meta_batch_size,
        loader_workers=args.workers,
        plot=args.plot,
        index_file_name=args.index_file_name
    )

    # --- Pull ONE TASK from train_loader and print shapes/values ---
    task_batch = next(iter(train_loader))

    (Xs, Ys), (Xq, Yq), pid = task_batch
    
    print(f"Task patient ids: {pid}")
    print(f"Support X shape: {tuple(Xs.shape)}")
    print(f"Support Y shape: {tuple(Ys.shape)}")
    print(f"Query X shape: {tuple(Xq.shape)}")
    print(f"Query Y shape: {tuple(Yq.shape)}")

    sX, sY = Xs[0].float(), Ys[0].float()
    qX, qY = Xq[0].float(), Yq[0].float()
    
    print(f"Support X shape: {tuple(sX.shape)}")
    print(f"Support Y shape: {tuple(sY.shape)}")
    print(f"Query X shape: {tuple(qX.shape)}")
    print(f"Query Y shape: {tuple(qY.shape)}")