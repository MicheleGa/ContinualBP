from collections import defaultdict
import os 
import argparse
from distutils.util import strtobool
import argparse
import numpy as np
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import PhysioDataset  


def _stack_batch(batch):
    """
    Normalize (X, Y) pairs into tensors with shapes:
      - X: (B, C, T)
      - Y: (B, T)   for sig2sig=True
      - Y: list [SBP:(B,1), DBP:(B,1)] for sig2sig=False
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
        - Sample K support windows and Q query windows from that patient
        - Return ((X_s, Y_s), (X_q, Y_q), patient_id)

    It wraps an existing PhysioDataset (so it fully reuses your LMDB layout and __getitem__).
    """
    def __init__(
        self,
        base_dataset: PhysioDataset,
        patient_ids: List[str],
        k_support: int = 8,
        k_query: int = 8,
        allow_replacement: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.ds = base_dataset
        self.patient_ids = list(patient_ids)
        self.k_support = k_support
        self.k_query = k_query
        self.allow_replacement = allow_replacement
        self.rng = np.random.default_rng(seed)

        # We rely on ds.index_by_subject_id to map patient -> list of sample_ids.
        # ds.__getitem__(sample_id) already pulls the right signals/targets from LMDB.
        self.index_by_subject_id = self.ds.index_by_subject_id
        self.bp_to_category = self.ds.bp_to_category
        
        self._build_patient_bp_index()
        
    def __len__(self):
        # Length is "virtual": number of tasks you want per epoch.
        # You can set it to len(patient_ids) to iterate each patient once per epoch,
        # or any multiple thereof. We'll do one task per patient by default.
        return len(self.patient_ids)
    
    def _build_patient_bp_index(self):
        """
        Build phenotype-aware index: for each patient, group sample_ids by BP category.
        """
        self.patient_bp_index = {}
        for pid in self.patient_ids:
            sample_ids = self.index_by_subject_id[pid]
            self.patient_bp_index[pid] = defaultdict(list)
            for sid in sample_ids:
                sbp = np.squeeze(np.frombuffer(
                    self.ds.lmdbtxn.get(f"{sid}-sbp".encode()), dtype="float32"))
                dbp = np.squeeze(np.frombuffer(
                    self.ds.lmdbtxn.get(f"{sid}-dbp".encode()), dtype="float32"))
                cat = self.bp_to_category(sbp, dbp)  # from PhysioDataset
                self.patient_bp_index[pid][cat].append(sid)


    def _sample_indices_for_patient(self, pid: str) -> Tuple[List[int], List[int]]:
        """
        Sample support/query windows for one patient,
        encouraging phenotype diversity.
        """
        total_needed = self.k_support + self.k_query
        sample_ids = self.index_by_subject_id[pid]

        if self.allow_replacement or total_needed > len(sample_ids):
            # Fallback: sample with replacement
            chosen = self.rng.choice(sample_ids, size=total_needed, replace=True)
            return list(chosen[:self.k_support]), list(chosen[self.k_support:])

        # Phenotype-aware sampling
        support_ids, query_ids = [], []
        bp_index = self.patient_bp_index[pid]

        # Flatten categories sorted by availability
        cats_sorted = sorted(bp_index.keys(), key=lambda c: -len(bp_index[c]))

        # Step 1: fill support with diverse categories
        while len(support_ids) < self.k_support and cats_sorted:
            for cat in list(cats_sorted):  # iterate over categories
                if len(support_ids) >= self.k_support:
                    break
                if bp_index[cat]:
                    sid = self.rng.choice(bp_index[cat])
                    support_ids.append(sid)
                    bp_index[cat].remove(sid)
                else:
                    cats_sorted.remove(cat)

        # Step 2: fill query with remaining, still try to diversify
        cats_sorted = sorted(bp_index.keys(), key=lambda c: -len(bp_index[c]))
        while len(query_ids) < self.k_query and cats_sorted:
            for cat in list(cats_sorted):
                if len(query_ids) >= self.k_query:
                    break
                if bp_index[cat]:
                    sid = self.rng.choice(bp_index[cat])
                    query_ids.append(sid)
                    bp_index[cat].remove(sid)
                else:
                    cats_sorted.remove(cat)

        # Step 3: If still not enough, backfill randomly
        remaining = [sid for sids in bp_index.values() for sid in sids]
        while len(support_ids) < self.k_support:
            support_ids.append(self.rng.choice(remaining))
        while len(query_ids) < self.k_query:
            query_ids.append(self.rng.choice(remaining))

        return support_ids, query_ids


    def __getitem__(self, idx):
        pid = self.patient_ids[idx % len(self.patient_ids)]
        support_ids, query_ids = self._sample_indices_for_patient(pid)

        # Pull (x,y) using the base dataset's __getitem__(sample_id)
        support_batch = [self.ds[s_id] for s_id in support_ids]
        query_batch   = [self.ds[q_id] for q_id in query_ids]
        
        Xs, Ys = _stack_batch(support_batch)
        Xq, Yq = _stack_batch(query_batch)
        
        return (Xs, Ys), (Xq, Yq), pid


def build_meta_splits_and_loaders(
    lmdb_folder: str,
    seed: int = 42,
    fs: int = 125,
    input_seq_len_s: int = 10,
    ecg: bool = False,
    sig2sig: bool = False,
    contrastive: bool = False, 
    pretraining_split_ratio=(0.7, 0.1, 0.2),
    mix_pretraining_subject_samples: bool = False,
    min_subject_sample_number: int = 0,
    loader_workers: int = 4,
    # meta/task params
    k_support: int = 8,
    k_query: int = 8,
    meta_batch_size: int = 16,
    tasks_per_epoch_train: Optional[int] = None,
    tasks_per_epoch_val: Optional[int] = None,
    tasks_per_epoch_test: Optional[int] = None,
):
    """
    Returns:
        base_dataset, meta_train_loader, meta_val_loader, meta_test_loader, split_ids
    """

    # 1) Instantiate existing PhysioDataset (reusing all its behaviors)
    base_ds = PhysioDataset(
        seed=seed,
        lmdb_folder=lmdb_folder,
        pretraining_split_ratio=list(pretraining_split_ratio),
        mix_pretraining_subject_samples=mix_pretraining_subject_samples,
        fs=fs,
        input_seq_len_s=input_seq_len_s,
        ecg=ecg,
        sig2sig=sig2sig,
        contrastive=contrastive,
        min_subject_sample_number=min_subject_sample_number,
        plot=False,
        savepath="./figs"
    )

    # 2) Use its split function to partition subjects (domain split). This fills:
    #    self.pretraining_train_subjects / val / test (when mix_pretraining_subject_samples=False)
    _ = base_ds.get_pretraining_samplers()

    if mix_pretraining_subject_samples:
        raise ValueError(
            "For meta-learning you should set mix_pretraining_subject_samples=False "
            "so that each split uses different SUBJECTS (domains)."
        )

    train_ids = base_ds.pretraining_train_subjects
    val_ids = base_ds.pretraining_val_subjects
    test_ids = base_ds.pretraining_test_subjects

    # 3) Create MetaTaskDatasets for each split
    meta_train_ds = MetaTaskDataset(
        base_dataset=base_ds,
        patient_ids=train_ids,
        k_support=k_support,
        k_query=k_query,
        allow_replacement=False,
        seed=seed,
    )
    meta_val_ds = MetaTaskDataset(
        base_dataset=base_ds,
        patient_ids=val_ids,
        k_support=k_support,
        k_query=k_query,
        allow_replacement=False,
        seed=seed,
    )
    meta_test_ds = MetaTaskDataset(
        base_dataset=base_ds,
        patient_ids=test_ids,
        k_support=k_support,
        k_query=k_query,
        allow_replacement=False,
        seed=seed,
    )

    # 4) DataLoaders: each batch = 1 task (support, query, pid).
    #    Set batch_size=1; number of tasks per epoch = len(dataset) by default (one per patient).
    def _mk_loader(ds, tasks_per_epoch, batch_size=1):
        # To cap per-epoch tasks, we can wrap the dataset so __len__ reports a custom size.
        if tasks_per_epoch is not None:
            class _LenWrap(Dataset):
                def __init__(self, base, length):
                    self.base = base; self.length = length
                def __len__(self): return self.length
                def __getitem__(self, i): return self.base[i]
            ds = _LenWrap(ds, tasks_per_epoch)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True)

    meta_train_loader = _mk_loader(meta_train_ds, tasks_per_epoch_train, batch_size=meta_batch_size)
    meta_val_loader   = _mk_loader(meta_val_ds, tasks_per_epoch_val, batch_size=1)  # keep =1 for adaptation
    meta_test_loader  = _mk_loader(meta_test_ds, tasks_per_epoch_test, batch_size=1)

    split_ids = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    return base_ds, meta_train_loader, meta_val_loader, meta_test_loader, split_ids


def parseargs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--dataset_name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--fs', type=int, default=125, help='signals frequency')
    parser.add_argument('--input_seq_len_s', type=int, default=10, help='single window duration')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--contrastive', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--k_support', type=int, default=5, help='meta-learning support set size')
    parser.add_argument('--k_query', type=int, default=10, help='meta-learning query set size')
    parser.add_argument('--meta_batch_size', type=int, default=4, help='meta batch size')
    parser.add_argument('--workers', type=int, default=2, help='parallel data loaders')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    
    args = parseargs()

    base_ds, train_loader, val_loader, test_loader, splits = build_meta_splits_and_loaders(
        lmdb_folder=os.path.join(args.dataset_folder, args.dataset_name),
        seed=args.seed,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        sig2sig=args.sig2sig,
        contrastive=args.contrastive,
        pretraining_split_ratio=(0.7, 0.1, 0.2),
        mix_pretraining_subject_samples=False,     # IMPORTANT for meta-learning to test on unseen subjects
        k_support=args.k_support,
        k_query=args.k_query,
        meta_batch_size=args.meta_batch_size,
        loader_workers=args.workers,
    )

    print(f"#Patients: train={len(splits['train_ids'])}, val={len(splits['val_ids'])}, test={len(splits['test_ids'])}")

    # --- Pull ONE TASK from train_loader and print shapes/values ---
    task_batch = next(iter(val_loader))

    (Xs, Ys), (Xq, Yq), pid = task_batch
    
    if len(Xs.shape) == 3:
        Xs = Xs.unsqueeze(-1)
    if len(Xq.shape) == 3:
        Xq = Xq.unsqueeze(-1) 
    
    print(f"[Task patient id] {pid}")
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