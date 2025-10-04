import numpy as np
import torch
import pickle
from online_dataset import OnlineDatasetBase  # <-- replace with your actual import path

# === 1. Initialize dataset ===
# NOTE: OnlineDatasetBase must be temporarily modified to return all the input/output modalities 
dataset = OnlineDatasetBase(
    seed=42,
    lmdb_folder="./lmdb/vital_db_pulse_db", 
    fs=125,
    input_seq_len_s=10,
    ecg=True,          # or False depending on what you want
    sig2sig=True
)

# === 2. Select the specific subject ===
subject_id = "p001326"

if subject_id not in dataset.index_by_subject_id:
    raise ValueError(f"Subject {subject_id} not found in dataset!")

# Get all sample indices for this subject
sample_indices = dataset.index_by_subject_id[subject_id]
print(f"Found {len(sample_indices)} samples for subject {subject_id}")

# === 3. Read and collect all samples ===
ppg_list = []
ecg_list = []
abp_list = []
sbp_list = []
dbp_list = []
map_list = []
timestamps_list = []

for idx in sample_indices:
    ppg, ecg, abp, sbp, dbp, map, timestamp = dataset[idx]
    ppg_list.append(ppg)
    ecg_list.append(ecg)
    abp_list.append(abp)
    sbp_list.append(sbp)
    dbp_list.append(dbp)
    map_list.append(map)
    timestamps_list.append(timestamp)

# Convert lists to numpy arrays
idx_array = np.array(sample_indices, dtype=np.int32)
ppg_array = np.array(ppg_list, dtype=np.float32)
ecg_array = np.array(ecg_list, dtype=np.float32)
abp_array = np.array(abp_list, dtype=np.float32)
sbp_array = np.array(sbp_list, dtype=np.float32)
dbp_array = np.array(dbp_list, dtype=np.float32)
map_array = np.array(map_list, dtype=np.float32)
timestamps_array = np.array(timestamps_list, dtype=np.float32)

# === 4. Save everything ===
# NOTE: by running the online_dataset.py script with save_run True, it is possible to save the indexes of the runs with contiguous valid indices
# make sure patient id matches
np.savez(
    f"../notebooks/data/{subject_id}_samples.npz",
    idxs=idx_array,
    ppgs=ppg_array,
    ecgs=ecg_array,
    abps=abp_array,
    sbps=sbp_array,
    dbps=dbp_array,
    maps=map_array,
    timestamps=timestamps_array
)

print(f"Saved {len(abp_list)} samples for {subject_id} to {subject_id}_samples.npz")
