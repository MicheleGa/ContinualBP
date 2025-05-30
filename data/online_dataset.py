import os
import argparse
from distutils.util import strtobool
import pickle
import pprint
import numpy as np
import sys
import lmdb
import torch
from torch.utils.data import Dataset
from preprocessing_utils.data_visualization import plot_subject_validity_over_time


class OnlinePhysioDataset(Dataset):
    def __init__(self,
                 lmdb_folder,
                 fs=125,
                 input_seq_len_s=5,
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 ppg_derivatives=False,
                 ppg_emd=False,
                 ppg_freqs=False,
                 min_subject_sample_number=0):
        super(OnlinePhysioDataset, self).__init__()

        LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000 # 1T

        self.dataset_folder = lmdb_folder
        self.min_subject_sample_number = min_subject_sample_number
        
        # Open LMDB in readonly mode with no locks for safe concurrent access across processes
        self.lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE, readonly=True, lock=False)
        self.lmdbtxn = self.lmdbenv.begin() # Start a read transaction

        # Load metadata
        self.subject_list: list = pickle.loads(self.lmdbtxn.get("subject_list".encode()))
        self.index_by_subject_id: dict = pickle.loads(self.lmdbtxn.get("index_by_subject_id".encode())) # Maps subject_id to list of LMDB keys (strings)

        # Filter subjects with insufficient samples
        self._check_subjects_list(min_subject_sample_number=min_subject_sample_number)

        # Configure signal loading
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.ppg_freqs = ppg_freqs
        self.fs = fs
        self.input_seq_len_s = input_seq_len_s
        self.sample_length_in_samples = self.fs * self.input_seq_len_s

        print("{:s} initialized for online learning with following configuration:".format(self.__class__.__name__))
        pprint.pprint(
            {
                "LMDB Folder": lmdb_folder,
                "Sampling Frequency (Hz)": fs,
                "Input Sequence Length (s)": input_seq_len_s,
                "Include ECG": ecg,
                "Include RESP": resp,
                "Signal-to-Signal Prediction": sig2sig,
                "Include PPG Derivatives": ppg_derivatives,
                "Include PPG EMD": ppg_emd,
                "Include PPG Freqs": ppg_freqs,
                "Min Subject Sample Number": min_subject_sample_number,
                "Total Subjects Available": len(self.subject_list),
            }
        )

    def _check_subjects_list(self, min_subject_sample_number=0):
        """Removes subjects with less than min_subject_sample_number samples."""
        invalid_subjects = []
        for subject in self.subject_list:
            if len(self.index_by_subject_id.get(subject, [])) < min_subject_sample_number:
                invalid_subjects.append(subject)

        if invalid_subjects:
            print(f"Removing {len(invalid_subjects)} invalid subjects from the dataset (less than {min_subject_sample_number} samples each).")
            for subject in invalid_subjects:
                self.subject_list.remove(subject)
                del self.index_by_subject_id[subject]

    def before_pickle(self):
        """Closes LMDB environment and transaction before pickling."""
        if self.lmdbtxn:
            self.lmdbtxn.abort()
            self.lmdbtxn = None
        if self.lmdbenv:
            self.lmdbenv.close()
            self.lmdbenv = None

    def __len__(self):
        """Returns the total number of samples across all subjects available in the dataset."""
        return sum(len(keys) for keys in self.index_by_subject_id.values())

    def __getitem__(self, lmdb_key: str):
        """
        Retrieves a sample from LMDB using its direct LMDB key string.

        Args:
            lmdb_key (str): The specific key string (e.g., "subject_id_window_idx") for the sample.

        Returns:
            dict: A dictionary containing loaded signals, annotations, and validity flags.
        """
        sample = dict()

        # Load raw ABP (always saved) and validity flags (always saved)
        # These are used for analysis/plotting purposes, even if processed signals are None
        sample['abp_raw'] = np.frombuffer(self.lmdbtxn.get(f"{lmdb_key}-abp_raw".encode()), dtype="float32")
        sample['ppg_valid'] = bool(np.frombuffer(self.lmdbtxn.get(f"{lmdb_key}-ppg_valid".encode()), dtype="int8"))
        sample['ecg_valid'] = bool(np.frombuffer(self.lmdbtxn.get(f"{lmdb_key}-ecg_valid".encode()), dtype="int8"))
        sample['abp_valid'] = bool(np.frombuffer(self.lmdbtxn.get(f"{lmdb_key}-abp_valid".encode()), dtype="int8"))

        # Load processed input signals (might be None if invalid during preprocessing, or if not configured)
        # We check for existence of the key in LMDB before trying to load
        processed_ppg = None
        ppg_bytes = self.lmdbtxn.get(f"{lmdb_key}-ppg".encode())
        if ppg_bytes: processed_ppg = np.frombuffer(ppg_bytes, dtype="float32")

        processed_ecg = None
        ecg_bytes = self.lmdbtxn.get(f"{lmdb_key}-ecg".encode())
        if ecg_bytes: processed_ecg = np.frombuffer(ecg_bytes, dtype="float32")
        
        processed_resp = None
        resp_bytes = self.lmdbtxn.get(f"{lmdb_key}-resp".encode())
        if resp_bytes: processed_resp = np.frombuffer(resp_bytes, dtype="float32")

        processed_vpg = None
        vpg_bytes = self.lmdbtxn.get(f"{lmdb_key}-vpg".encode())
        if vpg_bytes: processed_vpg = np.frombuffer(vpg_bytes, dtype="float32")

        processed_apg = None
        apg_bytes = self.lmdbtxn.get(f"{lmdb_key}-apg".encode())
        if apg_bytes: processed_apg = np.frombuffer(apg_bytes, dtype="float32")

        processed_imfs = None
        imfs_bytes = self.lmdbtxn.get(f"{lmdb_key}-imfs".encode())
        if imfs_bytes: processed_imfs = np.frombuffer(imfs_bytes, dtype="float32").reshape((4, self.sample_length_in_samples)).T

        processed_ppg_freqs = None
        ppg_freqs_bytes = self.lmdbtxn.get(f"{lmdb_key}-ppg_freqs".encode())
        if ppg_freqs_bytes: processed_ppg_freqs = np.frombuffer(ppg_freqs_bytes, dtype="float32").reshape((16, self.sample_length_in_samples)).T

        # Assemble `sig` based on configuration and available processed signals
        signals_list = []
        if self.ppg_emd and processed_imfs is not None:
            signals_list = [np.expand_dims(processed_imfs[:, i], axis=-1) for i in range(processed_imfs.shape[1])]
        elif self.ppg_freqs and processed_ppg_freqs is not None:
            signals_list = [np.expand_dims(processed_ppg_freqs[:, i], axis=-1) for i in range(processed_ppg_freqs.shape[1])]
        elif self.ppg_derivatives and processed_ppg is not None and processed_vpg is not None and processed_apg is not None:
            signals_list = [np.expand_dims(processed_ppg, axis=-1), np.expand_dims(processed_vpg, axis=-1), np.expand_dims(processed_apg, axis=-1)]
        elif self.ecg and processed_ppg is not None and processed_ecg is not None:
            signals_list.append(np.expand_dims(processed_ppg, axis=-1))
            signals_list.append(np.expand_dims(processed_ecg, axis=-1))
            if self.resp and processed_resp is not None:
                signals_list.append(np.expand_dims(processed_resp, axis=-1))
        elif processed_ppg is not None: # Default to PPG only if no other specific config matches and PPG is available
            signals_list.append(np.expand_dims(processed_ppg, axis=-1))

        if signals_list:
            sample['sig'] = np.concatenate(signals_list, axis=-1)
        else:
            # If no valid input signals were found for the chosen configuration or it's a completely bad window
            sample['sig'] = np.zeros((self.sample_length_in_samples, 1), dtype=np.float32) # Placeholder zero array

        # Load annotation: processed ABP or SBP/DBP
        abp_processed_bytes = self.lmdbtxn.get(f"{lmdb_key}-abp".encode())
        if self.sig2sig:
            # If processed ABP exists, use it; otherwise, use NaN-filled array for placeholder
            sample['abp_processed'] = np.squeeze(np.frombuffer(abp_processed_bytes, dtype="float32")) if abp_processed_bytes else np.full_like(sample['abp_raw'], np.nan)
            sample['sbp'] = np.nan # Not applicable for sig2sig output
            sample['dbp'] = np.nan # Not applicable for sig2sig output
        else:
            sample['sbp'] = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{lmdb_key}-sbp".encode()), dtype="float32"))
            sample['dbp'] = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{lmdb_key}-dbp".encode()), dtype="float32"))
            sample['abp_processed'] = np.full_like(sample['abp_raw'], np.nan) # Not applicable for BP estimation output


        # Store the LMDB key in the returned sample dictionary for easy access
        sample['lmdb_key'] = lmdb_key

        # Ensure all numpy arrays are writeable
        for k in sample:
            if isinstance(sample[k], np.ndarray):
                sample[k] = np.require(sample[k], requirements=['O', 'W'])
                sample[k].setflags(write=1)

        # Cast to torch tensor for primary output signals and annotations
        sample['sig_tensor'] = torch.tensor(sample['sig'], dtype=torch.float32)
        if self.sig2sig:
            sample['annotation_tensor'] = torch.tensor(sample['abp_processed'], dtype=torch.float32)
        else:
            sample['annotation_tensor'] = [
                torch.tensor(sample['sbp'], dtype=torch.float32).unsqueeze(-1),
                torch.tensor(sample['dbp'], dtype=torch.float32).unsqueeze(-1)
            ]

        return sample


class ContinualLearningDataset: # Note: This class does NOT inherit from Dataset. It's a manager.
    def __init__(self, online_physio_dataset_instance: OnlinePhysioDataset):
        """
        Initializes the ContinualLearningDataset for subject-specific online learning.

        Args:
            online_physio_dataset_instance (OnlinePhysioDataset): An *already initialized*
                                                            instance of OnlinePhysioDataset.
        """
        # We hold a reference to the already-opened OnlinePhysioDataset
        # This instance already has the LMDB environment and transaction open.
        self.base_dataset = online_physio_dataset_instance 

        self.active_subject_id = None
        self.active_subject_keys = []  # List of LMDB key strings for the current active subject, in temporal order
        self.current_sample_idx = 0    # Current index within active_subject_keys
        self.current_epoch_samples = [] # Cache of samples for the current epoch/pass over a subject for analysis/plotting

        print(f"{self.__class__.__name__} initialized for continual learning.")

    def set_active_subject(self, subject_id: int):
        """
        Sets the active subject for online learning and loads their sample keys.
        Resets the internal sample pointer.

        Args:
            subject_id (int): The ID of the subject to set as active.
        """
        # Access subject_list and index_by_subject_id from the base_dataset instance
        if subject_id not in self.base_dataset.subject_list:
            raise ValueError(f"Subject {subject_id} not found in the dataset.")

        self.active_subject_id = subject_id
        
        # Retrieve all LMDB keys for this subject and sort them to ensure temporal order.
        unsorted_keys = self.base_dataset.index_by_subject_id[subject_id]
        self.active_subject_keys = sorted(
            unsorted_keys,
            key=lambda k: int(k.split('_')[-1])
        )
        self.current_sample_idx = 0
        self.current_epoch_samples = [] # Clear cache for new subject or new pass

        print(f"Active subject set to {subject_id}. Total samples: {len(self.active_subject_keys)}")

    def get_next_sample(self):
        """
        Retrieves the next sample for the active subject in chronological order.

        Returns:
            dict or None: A dictionary containing the loaded sample data (signals, annotations, flags, etc.),
                          or None if all samples for the active subject have been processed.
        """
        if self.active_subject_id is None:
            raise RuntimeError("No active subject set. Call set_active_subject() first.")

        if self.current_sample_idx >= len(self.active_subject_keys):
            print(f"All samples for Subject S{self.active_subject_id} processed for this pass.")
            return None # Indicate end of subject's data for current pass

        lmdb_key = self.active_subject_keys[self.current_sample_idx]
        
        # Use the base_dataset's __getitem__ to load the full sample dictionary
        sample_data = self.base_dataset.__getitem__(lmdb_key)

        self.current_epoch_samples.append(sample_data) # Cache for later analysis/plotting
        self.current_sample_idx += 1
        
        return sample_data

    def get_num_remaining_samples(self):
        """Returns the number of samples remaining for the active subject in the current pass."""
        if self.active_subject_id is None:
            return 0
        return len(self.active_subject_keys) - self.current_sample_idx

    def analyze_and_plot_active_subject_sequences(self, savepath: str):
        """
        Analyzes and plots sequence validity for the currently active subject
        based on the samples collected in the current epoch/pass (`self.current_epoch_samples`).
        """
        if self.active_subject_id is None:
            print("No active subject set for analysis.")
            return
        if not self.current_epoch_samples:
            print(f"No samples collected for Subject S{self.active_subject_id} in the current pass for analysis.")
            return

        print(f"\nAnalyzing sequences for Subject S{self.active_subject_id} from cached samples...")

        subject_windows_for_analysis = self.current_epoch_samples

        supervised_sequence_lengths = []
        unlabeled_sequence_lengths = []
        ppg_ecg_any_abp_sequence_lengths = []

        current_supervised_sequence_length = 0
        current_unlabeled_sequence_length = 0
        current_ppg_ecg_any_abp_sequence_length = 0

        min_supervised_len_info = {'len': float('inf'), 'start_key': None, 'end_key': None}
        max_supervised_len_info = {'len': 0, 'start_key': None, 'end_key': None}
        min_unlabeled_len_info = {'len': float('inf'), 'start_key': None, 'end_key': None}
        max_unlabeled_len_info = {'len': 0, 'start_key': None, 'end_key': None}
        min_any_abp_len_info = {'len': float('inf'), 'start_key': None, 'end_key': None}
        max_any_abp_len_info = {'len': 0, 'start_key': None, 'end_key': None}

        current_supervised_start_key = None
        current_unlabeled_start_key = None
        current_any_abp_start_key = None

        for i, window_data in enumerate(subject_windows_for_analysis):
            ppg_valid = window_data['ppg_valid']
            ecg_valid = window_data['ecg_valid']
            abp_valid = window_data['abp_valid']
            current_key = window_data['lmdb_key'] # Get the LMDB key from the stored sample data itself

            # Supervised sequences
            if ppg_valid and ecg_valid and abp_valid:
                if current_supervised_sequence_length == 0:
                    current_supervised_start_key = current_key
                current_supervised_sequence_length += 1
                
                if current_unlabeled_sequence_length > 0:
                    unlabeled_sequence_lengths.append(current_unlabeled_sequence_length)
                    if current_unlabeled_sequence_length < min_unlabeled_len_info['len']:
                        min_unlabeled_len_info['len'] = current_unlabeled_sequence_length
                        min_unlabeled_len_info['start_key'] = current_unlabeled_start_key
                        min_unlabeled_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    if current_unlabeled_sequence_length > max_unlabeled_len_info['len']:
                        max_unlabeled_len_info['len'] = current_unlabeled_sequence_length
                        max_unlabeled_len_info['start_key'] = current_unlabeled_start_key
                        max_unlabeled_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    current_unlabeled_sequence_length = 0
            else:
                if current_supervised_sequence_length > 0:
                    supervised_sequence_lengths.append(current_supervised_sequence_length)
                    if current_supervised_sequence_length < min_supervised_len_info['len']:
                        min_supervised_len_info['len'] = current_supervised_sequence_length
                        min_supervised_len_info['start_key'] = current_supervised_start_key
                        min_supervised_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    if current_supervised_sequence_length > max_supervised_len_info['len']:
                        max_supervised_len_info['len'] = current_supervised_sequence_length
                        max_supervised_len_info['start_key'] = current_supervised_start_key
                        max_supervised_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    current_supervised_sequence_length = 0
            
            # Unlabeled (pseudo-labeling candidate) sequences
            if ppg_valid and ecg_valid and not abp_valid:
                if current_unlabeled_sequence_length == 0:
                    current_unlabeled_start_key = current_key
                current_unlabeled_sequence_length += 1
            else:
                if current_unlabeled_sequence_length > 0:
                    unlabeled_sequence_lengths.append(current_unlabeled_sequence_length)
                    if current_unlabeled_sequence_length < min_unlabeled_len_info['len']:
                        min_unlabeled_len_info['len'] = current_unlabeled_sequence_length
                        min_unlabeled_len_info['start_key'] = current_unlabeled_start_key
                        min_unlabeled_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    if current_unlabeled_sequence_length > max_unlabeled_len_info['len']:
                        max_unlabeled_len_info['len'] = current_unlabeled_sequence_length
                        max_unlabeled_len_info['start_key'] = current_unlabeled_start_key
                        max_unlabeled_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    current_unlabeled_sequence_length = 0

            # PPG and ECG valid (any ABP) sequences
            if ppg_valid and ecg_valid:
                if current_ppg_ecg_any_abp_sequence_length == 0:
                    current_any_abp_start_key = current_key
                current_ppg_ecg_any_abp_sequence_length += 1
            else:
                if current_ppg_ecg_any_abp_sequence_length > 0:
                    ppg_ecg_any_abp_sequence_lengths.append(current_ppg_ecg_any_abp_sequence_length)
                    if current_ppg_ecg_any_abp_sequence_length < min_any_abp_len_info['len']:
                        min_any_abp_len_info['len'] = current_ppg_ecg_any_abp_sequence_length
                        min_any_abp_len_info['start_key'] = current_any_abp_start_key
                        min_any_abp_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    if current_ppg_ecg_any_abp_sequence_length > max_any_abp_len_info['len']:
                        max_any_abp_len_info['len'] = current_ppg_ecg_any_abp_sequence_length
                        max_any_abp_len_info['start_key'] = current_any_abp_start_key
                        max_any_abp_len_info['end_key'] = subject_windows_for_analysis[i-1]['lmdb_key']
                    current_ppg_ecg_any_abp_sequence_length = 0

        # Handle remaining sequences at the end of the subject data
        if current_supervised_sequence_length > 0:
            supervised_sequence_lengths.append(current_supervised_sequence_length)
            if current_supervised_sequence_length < min_supervised_len_info['len']:
                min_supervised_len_info['len'] = current_supervised_sequence_length
                min_supervised_len_info['start_key'] = current_supervised_start_key
                min_supervised_len_info['end_key'] = subject_windows_for_analysis[-1]['lmdb_key']
            if current_supervised_sequence_length > max_supervised_len_info['len']:
                max_supervised_len_info['len'] = current_supervised_sequence_length
                max_supervised_len_info['start_key'] = current_supervised_start_key
                max_supervised_len_info['end_key'] = subject_windows_for_analysis[-1]['lmdb_key']

        if current_unlabeled_sequence_length > 0:
            unlabeled_sequence_lengths.append(current_unlabeled_sequence_length)
            if current_unlabeled_sequence_length < min_unlabeled_len_info['len']:
                min_unlabeled_len_info['len'] = current_unlabeled_sequence_length
                min_unlabeled_len_info['start_key'] = current_unlabeled_start_key
                min_unlabeled_len_info['end_key'] = subject_windows_for_analysis[-1]['lmdb_key']
            if current_unlabeled_sequence_length > max_unlabeled_len_info['len']:
                max_unlabeled_len_info['len'] = current_unlabeled_sequence_length
                max_unlabeled_len_info['start_key'] = current_unlabeled_start_key
                max_unlabeled_len_info['end_key'] = subject_windows_for_analysis[-1]['lmdb_key']

        if current_ppg_ecg_any_abp_sequence_length > 0:
            ppg_ecg_any_abp_sequence_lengths.append(current_ppg_ecg_any_abp_sequence_length)
            if current_ppg_ecg_any_abp_sequence_length < min_any_abp_len_info['len']:
                min_any_abp_len_info['len'] = current_ppg_ecg_any_abp_sequence_length
                min_any_abp_len_info['start_key'] = current_any_abp_start_key
                min_any_abp_len_info['end_key'] = subject_windows_for_analysis[-1]['lmdb_key']
            if current_ppg_ecg_any_abp_sequence_length > max_any_abp_len_info['len']:
                max_any_abp_len_info['len'] = current_ppg_ecg_any_abp_sequence_length
                max_any_abp_len_info['start_key'] = current_any_abp_start_key
                max_any_abp_len_info['end_key'] = subject_windows_for_analysis[-1]['lmdb_key']

        print(f"\n--- Subject {self.active_subject_id} Sequence Analysis ---")
        if supervised_sequence_lengths:
            print(f"Supervised Sequences (PPG, ECG, ABP Valid):")
            print(f"  Min Length: {min_supervised_len_info['len']} windows ({min_supervised_len_info['len'] * self.base_dataset.input_seq_len_s} s), Start Key: {min_supervised_len_info['start_key']}, End Key: {min_supervised_len_info['end_key']}")
            print(f"  Max Length: {max_supervised_len_info['len']} windows ({max_supervised_len_info['len'] * self.base_dataset.input_seq_len_s} s), Start Key: {max_supervised_len_info['start_key']}, End Key: {max_supervised_len_info['end_key']}")
            print(f"  Mean Length: {np.mean(supervised_sequence_lengths):.2f} windows")
        else:
            print("No supervised sequences found for this subject.")

        if unlabeled_sequence_lengths:
            print(f"Unlabeled Sequences (PPG, ECG Valid, ABP Invalid):")
            print(f"  Min Length: {min_unlabeled_len_info['len']} windows ({min_unlabeled_len_info['len'] * self.base_dataset.input_seq_len_s} s), Start Key: {min_unlabeled_len_info['start_key']}, End Key: {min_unlabeled_len_info['end_key']}")
            print(f"  Max Length: {max_unlabeled_len_info['len']} windows ({max_unlabeled_len_info['len'] * self.base_dataset.input_seq_len_s} s), Start Key: {max_unlabeled_len_info['start_key']}, End Key: {max_unlabeled_len_info['end_key']}")
            print(f"  Mean Length: {np.mean(unlabeled_sequence_lengths):.2f} windows")
        else:
            print("No unlabeled (pseudo-labeling candidate) sequences found for this subject.")

        if ppg_ecg_any_abp_sequence_lengths:
            print(f"PPG and ECG Valid (regardless of ABP) Sequences:")
            print(f"  Min Length: {min_any_abp_len_info['len']} windows ({min_any_abp_len_info['len'] * self.base_dataset.input_seq_len_s} s), Start Key: {min_any_abp_len_info['start_key']}, End Key: {min_any_abp_len_info['end_key']}")
            print(f"  Max Length: {max_any_abp_len_info['len']} windows ({max_any_abp_len_info['len'] * self.base_dataset.input_seq_len_s} s), Start Key: {max_any_abp_len_info['start_key']}, End Key: {max_any_abp_len_info['end_key']}")
            print(f"  Mean Length: {np.mean(ppg_ecg_any_abp_sequence_lengths):.2f} windows")
        else:
            print("No sequences found where PPG and ECG were valid for this subject.")

        print(f"\nPlotting validity for Subject S{self.active_subject_id} to {savepath}")
        plot_subject_validity_over_time(
            self.active_subject_id,
            self.current_epoch_samples, # Use the cached full sample data for plotting
            self.base_dataset.input_seq_len_s,
            self.base_dataset.fs,
            savepath
        )


def parseargs():
    parser = argparse.ArgumentParser(description="Dataset overview")

    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--save_path', default='./data_figs', type=str, help='where to save graphs from dataset analysis')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--pretraining_ratio', default=0.8, type=float, help='pretraining ratio of the whole dataset')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.1,0.2', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--personalization_sample_number', default=50, type=int, help='number of samples to take for personalization')
    parser.add_argument('--mix_pretraining_subject_samples', default='True', type=lambda x: bool(strtobool(x)), help='whether to mix pretraining subject samples among train/val/test or not')
    parser.add_argument('--plot', default='False', type=lambda x: bool(strtobool(x)), help='plot dataset overview or not (# subjects per pretraining/personalization steps, # samples in pretraining splits)')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg imfs or not')
    parser.add_argument('--ppg_freqs', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg freqs or not')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--plot_aug', default='False', type=lambda x: bool(strtobool(x)), help='plot signal augmentations or not')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of loader workers')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()  
    
    # RESP is loaded only if ECG is also loaded
    if args.resp and not args.ecg:
        raise ValueError('RESP can be loaded only along with ECG')
    
    # PPG derivatives/PPG EMD/PPG freqs are loaded only if ecg (and optionally resp) are not present
    if (args.ppg_derivatives or args.ppg_emd or args.ppg_freqs) and args.ecg: 
        raise ValueError('PPG derivatives/emd/scalogram can be loaded only without ECG (and optionally RESP)')  
    
    root_figs_folder = os.path.join(args.save_path, args.name) 
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)

    # Instantiate OnlinePhysioDataset (the base for continual learning)
    online_physio_dataset_base = OnlinePhysioDataset(
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        ppg_derivatives=args.ppg_derivatives,
        ppg_emd=args.ppg_emd,
        ppg_freqs=args.ppg_freqs,
        min_subject_sample_number=0 # Keep 0 for now as it doesn't affect data loading
    )

    # --- Continual Learning Setup ---
    continual_ds = ContinualLearningDataset(online_physio_dataset_base)

    # Example: Select a subject for online learning (e.g., the first subject in the list)
    if continual_ds.base_dataset.subject_list: # Access subject_list through the base_dataset instance
        online_subject_id = continual_ds.base_dataset.subject_list[1]
    else:
        print("No subjects available in the dataset for online learning.")
        sys.exit(1)

    continual_ds.set_active_subject(online_subject_id)
    
    # --- Simulate Online Training Loop for a Subject ---
    print(f"\nStarting simulated online learning for Subject S{online_subject_id}...")
    sample_count = 0
    while True:
        sample_data = continual_ds.get_next_sample() # Returns a dict
        if sample_data is None:
            break # No more samples for this subject

        signals = sample_data['sig_tensor'] # Use the tensor version
        annotation = sample_data['annotation_tensor'] # Use the tensor version
        abp_valid = sample_data['abp_valid']
        ppg_valid = sample_data['ppg_valid']
        ecg_valid = sample_data['ecg_valid']
        lmdb_key = sample_data['lmdb_key'] # The actual LMDB key for this sample

        # Here's where your online learning logic goes:
        # 1. Feed `signals` (torch.Tensor) to your pre-trained model.
        # 2. If `abp_valid` is True:
        #    This is a labeled sample. Use `signals` and `annotation` (torch.Tensor(s)) for supervised fine-tuning.
        # 3. If `abp_valid` is False:
        #    This is an unlabeled sample. Generate a pseudo-label for `signals` using your model.
        #    Use `signals` and the pseudo-label for semi-supervised training.
        # 4. Apply continual learning strategies (e.g., experience replay, regularization)
        #    based on the sample's type (labeled/unlabeled) and your chosen method.

        print(f"Processing sample {lmdb_key}: "
              f"Signals shape: {signals.shape}, "
              f"ABP Valid: {abp_valid}, "
              f"PPG Valid: {ppg_valid}, "
              f"ECG Valid: {ecg_valid}")
        
        sample_count += 1
        # Add a break for demonstration purposes to avoid infinite loop on very long recordings
        # if sample_count >= 10: # Process first 10 samples
        #    break

    print(f"Finished processing {sample_count} samples for Subject {online_subject_id}.")
    
    # --- Call analysis and plotting AFTER the loop ---
    # This ensures `current_epoch_samples` has all data for the subject
    print(f"\nPerforming analysis and plotting for Subject S{online_subject_id}...")
    continual_ds.analyze_and_plot_active_subject_sequences(root_figs_folder)

    # Close LMDB environments for all datasets
    online_physio_dataset_base.before_pickle()