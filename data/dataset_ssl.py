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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from preprocessing_utils.data_visualization import plot_signals, plot_pretraining_personalization_subjects_distribution, plot_subject_sample_distribution, plot_train_val_test_samples_distribution, calculate_dataset_mean_std
from preprocessing_utils.augmentations import RandomAugmentor, Identity, Jitter, TimeWarp, Scaling, MagnitudeWarp, Flip 
from preprocessing_utils.split import split_train_val_test


class PhysioDatasetSSL(Dataset):
    def __init__(self,
                 seed,
                 lmdb_folder,
                 pretraining_ratio=0.8,
                 pretraining_split_ratio=[0.7, 0.1, 0.2],
                 personalization_sample_number=50,
                 mix_pretraining_subject_samples=False,
                 fs=125,
                 input_seq_len_s=5,
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 ppg_derivatives=False,
                 ppg_emd=False,
                 ppg_freqs=False,
                 min_subject_sample_number=0,
                 plot=False,
                 savepath='./figs',
                 masking_ratio=0.15, 
                 augmentation_types=['jitter', 'scaling', 'magnitude_warp', 'flip'],
                 aug_prob=0.2 
                 ):
        super(PhysioDatasetSSL, self).__init__()

        # Generic arguments
        LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000 # 1T

        self.seed = seed
        self.dataset_folder = lmdb_folder
        self.min_subject_sample_number = min_subject_sample_number
        self.lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE, readonly=True, lock=False) # Open in read-only mode for safety
        self.lmdbtxn = self.lmdbenv.begin(buffers=True) # Use buffers=True for faster direct access to byte arrays

        # Subject/Sample lists/dicts
        self.subject_list:list = pickle.loads(self.lmdbtxn.get("subject_list".encode()))
        self.index_by_subject_id:dict = pickle.loads(self.lmdbtxn.get("index_by_subject_id".encode()))
        self.index_by_sample_id = pickle.loads(self.lmdbtxn.get("index_by_sample_id".encode())) # This should now be a list of tuples (subject_id, segment_idx)
        self.check_subjects_list(min_subject_sample_number=min_subject_sample_number)

        # Which input data to load (PPG + VPG + APG, PPG + ECG, etc.), PPG is always loaded
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig # Predict annotation over the whole analysis window or not
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.ppg_freqs = ppg_freqs
        self.fs = fs
        self.input_seq_len_s = input_seq_len_s
        self.sample_len = self.fs * self.input_seq_len_s # Consistent way to get 625

        # Plot arguments
        self.plot = plot
        self.savepath = savepath

        # Dataset split
        self.total_subject_n = len(self.subject_list)
        # self.total_sample_n = len(self.index_by_sample_id) # This was for the old structure. We'll use len(self.index_by_sample_id) directly if it's the valid samples.
        self.pretraining_ratio = pretraining_ratio # Percentage of pretraining subjects of the whole dataset
        self.pretraining_split_ratio = pretraining_split_ratio # To divide pretraining from personalization, and then to divide the pretraining dataset
        self.mix_pretraining_subject_samples = mix_pretraining_subject_samples # Whether to split train/val/test during pretraining subjectwise or not
        self.personalization_sample_number = personalization_sample_number # To split subject data during the personalization stage

        # Split pretraining and personalization subjects
        # Ensure subject_list is a list of actual subject IDs, not (subject_id, segment_idx) tuples if that's what index_by_sample_id implies
        # Assuming self.subject_list contains unique subject IDs (e.g., 's001', 's002')
        pretraining_subjects, personalization_subjects = train_test_split(self.subject_list, test_size=(1.0-self.pretraining_ratio), random_state=seed)

        self.subjects_for_pretraining = pretraining_subjects
        self.subjects_for_personalization = personalization_subjects

        if self.plot:
            plot_pretraining_personalization_subjects_distribution(self.subjects_for_pretraining, self.subjects_for_personalization, title=f'Pretraining vs Personalization Subjects Split (Total {len(self.subject_list)} subjects)', savepath=os.path.join(savepath, 'pretraining_vs_personalization_subjects_distribution.jpg'))
            plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, 'pretraining_subject_sample_distribution.jpg'))
            plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_personalization, savepath=os.path.join(savepath, 'personalization_subject_sample_distribution.jpg'))

        print("{:s} initialized with following configuration:".format(self.__class__.__name__))
        pprint.pprint(
            {
                "Subjects":
                {
                    "Total": len(self.subject_list),
                    "Pretraining": len(self.subjects_for_pretraining),
                    "Personalization": len(self.subjects_for_personalization)
                },
                "Samples": len(self.index_by_sample_id), # Corrected to reflect actual number of storable samples
                "Pretraining Ratio": self.pretraining_ratio
            }
        )

        # Self-Supervised augmentations and pretext mask task
        # Initialize augmentors for SimCLR. Each channel augmented independently.
        # For simplicity, using same set of augmentations for both, but could be different
        self.masking_ratio = masking_ratio
        self.augmentor_ppg = self._create_augmentor(augmentation_types, aug_prob)
        self.augmentor_ecg = self._create_augmentor(augmentation_types, aug_prob) 
        

    def _create_augmentor(self, augmentation_types, aug_prob):
        r"""Helper to create RandomAugmentor based on types."""
        augmentations = []
        for aug_type in augmentation_types:
            if aug_type == 'identity':
                augmentations.append(Identity(prob=aug_prob))
            elif aug_type == 'jitter':
                augmentations.append(Jitter(prob=aug_prob))
            elif aug_type == 'scaling':
                augmentations.append(Scaling(prob=aug_prob))
            elif aug_type == 'magnitude_warp':
                augmentations.append(MagnitudeWarp(prob=aug_prob))
            elif aug_type == 'flip':
                augmentations.append(Flip(prob=aug_prob))
            
        return RandomAugmentor(augmentations)


    def check_subjects_list(self, min_subject_sample_number=0):
        # Considering preprocessing in the mimic_iii, when a subject has no valid samples,
        # its ID is in the self.index_by_subject_id but not in the self.index_by_sample_id as the for loop inside
        # with lmdbenv.begin(write=True) as txn: deos not make this check
        invalid_subjects = list()
        for subject in self.subject_list:
            if len(self.index_by_subject_id[subject]) <= min_subject_sample_number:
                invalid_subjects.append(subject)
        
        if len(invalid_subjects) > 0:
            print("Invalid subjects found in the dataset, removing them ...")
            for subject in invalid_subjects:
                self.subject_list.remove(subject)
                del self.index_by_subject_id[subject]
                
    
    def _get_signal_from_lmdb(self, sample_id, key_prefix):
        r"""Helper to get a signal from LMDB, handles missing keys."""
        key = f"{sample_id}-{key_prefix}".encode()
        value = self.lmdbtxn.get(key)
        if value is not None:
            # Assuming all signals are float32 and SIGNAL_LENGTH is consistent
            return np.frombuffer(value, dtype=np.float32)
        return None

    def _get_multi_channel_signal(self, sample_id, ppg_key_prefix, ecg_key_prefix):
        r"""Helper to get stacked PPG and ECG for a specific window type (prev/curr/next)."""
        ppg_signal = self._get_signal_from_lmdb(sample_id, ppg_key_prefix)
        ecg_signal = self._get_signal_from_lmdb(sample_id, ecg_key_prefix)

        if ppg_signal is not None and ecg_signal is not None:
            # Stack to [LENGTH, 2]
            return np.stack([ppg_signal, ecg_signal], axis=-1)
        return None # Return None if either is missing


    def _mask_signal(self, signal_tensor, masking_ratio, masking_strategy='adaptive_spans', min_span_length=None, max_span_length=None, num_spans=None):
        r"""
        Masks portions of physiological signals (ECG, PPG, etc.) using span-based masking
        optimized for masked modeling learning (MML).
        
        Parameters
        ------------
            signal_tensor (torch.Tensor): 
                Input signal of shape [LENGTH, MODALITIES].
            masking_ratio (float): 
                The ratio of the signal to mask (e.g., 0.15 = 15%).
            masking_strategy (str): 
                Strategy for span masking:
                - 'adaptive_spans': Variable span lengths based on signal characteristics
                - 'fixed_spans': Fixed number of spans with variable lengths
                - 'physiological': Spans mimicking real physiological artifacts
                - 'block': Large contiguous blocks (good for long sequences)
            min_span_length (int, optional): 
                Minimum span length. Defaults based on strategy.
            max_span_length (int, optional): 
                Maximum span length. Defaults based on strategy.
            num_spans (int, optional):
                Target number of spans (for 'fixed_spans' strategy).
        
        Returns
        ------------
            masked_signal (torch.Tensor): 
                The masked signal with spans set to zero.
            mask_indices (torch.Tensor): 
                Boolean tensor indicating masked positions [LENGTH].
        """
        length = signal_tensor.shape[0]
        num_mask_tokens = int(length * masking_ratio)
        
        if num_mask_tokens == 0:
            return signal_tensor.clone(), torch.zeros(length, dtype=torch.bool)
        
        # Set strategy-specific defaults
        if masking_strategy == 'adaptive_spans':
            if min_span_length is None:
                min_span_length = max(1, length // 200)  # ~0.5% of signal
            if max_span_length is None:
                max_span_length = max(min_span_length, length // 20)  # ~5% of signal
                
        elif masking_strategy == 'physiological':
            # Mimic real physiological artifacts
            if min_span_length is None:
                min_span_length = max(1, length // 100)  # Short artifacts
            if max_span_length is None:
                max_span_length = max(min_span_length, length // 10)  # Long artifacts
                
        elif masking_strategy == 'block':
            # Larger contiguous blocks
            if min_span_length is None:
                min_span_length = max(1, length // 50)
            if max_span_length is None:
                max_span_length = max(min_span_length, length // 5)
                
        elif masking_strategy == 'fixed_spans':
            if num_spans is None:
                num_spans = max(1, int(length * masking_ratio / 10))  # ~10 tokens per span avg
            avg_span_length = max(1, num_mask_tokens // num_spans)
            if min_span_length is None:
                min_span_length = max(1, avg_span_length // 2)
            if max_span_length is None:
                max_span_length = max(min_span_length, avg_span_length * 2)
        
        # Default fallback
        if min_span_length is None:
            min_span_length = 1
        if max_span_length is None:
            max_span_length = max(min_span_length, length // 20)
        
        return self._generate_span_mask(signal_tensor, num_mask_tokens, 
                                min_span_length, max_span_length, 
                                masking_strategy, num_spans)


    def _generate_span_mask(self, signal_tensor, num_mask_tokens, min_span_length, max_span_length, strategy, num_spans=None):
        """Generate span mask based on strategy."""
        length = signal_tensor.shape[0]
        mask_indices = torch.zeros(length, dtype=torch.bool)
        masked_tokens = 0
        
        if strategy == 'fixed_spans' and num_spans:
            return self._fixed_spans_masking(signal_tensor, num_mask_tokens, num_spans, min_span_length, max_span_length)
        
        # Adaptive span generation
        max_attempts = min(1000, length * 2)
        attempts = 0
        consecutive_failures = 0
        
        while masked_tokens < num_mask_tokens and attempts < max_attempts:
            attempts += 1
            
            # Determine span length based on strategy
            if strategy == 'physiological':
                span_length = self._physiological_span_length(min_span_length, max_span_length)
            elif strategy == 'block':
                # Favor larger spans for block strategy
                span_length = self._block_span_length(min_span_length, max_span_length, num_mask_tokens - masked_tokens)
            else:  # adaptive_spans
                span_length = self._adaptive_span_length(min_span_length, max_span_length, length, masked_tokens, num_mask_tokens)
            
            span_length = min(span_length, num_mask_tokens - masked_tokens)
            
            if span_length <= 0:
                break
            
            # Find valid start position
            max_start = length - span_length
            if max_start < 0:
                break
                
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + span_length
            
            # Check for overlap
            if not mask_indices[start_idx:end_idx].any():
                mask_indices[start_idx:end_idx] = True
                masked_tokens += span_length
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                
                # If many consecutive failures, try smaller spans or allow partial overlap
                if consecutive_failures > 20:
                    available_in_span = (~mask_indices[start_idx:end_idx]).sum().item()
                    if available_in_span > 0:
                        # Mask only the available positions in this span
                        mask_indices[start_idx:end_idx] = mask_indices[start_idx:end_idx] | ~mask_indices[start_idx:end_idx]
                        masked_tokens += available_in_span
                        consecutive_failures = 0
        
        # Fill remaining tokens if needed
        if masked_tokens < num_mask_tokens:
            remaining_positions = (~mask_indices).nonzero(as_tuple=True)[0]
            if len(remaining_positions) > 0:
                remaining_needed = min(num_mask_tokens - masked_tokens, len(remaining_positions))
                # Prefer positions that extend existing spans
                extended_positions = self._find_span_extensions(mask_indices, remaining_positions, remaining_needed)
                mask_indices[extended_positions] = True
        
        # Create masked signal
        masked_signal = signal_tensor.clone()
        masked_signal[mask_indices, :] = 0.0
        
        return masked_signal, mask_indices


    def _physiological_span_length(self, min_len, max_len):
        """Generate span lengths mimicking physiological artifacts."""
        # Exponential distribution favoring shorter artifacts with occasional long ones
        if random.random() < 0.7:  # 70% short artifacts
            return random.randint(min_len, min(max_len, min_len * 3))
        else:  # 30% longer artifacts
            return random.randint(min_len * 2, max_len)


    def _block_span_length(self, min_len, max_len, remaining_tokens):
        """Generate larger block spans."""
        # Favor larger spans for block masking
        target_len = random.randint(max(min_len, max_len // 2), max_len)
        return min(target_len, remaining_tokens)


    def _adaptive_span_length(self, min_len, max_len, total_length, masked_so_far, target_masked):
        """Adaptive span length based on progress."""
        progress = masked_so_far / target_masked if target_masked > 0 else 0
        
        if progress < 0.3:  # Early phase: varied lengths
            return random.randint(min_len, max_len)
        elif progress < 0.8:  # Middle phase: moderate lengths
            mid_len = (min_len + max_len) // 2
            return random.randint(min_len, mid_len)
        else:  # Late phase: smaller lengths to fine-tune
            return random.randint(min_len, min(max_len, min_len * 2))


    def _fixed_spans_masking(self, signal_tensor, num_mask_tokens, num_spans, min_len, max_len):
        """Generate exactly num_spans spans."""
        length = signal_tensor.shape[0]
        mask_indices = torch.zeros(length, dtype=torch.bool)
        
        # Calculate span lengths
        base_span_length = num_mask_tokens // num_spans
        extra_tokens = num_mask_tokens % num_spans
        
        spans_placed = 0
        max_attempts = num_spans * 10
        
        for attempt in range(max_attempts):
            if spans_placed >= num_spans:
                break
                
            # Determine span length for this span
            span_length = base_span_length
            if spans_placed < extra_tokens:
                span_length += 1
                
            span_length = max(min_len, min(max_len, span_length))
            
            # Find position
            max_start = length - span_length
            if max_start < 0:
                continue
                
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + span_length
            
            if not mask_indices[start_idx:end_idx].any():
                mask_indices[start_idx:end_idx] = True
                spans_placed += 1
        
        masked_signal = signal_tensor.clone()
        masked_signal[mask_indices, :] = 0.0
        
        return masked_signal, mask_indices


    def _find_span_extensions(self, mask_indices, remaining_positions, needed):
        """Find positions that extend existing spans (for more natural masking)."""
        if needed >= len(remaining_positions):
            return remaining_positions
        
        # Find positions adjacent to existing masks
        adjacent_positions = []
        for pos in remaining_positions:
            pos_int = pos.item()
            is_adjacent = False
            
            # Check if adjacent to existing mask
            if pos_int > 0 and mask_indices[pos_int - 1]:
                is_adjacent = True
            if pos_int < len(mask_indices) - 1 and mask_indices[pos_int + 1]:
                is_adjacent = True
                
            if is_adjacent:
                adjacent_positions.append(pos)
        
        # Prefer adjacent positions, then random
        if len(adjacent_positions) >= needed:
            return torch.tensor(random.sample(adjacent_positions, needed))
        else:
            selected = adjacent_positions.copy()
            remaining_pool = [p for p in remaining_positions if p not in adjacent_positions]
            additional_needed = needed - len(selected)
            if additional_needed > 0 and remaining_pool:
                additional = random.sample(remaining_pool, min(additional_needed, len(remaining_pool)))
                selected.extend(additional)
            return torch.tensor(selected)


    # Utility function for common use cases
    def mask_ecg_signal(self, signal_tensor, masking_ratio=0.15):
        """Convenience function for ECG signal masking."""
        # ECG-specific parameters (assuming ~250-500 Hz sampling)
        length = signal_tensor.shape[0]
        return self._mask_signal(None, 
                                 signal_tensor, 
                                 masking_ratio, 
                                 masking_strategy='physiological', 
                                 min_span_length=max(1, length // 250),  # ~4ms at 250Hz
                                 max_span_length=max(1, length // 25))   # ~40ms at 250Hz


    def mask_ppg_signal(self, signal_tensor, masking_ratio=0.15):
        """Convenience function for PPG signal masking."""
        # PPG-specific parameters (assuming ~100-125 Hz sampling)
        length = signal_tensor.shape[0]
        return self._mask_signal(None, 
                                 signal_tensor, 
                                 masking_ratio,
                                 masking_strategy='physiological',
                                 min_span_length=max(1, length // 125),  # ~8ms at 125Hz
                                 max_span_length=max(1, length // 12))   # ~80ms at 125Hz
    

    def get_pretraining_samplers(self):

        print("{:s} {:s}".format(self.__class__.__name__, sys._getframe().f_code.co_name))

        if self.mix_pretraining_subject_samples:

            print("Pretraining train/val/test sets samples are sampled from the same subject set")

            pretraining_train_sample_ids = []
            pretraining_val_sample_ids = []
            pretraining_test_sample_ids = []

            # Loop over subjects for pretraining
            for subject in self.subjects_for_pretraining:

                # Get all sample_ids (LMDB keys) for a subject
                subject_sample_ids = self.index_by_subject_id[subject]

                # First X% samples for training, next Y% for valid, and remaining for test
                # Ensure split_train_val_test works with actual sample_ids (integers)
                train_idx, valid_idx, test_idx = split_train_val_test(subject_sample_ids, split_ratio=self.pretraining_split_ratio, shuffle=True)

                # Add to the training/val/test list: note that each subject is contributing equally ot he split in this way
                pretraining_train_sample_ids.extend(train_idx)
                pretraining_val_sample_ids.extend(valid_idx)
                pretraining_test_sample_ids.extend(test_idx)

        else:

            print("Pretraining train/val/test sets samples are sampled from different subject sets")

            # Split pretraining subjects into train/val/test
            _, val_ratio, test_ratio = self.pretraining_split_ratio

            self.pretraining_train_subjects, pretraining_val_test_subjects = train_test_split(
                self.subjects_for_pretraining,
                test_size=(val_ratio + test_ratio),
                random_state=self.seed
                )

            self.pretraining_val_subjects, self.pretraining_test_subjects = train_test_split(
                pretraining_val_test_subjects,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                random_state=self.seed
                )

            print("Pretraining Set Split:")
            print(f"Number of pretraining training IDs: {len(self.pretraining_train_subjects)}")
            print(f"Number of pretraining validation IDs: {len(self.pretraining_val_subjects)}")
            print(f"Number of pretraining test IDs: {len(self.pretraining_test_subjects)}")

            pretraining_train_sample_ids = []
            for subject_id in self.pretraining_train_subjects:
                pretraining_train_sample_ids.extend(self.index_by_subject_id[subject_id])

            pretraining_val_sample_ids = []
            for subject_id in self.pretraining_val_subjects:
                pretraining_val_sample_ids.extend(self.index_by_subject_id[subject_id])

            pretraining_test_sample_ids = []
            for subject_id in self.pretraining_test_subjects:
                pretraining_test_sample_ids.extend(self.index_by_subject_id[subject_id])

        # Shuffle train partition (seed set in the fixseed function in utils)
        random.shuffle(pretraining_train_sample_ids)

        pprint.pprint(
            {
                "Pretraining Samples (# per split)":
                {
                    "Train": len(pretraining_train_sample_ids),
                    "Valid": len(pretraining_val_sample_ids),
                    "Test": len(pretraining_test_sample_ids)
                }
            }
        )

        # Plot
        if self.plot:
            plot_name = 'pretraining_dataset_overview_mixed.jpg' if self.mix_pretraining_subject_samples else 'pretraining_dataset_overview_non_mixed.jpg'
            plot_train_val_test_samples_distribution(
                pretraining_train_sample_ids,
                pretraining_val_sample_ids,
                pretraining_test_sample_ids,
                title=f'Pretraining Dataset Sample Distribution (From {len(self.subjects_for_pretraining)} subjects)',
                savepath=os.path.join(self.savepath, plot_name)
                )

        return (SubsetRandomSampler(pretraining_train_sample_ids), SubsetRandomSampler(pretraining_val_sample_ids), SubsetRandomSampler(pretraining_test_sample_ids))

    def before_pickle(self):
        self.lmdbenv = None
        self.lmdbtxn = None

    def __len__(self):
        # Return the total number of valid samples that can be loaded
        return len(self.index_by_sample_id)

    def __getitem__(self, index_in_lmdb):
        # `index_in_lmdb` here refers to the actual `sample_id` (integer) stored in LMDB,
        # The sampler will provide these actual LMDB sample_ids.

        # --- Load Current Window Signals (PPG and ECG are always needed for self-supervised) ---
        current_ppg = self._get_signal_from_lmdb(index_in_lmdb, "curr_ppg")
        current_ecg = self._get_signal_from_lmdb(index_in_lmdb, "curr_ecg")

        if current_ppg is None or current_ecg is None:
            # This should ideally not happen 
            raise ValueError(f"Missing current PPG or ECG for sample_id {index_in_lmdb}. Data corruption or preprocessing error.")
        
        current_ppg_ecg_stacked = np.stack([current_ppg, current_ecg], axis=-1) # Shape [LENGTH, 2]
        
        # --- Self-supervised Mode Logic ---
       
        # 1. Load Triplet for cross-windows reconstruction
        # prev_ppg_ecg_pair and next_ppg_ecg_pair are stored as (ppg_array, ecg_array) or (None, None)
        # They were saved in the LMDB directly using `f"{sample_id}-prev_ppg"` etc.
        
        prev_ppg_ecg = self._get_multi_channel_signal(index_in_lmdb, "prev_ppg", "prev_ecg")
        next_ppg_ecg = self._get_multi_channel_signal(index_in_lmdb, "next_ppg", "next_ecg")
        
        # 2. Masked Signal for MSR
        # Apply masking to the current_ppg_ecg_stacked
        current_ppg_ecg_tensor = torch.from_numpy(current_ppg_ecg_stacked)
        masked_current_ppg_ecg, mask_indices = self._mask_signal(current_ppg_ecg_tensor, self.masking_ratio, masking_strategy='fixed_spans', num_spans=5)

        # 3. Augmented Signals for SimCLR
        # Apply two different augmentations to the current_ppg_ecg_stacked
        aug1_ppg = self.augmentor_ppg(torch.from_numpy(current_ppg_ecg_stacked[:, 0])) # PPG channel
        aug1_ecg = self.augmentor_ecg(torch.from_numpy(current_ppg_ecg_stacked[:, 1])) # ECG channel
        aug1_ppg_ecg = torch.stack([aug1_ppg, aug1_ecg], dim=-1).squeeze()

        aug2_ppg = self.augmentor_ppg(torch.from_numpy(current_ppg_ecg_stacked[:, 0]))
        aug2_ecg = self.augmentor_ecg(torch.from_numpy(current_ppg_ecg_stacked[:, 1]))
        aug2_ppg_ecg = torch.stack([aug2_ppg, aug2_ecg], dim=-1).squeeze()

        # Return a dictionary suitable for self-supervised training
        return {
            'current_signal': current_ppg_ecg_tensor,          # For Jigsaw, base for MSR/SimCLR
            'masked_signal': masked_current_ppg_ecg,           # For MSR input
            'mask_indices': mask_indices,                      # For MSR loss (to compute loss only on masked parts)
            'aug1_signal': aug1_ppg_ecg,                       # For SimCLR view 1
            'aug2_signal': aug2_ppg_ecg,                       # For SimCLR view 2
            'prev_signal': torch.from_numpy(prev_ppg_ecg) if prev_ppg_ecg is not None else None, # For CWG (input or context)
            'next_signal': torch.from_numpy(next_ppg_ecg) if next_ppg_ecg is not None else None, # For CWG (target)
            'sample_id': index_in_lmdb, # Useful for debugging or associating with original data
        }

        
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
    parser.add_argument('--loader_worker', default=4, type=int, help='number of loader workers')

    # New arguments for self-supervised learning
    parser.add_argument('--masking_ratio', default=0.15, type=float, help='Ratio of signal length to mask for MSR task')
    parser.add_argument('--augmentation_types', default='jitter,scaling,magnitude_warp,flip', type=str, help='Comma-separated list of augmentation types for SimCLR')
    parser.add_argument('--aug_prob', default=0.2, type=float, help='Probability for each individual augmentation in RandomAugmentor')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    global args
    args = parseargs()

    assert args.ecg, "ECG data must be enabled for this script to run."
    
    # RESP is loaded only if ECG is also loaded
    if args.resp and not args.ecg:
        raise ValueError('RESP can be loaded only along with ECG')

    # PPG derivatives/PPG EMD/PPG freqs are loaded only if ecg (and optionally resp) are not present
    if (args.ppg_derivatives or args.ppg_emd or args.ppg_freqs) and args.ecg:
        raise ValueError('PPG derivatives/emd/scalogram can be loaded only without ECG (and optionally RESP)')

    root_figs_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)

    # Initialize dataset with self_supervised_mode argument
    dataset = PhysioDatasetSSL(
        seed=args.seed,
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        pretraining_ratio=args.pretraining_ratio,
        pretraining_split_ratio=list(map(float, args.pretraining_tr_val_tt_split_ratio.split(','))),
        personalization_sample_number=args.personalization_sample_number,
        mix_pretraining_subject_samples=args.mix_pretraining_subject_samples,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        ppg_derivatives=args.ppg_derivatives,
        ppg_emd=args.ppg_emd,
        ppg_freqs=args.ppg_freqs,
        plot=args.plot,
        savepath=root_figs_folder,
        masking_ratio=args.masking_ratio,
        augmentation_types=args.augmentation_types.split(','),
        aug_prob=args.aug_prob
    )

    # Pretraining statistics
    (train_sampler, val_sampler, test_sampler) = dataset.get_pretraining_samplers()

    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    valid_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)

    # --- Test Data Loading (Modified for self-supervised output) ---
    input_batch = next(iter(train_dataloader))
    
    print("\n--- Self-Supervised Mode Sample Output Structure ---")
    # Print keys and shapes for verification
    for key, value in input_batch.items():
        if value is None:
            print(f"Key: {key}, Value: None")
        elif isinstance(value, torch.Tensor):
            print(f"Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"Key: {key}, Type: List of Tensors/None")
            for i, item in enumerate(value):
                if item is None:
                    print(f"  Item {i}: None")
                elif isinstance(item, torch.Tensor):
                    print(f"  Item {i}: Shape: {item.shape}, Dtype: {item.dtype}")
        else:
            print(f"Key: {key}, Type: {type(value)}")

    # Example plotting for self-supervised mode
    idx = np.random.randint(0, input_batch['current_signal'].shape[0])
    print(f"\nPlotting sample {idx} from batch for self-supervised mode...")

    # Original signal (for reference)
    plot_signals(
        input_batch['current_signal'][idx].numpy().T,
        fs=args.fs,
        labels=['PPG', 'ECG'],
        title=f'Original Signal (Sample {idx})',
        savepath=root_figs_folder,
        ylabels=['a.u.', 'mV']
    )

    # Masked signal () the list is required by plot_signals
    masked_sig_to_plot = input_batch['masked_signal'][idx].numpy().T
    plot_signals(
        masked_sig_to_plot,
        fs=args.fs,
        labels=['PPG (Masked)', 'ECG (Masked)'],
        title=f'Masked Signal (Sample {idx})',
        savepath=root_figs_folder,
        ylabels=['a.u.', 'mV']
    )

    # Augmented signal 1
    aug1_sig_to_plot = input_batch['aug1_signal'][idx].numpy().T
    plot_signals(
        aug1_sig_to_plot,
        fs=args.fs,
        labels=['PPG (Aug 1)', 'ECG (Aug 1)'],
        title=f'Augmented Signal 1 (Sample {idx})',
        savepath=root_figs_folder,
        ylabels=['a.u.', 'mV']
    )

    # Augmented signal 2
    aug2_sig_to_plot = input_batch['aug2_signal'][idx].numpy().T
    plot_signals(
        aug2_sig_to_plot,
        fs=args.fs,
        labels=['PPG (Aug 2)', 'ECG (Aug 2)'],
        title=f'Augmented Signal 2 (Sample {idx})',
        savepath=root_figs_folder,
        ylabels=['a.u.', 'mV']
    )
    
    # You might want to plot prev/next if they are not None as well
    plot_signals(
        input_batch['next_signal'][idx].numpy().T,
        fs=args.fs,
        labels=['PPG (Next)', 'ECG (Next)'],
        title=f'Next Window Signal (Sample {idx})',
        savepath=root_figs_folder,
        ylabels=['a.u.', 'mV']
    )
    plot_signals(
        input_batch['prev_signal'][idx].numpy().T,
        fs=args.fs,
        labels=['PPG (Prev)', 'ECG (Prev)'],
        title=f'Previous Window Signal (Sample {idx})',
        savepath=root_figs_folder,
        ylabels=['a.u.', 'mV']
    )

    