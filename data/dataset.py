import os 
import argparse
import pickle
import numpy as np
from collections import defaultdict, Counter
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from preprocessing_utils.data_visualization import (
    plot_signals, plot_subject_sample_distribution, plot_age_gender_distribution,
    plot_train_val_test_samples_distribution, calculate_dataloaders_mean_std
)


class PhysioDataset(Dataset):
    def __init__(self,
                 seed, 
                 lmdb_folder, 
                 pretraining_split_ratio=[0.7, 0.1, 0.2], 
                 meta_split_ratio=0.2,
                 fs=125, 
                 input_seq_len_s=5, 
                 ecg=False, 
                 min_subject_sample_number=0, 
                 drift_aware=False,
                 plot=False, 
                 savepath='./figs'):
        super(PhysioDataset, self).__init__()

        # Generic arguments
        LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000 # 1T

        self.seed = seed
        self.dataset_folder = lmdb_folder
        self.min_subject_sample_number = min_subject_sample_number
        self.lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE)
        self.lmdbtxn = self.lmdbenv.begin()

        # Subject/Sample lists/dicts
        self.subjects_for_pretraining:list = pickle.loads(self.lmdbtxn.get("subject_list".encode()))
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
        self.drift_aware = drift_aware
        self.total_subject_n = len(self.subjects_for_pretraining)
        self.pretraining_split_ratio = pretraining_split_ratio # To divide pretraining from personalization, and then to divide the pretraining dataset
        self.meta_split_ratio = meta_split_ratio

        if self.plot:
            if self.min_subject_sample_number > 0:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, f'pretraining_subject_sample_distribution_min_sample_{self.min_subject_sample_number}.jpg'))
            else:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, 'pretraining_subject_sample_distribution.jpg'))
        
        print(f"[PhysioDataset] Initialized with following configuration:")
        print(f"\t-Total Subjects: {len(self.subjects_for_pretraining)}")
        print(f"\t-Total Samples: {len(self.index_by_sample_id)}")
        
                    
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
        for subject in self.subjects_for_pretraining:
            if len(self.index_by_subject_id[subject]) <= min_subject_sample_number:
                invalid_subjects.append(subject)
        
        if len(invalid_subjects) > 0:
            print("[PhysioDataset] Invalid subjects found in the dataset, removing them ...")
            for subject in invalid_subjects:
                self.subjects_for_pretraining.remove(subject)
                del self.index_by_subject_id[subject]

        # Shorten each subject list to min_subject_sample_number
        if min_subject_sample_number > 0:
            for subject in self.subjects_for_pretraining:
                if len(self.index_by_subject_id[subject]) > min_subject_sample_number:
                    self.index_by_subject_id[subject] = self.index_by_subject_id[subject][:min_subject_sample_number]

    def get_pretraining_samplers(self):
        r"""
        Partitions the pre-training dataset into specialized cohorts for Supervised Learning 
        and Meta-Learning, ensuring strict subject-level isolation.

        The partitioning follows a hierarchical strategy:
        1.  **Macro Split**: Divides all pre-training subjects into Train, Validation, 
            and Test sets based on `pretraining_split_ratio`.
        2.  **Drift Categorization**: Analyzes the Training set to classify subjects into 
            'low', 'medium', or 'high' drift regimes using quartile thresholds ($Q25$ and $Q75$).
        3.  **Specialized Train Split**: Further divides the Training cohort into:
            - **Supervised Pre-train**: For standard representation learning.
            - **Meta-Learning**: For training the model's ability to adapt quickly (MAML).
        4.  **Drift-Awareness**: If `drift_aware` is enabled, the split ensures a 
            balanced representation of different physiological drift regimes in both 
            training pools.

        Returns
        ------------
        supervised_sampler (SubsetRandomSampler):
            Indices for standard supervised pre-training.
        meta_sampler (SubsetRandomSampler):
            Indices for meta-learning tasks.
        val_sampler (SubsetRandomSampler):
            Indices for hyperparameter validation.
        test_sampler (SubsetRandomSampler):
            Indices for final pre-training performance assessment.
        """
            
        print("[PhysioDataset] Pretraining train/val/test sets samples are sampled from different subject sets")
        
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
        
        self.supervised_pretrain_subjects, self.meta_learning_subjects = train_test_split(
            self.pretraining_train_subjects,
            test_size=self.meta_split_ratio,
            random_state=self.seed
        )
        
        print("[PhysioDataset] Pretraining subjects per split")
        print(f"\t-# of train subjects: {len(self.pretraining_train_subjects)}")
        print(f"\t\t-# supervised pretrain subjects: {len(self.supervised_pretrain_subjects)}")
        print(f"\t\t-# meta-learning subjects: {len(self.meta_learning_subjects)}")
        print(f"\t-# of val subjects: {len(self.pretraining_val_subjects)}")
        print(f"\t-# of test subjects: {len(self.pretraining_test_subjects)}")

        supervised_pretraining_train_sample_ids = []
        for subject_id in self.supervised_pretrain_subjects:
            supervised_pretraining_train_sample_ids.extend(self.index_by_subject_id[subject_id])

        meta_train_sample_ids = []
        for subject_id in self.meta_learning_subjects:
            meta_train_sample_ids.extend(self.index_by_subject_id[subject_id])

        pretraining_val_sample_ids = []
        for subject_id in self.pretraining_val_subjects:
            pretraining_val_sample_ids.extend(self.index_by_subject_id[subject_id])

        pretraining_test_sample_ids = []
        for subject_id in self.pretraining_test_subjects:
            pretraining_test_sample_ids.extend(self.index_by_subject_id[subject_id])

        print("[PhysioDataset] Pretraining samples per split")
        print(f"\t-# of train samples: {len(supervised_pretraining_train_sample_ids) + len(meta_train_sample_ids)}")
        print(f"\t\t-# supervised pretraining samples: {len(supervised_pretraining_train_sample_ids)}")
        print(f"\t\t-# meta-learning pretraining samples: {len(meta_train_sample_ids)}")
        print(f"\t-# of val samples: {len(pretraining_val_sample_ids)}")
        print(f"\t-# of test samples: {len(pretraining_test_sample_ids)}")
        
        #with open('pulse_db_supervised_training_ids', 'w') as pulse_db_file:
        #    for item in self.supervised_pretrain_subjects:
        #        pulse_db_file.write(f"{item}\n")
        #with open('pulse_db_metalearning_training_ids', 'w') as pulse_db_file:
        #    for item in self.meta_learning_subjects:
        #        pulse_db_file.write(f"{item}\n")
        #with open('pulse_db_validation_ids', 'w') as pulse_db_file:
        #    for item in self.pretraining_val_subjects:
        #        pulse_db_file.write(f"{item}\n")
        #with open('pulse_db_test_ids', 'w') as pulse_db_file:
        #    for item in self.pretraining_test_subjects:
        #        pulse_db_file.write(f"{item}\n")
        
        # Plot staticts per split
        if self.plot:
            plot_name = 'pretraining_dataset_overview.jpg'
            plot_train_val_test_samples_distribution(
                supervised_pretraining_train_sample_ids, 
                pretraining_val_sample_ids, 
                pretraining_test_sample_ids, 
                title=f'Pretraining Dataset Sample Distribution (From {len(self.subjects_for_pretraining)} subjects)', 
                savepath=os.path.join(self.savepath, plot_name)
                )

        return (
            SubsetRandomSampler(supervised_pretraining_train_sample_ids), 
            SubsetRandomSampler(meta_train_sample_ids), 
            SubsetRandomSampler(pretraining_val_sample_ids), 
            SubsetRandomSampler(pretraining_test_sample_ids)
            )
    
    # TO REMOVE
    #def get_pretraining_samplers_v2(self, test_ids_path):
    #    r"""
    #    Version 2 of pre-training samplers. 
    #    Loads specific test subject IDs from a text file, formats them to match 
    #    LMDB keys (adding 'p0' prefix), and splits the remaining subjects.
    #    """
    #    print(f"[PhysioDataset] Loading test subjects from: {test_ids_path}")
    #    
    #    # 1. Read and format Test IDs from text file
    #    with open(test_ids_path, 'r') as f:
    #        # Strip whitespace and prepend 'p0' to match LMDB format (e.g., 1234 -> p01234)
    #        raw_test_ids = [line.strip() for line in f if line.strip()]
    #        self.pretraining_test_subjects = [f"p0{tid}" for tid in raw_test_ids]
#
    #    # 2. Identify remaining subjects for Train/Val
    #    test_set = set(self.pretraining_test_subjects)
    #    remaining_subjects = [s for s in self.subjects_for_pretraining if s not in test_set]
    #    
    #    # Calculate validation ratio relative to the remaining pool
    #    # Original ratio was based on total; we adjust to keep absolute sizes similar
    #    _, val_ratio, _ = self.pretraining_split_ratio
    #    total_n = len(self.subjects_for_pretraining)
    #    target_val_size = int(total_n * val_ratio)
    #    val_relative_ratio = target_val_size / len(remaining_subjects)
#
    #    # 3. Split remaining into Train and Validation
    #    self.pretraining_train_subjects, self.pretraining_val_subjects = train_test_split(
    #        remaining_subjects,
    #        test_size=val_relative_ratio,
    #        random_state=self.seed
    #    )
#
    #    # 4. Split remaining training subjects into supervised and meta leanring subjects
    #    self.supervised_pretrain_subjects, self.meta_learning_subjects = train_test_split(
    #        self.pretraining_train_subjects,
    #        test_size=self.meta_split_ratio,
    #        random_state=self.seed
    #    )
#
    #    # 5. Map Subject IDs back to Sample IDs for Samplers
    #    def get_samples_for_subjects(subject_list):
    #        sample_ids = []
    #        for sid in subject_list:
    #            # Ensure the subject actually exists in the index
    #            if sid in self.index_by_subject_id:
    #                sample_ids.extend(self.index_by_subject_id[sid])
    #        return sample_ids
#
    #    sup_train_ids = get_samples_for_subjects(self.supervised_pretrain_subjects)
    #    meta_train_ids = get_samples_for_subjects(self.meta_learning_subjects)
    #    val_ids = get_samples_for_subjects(self.pretraining_val_subjects)
    #    test_ids = get_samples_for_subjects(self.pretraining_test_subjects)
#
    #    print(f"[PhysioDataset] Split Complete:")
    #    print(f"\t- Test Subjects (from file): {len(self.pretraining_test_subjects)}")
    #    print(f"\t- Train Subjects: {len(self.pretraining_train_subjects)} (Sup: {len(self.supervised_pretrain_subjects)}, Meta: {len(self.meta_learning_subjects)})")
    #    print(f"\t- Val Subjects: {len(self.pretraining_val_subjects)}")
#
    #    return (
    #        SubsetRandomSampler(sup_train_ids), 
    #        SubsetRandomSampler(meta_train_ids), 
    #        SubsetRandomSampler(val_ids), 
    #        SubsetRandomSampler(test_ids)
    #    )
    
    def __len__(self):
        return len(self.index_by_sample_id)

    def __getitem__(self, index):
        
        # ----------------------------------------
        # === Load raw input signals (PPG/ECG) ===
        # ----------------------------------------
        ppg = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-ppg".encode()), dtype="float32"))
        ecg = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-ecg".encode()), dtype="float32"))
       
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

        # ------------------------
        # === Load annotations ===
        # ------------------------
        
        # SBP, DBP, MAP values
        sbp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-sbp".encode()), dtype="float32"))
        dbp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-dbp".encode()), dtype="float32"))
        map = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{index}-map".encode()), dtype="float32"))
        
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

        return signals, annotation
    
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
        

def parseargs():
    parser = argparse.ArgumentParser(description="PhysioDataset")

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--figs_folder', default='./data_figs', type=str, help='where to save graphs from dataset analysis')
    parser.add_argument('--index_file_name', default='', type=str, help='name of the dataset index file')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.1,0.2', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--meta_train_split_ratio', default=0.2, type=float, help='percentage of subjects to extract from the training subjects for meta-learning')
    parser.add_argument('--min_subject_sample_number', default=0, type=int, help='minimum number of samples per subject to consider it valid, 0 means no limit')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False, help='plot dataset overview or not (# subjects per pretraining/personalization steps, # samples in pretraining splits)')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='whether to load only ecg or not')
    parser.add_argument('--drift_aware', action=argparse.BooleanOptionalAction, default=False, help='sample training subjects according to their SBP drift over time or nots')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--plot_aug', action=argparse.BooleanOptionalAction, default=False, help='plot signal augmentations or not')
    parser.add_argument('--loader_worker', default=8, type=int, help='number of loader workers')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()  
    
    # Figs folder for logging dataset statistics
    root_figs_folder = os.path.join(args.figs_folder, args.name) 
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)
    
    # Instantiate dataset
    dataset = PhysioDataset(
        seed=args.seed,
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        pretraining_split_ratio=list(map(float, args.pretraining_tr_val_tt_split_ratio.split(','))),
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        min_subject_sample_number=args.min_subject_sample_number,
        drift_aware=args.drift_aware,
        meta_split_ratio=args.meta_train_split_ratio,
        plot=args.plot, 
        savepath=root_figs_folder
    )
    
    (supervised_train_sampler, _,  val_sampler, test_sampler) = dataset.get_pretraining_samplers()
    
    # TO REMOVE
    #(supervised_train_sampler, _,  val_sampler, test_sampler) = dataset.get_pretraining_samplers_v2('./subject_list.txt')
    
    supervised_train_dataloader = DataLoader(dataset, sampler=supervised_train_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)    
    valid_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    
    # Get a sample batch and log shape along with I/O model outputs
    input_batch = next(iter(supervised_train_dataloader))
    sig = input_batch[0]
    sig = sig.unsqueeze(-1) if len(sig.shape) == 2 else sig
    idx = np.random.randint(0, sig.shape[0])

    annotation = input_batch[1]

    print(f"Input batch shape: {sig.shape}, Annotation batch shape: {annotation.shape}")

    sbp_val = annotation[idx, 0].squeeze().numpy()
    dbp_val = annotation[idx, 1].squeeze().numpy()
    map_val = annotation[idx, 2].squeeze().numpy()
        
    if args.plot:     
        if args.index_file_name != '':
            print(f'Plot age and gender distribution of valid subjects from {args.index_file_name}')
            plot_age_gender_distribution(dataset.subjects_for_pretraining, args.index_file_name, savepath=root_figs_folder)
        else:
            print('No index file provided...')
            
        calculate_dataloaders_mean_std(
            dataloaders=[supervised_train_dataloader, valid_dataloader, test_dataloader], 
            dataloaders_names=['Pretraining-Train', 'Pretraining-Val', 'Pretraining-Test'], 
            savepath=root_figs_folder
            ) 

        if args.ecg:            
            plot_signals(
                sig[idx, :, :].T, 
                fs=args.fs, 
                labels=['PPG', 'ECG'], 
                title=f'Input: PPG + ECG, Output: [SBP {sbp_val:.2f} - DBP {dbp_val:.2f} - MAP {map_val:.2f}]', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV']
                )
                
        else:
            plot_signals(
                sig[idx, :, :].T, 
                fs=args.fs,
                labels=['PPG'], 
                title=f'Input: PPG, Ouput: [SBP {sbp_val:.2f} - DBP {dbp_val:.2f} - MAP {map_val:.2f}]', 
                savepath=root_figs_folder, 
                ylabels=['a.u.']
                )