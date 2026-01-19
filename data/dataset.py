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
    plot_train_val_test_samples_distribution, calculate_dataloaders_mean_std,     
    plot_drift_regime_counts, plot_sbp_drift_distribution
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
        
        print("[PhysioDataset] Computing SBP drift information for each subject ...")
        self.subject_drift_info = self.compute_subject_sbp_drift()
                    
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

    def compute_subject_sbp_drift(self):
        r"""
        Analyzes the 'drift' or physiological variance by retrieving 
        the ground-truth SBP labels for all windows associated with a subject.

        Statistical Metrics Computed:
        1.  **SBP Standard Deviation**: Measures the average dispersion of BP values 
            around the mean for that subject.
        2.  **SBP Range**: The absolute difference between the highest and lowest 
            recorded SBP (Max - Min).
        3.  **Sample Count**: The total number of valid windows used for the calculation.

        Returns
        ------------
        drift_info (dict): 
            A dictionary keyed by `subject_id`, where each value is a sub-dictionary 
            containing the calculated 'sbp_std_over_time', 'sbp_range', and 'num_samples'.
        """
        drift_info = {}

        for subject_id, sample_ids in self.index_by_subject_id.items():
            sbps = []

            for sid in sample_ids:
                sbp = np.squeeze(np.frombuffer(self.lmdbtxn.get(f"{sid}-sbp".encode()), dtype="float32"))
                sbps.append(sbp)
            sbps = np.asarray(sbps, dtype=np.float32)
            
            if len(sbps) < 2:
                continue

            drift_info[subject_id] = {
                "sbp_std_over_time": float(np.std(sbps)),
                "sbp_range": float(np.max(sbps) - np.min(sbps)),
                "num_samples": len(sbps)
            }

        return drift_info

    def drift_aware_train_split(self, train_subjects, train_drift_info):
        r"""
        Split training subjects into supervised and meta-learning sets
        in a drift-aware manner.

        Args:
            train_subjects (list): subject IDs in training split
            train_drift_info (dict): subject_id -> {
                "sbp_std_over_time": float,
                "drift_regime": "low" | "medium" | "high"
            }

        Returns:
            supervised_subjects (list)
            meta_learning_subjects (list)
        """

        rng = np.random.default_rng(self.seed)

        # Group subjects by drift regime
        regime_groups = defaultdict(list)
        for sid in train_subjects:
            info = train_drift_info.get(sid)
            if info is None:
                raise ValueError(f"Subject {sid} has no drift info")
            regime = info["drift_regime"]
            regime_groups[regime].append(sid)

        # Shuffle each regime group for randomness
        for regime in regime_groups:
            rng.shuffle(regime_groups[regime])

        total_train = len(train_subjects)
        target_meta = int(round(self.meta_split_ratio * total_train))

        regimes = ["low", "medium", "high"]
        per_regime_target = target_meta // 3
        remainder = target_meta % 3

        meta_subjects = []

        # First pass: equal quota per regime
        for regime in regimes:
            available = regime_groups.get(regime, [])
            take = min(len(available), per_regime_target)
            meta_subjects.extend(available[:take])
            regime_groups[regime] = available[take:]

        # Second pass: distribute remainder fairly
        if remainder > 0:
            # pool remaining subjects across regimes
            leftovers = []
            for regime in regimes:
                leftovers.extend(regime_groups.get(regime, []))

            rng.shuffle(leftovers)
            meta_subjects.extend(leftovers[:remainder])

            # Remove taken remainder subjects from regime_groups
            taken_set = set(meta_subjects)
            for regime in regimes:
                regime_groups[regime] = [
                    s for s in regime_groups.get(regime, []) if s not in taken_set
                ]

        # Remaining subjects go to supervised learning
        supervised_subjects = []
        for remaining in regime_groups.values():
            supervised_subjects.extend(remaining)

        # Safety checks
        assert len(meta_subjects) == target_meta
        assert len(meta_subjects) + len(supervised_subjects) == total_train
        assert len(set(meta_subjects).intersection(supervised_subjects)) == 0

        return supervised_subjects, meta_subjects

    
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
        
        train_drift_info = {k: v for k, v in self.subject_drift_info.items() if k in self.pretraining_train_subjects}
        
        values = np.array([v["sbp_std_over_time"] for v in train_drift_info.values()])
        q25, q75 = np.percentile(values, [25, 75])
        
        for subject_id, info in train_drift_info.items():
            if info["sbp_std_over_time"] <= q25:
                info["drift_regime"] = "low"
            elif info["sbp_std_over_time"] >= q75:
                info["drift_regime"] = "high"
            else:
                info["drift_regime"] = "medium"
                
        if self.drift_aware:    
            self.supervised_pretrain_subjects, self.meta_learning_subjects = self.drift_aware_train_split(
                self.pretraining_train_subjects,
                train_drift_info
            )
        else:
            self.supervised_pretrain_subjects, self.meta_learning_subjects = train_test_split(
                self.pretraining_train_subjects,
                test_size=self.meta_split_ratio,
                random_state=self.seed
            )

        def regime_counts(subjects, drift_info):
            return Counter(
                drift_info[s]["drift_regime"]
                for s in subjects
                if s in drift_info
            )

        print("[PhysioDataset] Meta-learning regimes:", regime_counts(self.meta_learning_subjects, train_drift_info))
        print("[PhysioDataset] Supervised regimes:", regime_counts(self.supervised_pretrain_subjects, train_drift_info))
        
        if self.plot:
            plot_drift_regime_counts(
                train_drift_info,
                self.pretraining_train_subjects,
                q25=q25,
                q75=q75,
                title='drift_regime_counts_pretraining_train_split',
                savepath=self.savepath
            )
            
            plot_sbp_drift_distribution(
                train_drift_info,
                self.pretraining_train_subjects,
                title='sbp_drift_distribution_pretraining_train_split',
                savepath=self.savepath
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