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
from preprocessing_utils.data_visualization import plot_signals, plot_subject_sample_distribution, plot_train_val_test_samples_distribution, calculate_dataloaders_mean_std, plot_bp_pattern_distribution
from preprocessing_utils.split import split_train_val_test


class PhysioDataset(Dataset):
    def __init__(self,
                 seed, 
                 lmdb_folder, 
                 pretraining_split_ratio=[0.7, 0.1, 0.2], 
                 mix_pretraining_subject_samples=False, 
                 fs=125, 
                 input_seq_len_s=5, 
                 ecg=False, 
                 sig2sig=False,
                 min_subject_sample_number=0, 
                 contrastive=False,
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
        self.sig2sig = sig2sig
        self.fs = fs
        self.input_seq_len_s = input_seq_len_s
        self.contrastive = contrastive
        
        # Plot arguments
        self.plot = plot
        self.savepath = savepath

        # Dataset split
        self.total_subject_n = len(self.subjects_for_pretraining)
        self.pretraining_split_ratio = pretraining_split_ratio # To divide pretraining from personalization, and then to divide the pretraining dataset
        self.mix_pretraining_subject_samples = mix_pretraining_subject_samples # Whether to split train/val/test during pretraining subjectwise or not
        
        
        if self.plot:
            if self.min_subject_sample_number > 0:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, f'pretraining_subject_sample_distribution_min_sample_{self.min_subject_sample_number}.jpg'))
            else:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, 'pretraining_subject_sample_distribution.jpg'))
        
        print("{:s} initialized with following configuration:".format(self.__class__.__name__))
        pprint.pprint(
            {
                "Total Subjects": len(self.subjects_for_pretraining),
                "Total Samples": len(self.index_by_sample_id)
            }
        )
        
        # ---------------------------------------------------------
        # === Precompute BP categories and build index for contrastive sampling
        # ---------------------------------------------------------
        # Since its required only for training, we will nedd the sample ids for trianing
        pretraining_train_sample_ids, _, _ = self.get_pretraining_samplers()

        self.index_by_bp_category = {0: [], 1: [], 2: [], 3: [], 4: []}

        for sample_id in pretraining_train_sample_ids:
            sbp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{sample_id}-sbp".encode()), dtype="float32"))
            dbp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{sample_id}-dbp".encode()), dtype="float32"))
            
            cat = self.bp_to_category(sbp, dbp)
            self.index_by_bp_category[cat].append(sample_id)

        print("Built BP category index:")
        for cat, ids in self.index_by_bp_category.items():
            print(f"\tCategory {cat}: {len(ids)} samples")
    
    def check_subjects_list(self, min_subject_sample_number=0):
        # Considering preprocessing in the mimic_iii, when a subject has no valid samples,
        # its ID is in the self.index_by_subject_id but not in the self.index_by_sample_id as the for loop inside
        # with lmdbenv.begin(write=True) as txn: deos not make this check
        invalid_subjects = list()
        for subject in self.subjects_for_pretraining:
            if len(self.index_by_subject_id[subject]) <= min_subject_sample_number:
                invalid_subjects.append(subject)
        
        if len(invalid_subjects) > 0:
            print("Invalid subjects found in the dataset, removing them ...")
            for subject in invalid_subjects:
                self.subjects_for_pretraining.remove(subject)
                del self.index_by_subject_id[subject]

        # Shorten each subject list to min_subject_sample_number
        if min_subject_sample_number > 0:
            for subject in self.subjects_for_pretraining:
                if len(self.index_by_subject_id[subject]) > min_subject_sample_number:
                    self.index_by_subject_id[subject] = self.index_by_subject_id[subject][:min_subject_sample_number]
    
    def get_pretraining_samplers(self):
        
        print("{:s} {:s}".format(self.__class__.__name__, sys._getframe().f_code.co_name))
        
        if self.mix_pretraining_subject_samples:
        
            print("Pretraining train/val/test sets samples are sampled from the same subject set")
            
            pretraining_train_sample_ids = []
            pretraining_val_sample_ids = []
            pretraining_test_sample_ids = []
            
            # Loop over subjects for pretraining
            for subject in self.subjects_for_pretraining:
                
                # Get all samples of a subject
                subject_sample_ids = self.index_by_subject_id[subject]
                
                # First X% samples for training, next Y% for valid, and remaining for test
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
        
    def bp_to_category(self, sbp, dbp):
        """
        Categorize BP based on American Heart Association (AHA) guidelines:
        Source: https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
        
        - Category 0: Normal
            SBP < 120 mmHg AND DBP < 80 mmHg
        - Category 1: Elevated
            SBP 120 - 129 mmHg AND DBP < 80 mmHg
        - Category 2: Hypertension Stage 1
            SBP 130 - 139 mmHg OR DBP 80 - 89 mmHg
        - Category 3: Hypertension Stage 2
            SBP 140 - 180 mmHg OR DBP 90 - 120 mmHg
        - Category 4: Hypertensive Crisis
            SBP > 180 mmHg AND/OR DBP > 120 mmHg
        """
        if sbp < 120 and dbp < 80:
            return 0
        elif 120 <= sbp < 130 and dbp < 80:
            return 1
        elif (130 <= sbp < 140) or (80 <= dbp < 90):
            return 2
        elif (140 <= sbp <= 180) or (90 <= dbp <= 120):
            return 3
        else:
            return 4

    def __len__(self):
        return len(self.index_by_sample_id)
    
    def __getitem__(self, index):
        # ---------------------------------------------------------
        # === Load raw input signals (PPG or PPG+ECG) ===
        # ---------------------------------------------------------
        if self.ecg:
            ppg = np.frombuffer(
                self.lmdbtxn.get(f"{index}-ppg".encode()), dtype="float32")
            ecg = np.frombuffer(
                self.lmdbtxn.get(f"{index}-ecg".encode()), dtype="float32")

            # Shape: [time, 2]  (PPG, ECG)
            sig = np.stack((ppg, ecg), axis=-1)
        else:
            sig = np.frombuffer(
                self.lmdbtxn.get(f"{index}-ppg".encode()), dtype="float32")

        # Make arrays writable
        sig = np.require(sig, requirements=['O', 'W'])
        sig.setflags(write=1)
        
        # Cast to torch tensor
        signals = torch.tensor(sig)
            
        # ---------------------------------------------------------
        # === Load annotations ===
        #   - If sig2sig=True: full ABP waveform
        #   - Else: scalar SBP, DBP, MAP values
        # ---------------------------------------------------------
        if self.sig2sig:
            abp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-abp".encode()), dtype="float32"))
            
            # Make arrays writable
            abp = np.require(abp, requirements=['O', 'W'])
            abp.setflags(write=1)
            
            # Cast to torch tensor
            annotation = torch.tensor(abp)
        else:
            sbp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-sbp".encode()), dtype="float32"))
            dbp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-dbp".encode()), dtype="float32"))
            map = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{index}-map".encode()), dtype="float32"))
            
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

        # ---------------------------------------------------------
        # === Default return (val/test or no contrastive mode) ===
        # ---------------------------------------------------------
        if not self.contrastive:
            return signals, annotation
        
        # ---------------------------------------------------------
        # === Guardrails for contrastive mode ===
        # ---------------------------------------------------------
        if self.contrastive and self.mix_pretraining_subject_samples:
            raise ValueError("Contrastive learning not possible "
                            "when pretraining samples are mixed among subjects.")

        # -----------------------------------------------------------------
        # === Build positive pair (get sample of the same BP phenotype) ===
        # -----------------------------------------------------------------
        # Get anchor BP category
        anchor_sbp = annotation[0].item()
        anchor_dbp = annotation[1].item()
        bp_cat = self.bp_to_category(anchor_sbp, anchor_dbp)

        subject_id, _ = self.index_by_sample_id[index]
        
        # If subject is not in training set, skip contrastive
        # In test mode, may useful to get BP category distribution
        if subject_id in self.pretraining_val_subjects:
            return signals, annotation
        if subject_id in self.pretraining_test_subjects:
            return signals, annotation, bp_cat
        if subject_id not in self.pretraining_train_subjects:
            raise ValueError("Subject ID not found in train/val/test splits.")

        # Sample a positive from same BP category
        pos_candidates = self.index_by_bp_category[bp_cat]
        pos_index = random.choice(pos_candidates)
        while pos_index == index:
            pos_index = random.choice(pos_candidates)

        pos_ppg = np.frombuffer(self.lmdbtxn.get(f"{pos_index}-ppg".encode()), dtype="float32")
        if self.ecg:
            pos_ecg = np.frombuffer(self.lmdbtxn.get(f"{pos_index}-ecg".encode()), dtype="float32")
            pos_sig = np.stack((pos_ppg, pos_ecg), axis=-1)
        else:
            pos_sig = pos_ppg
        
        # Make arrays writable
        pos_sig = np.require(pos_sig, requirements=['O', 'W'])
        pos_sig.setflags(write=1)
        
        # Cast to torch tensor
        pos_signals = torch.tensor(pos_sig)
       
        if self.sig2sig:
            pos_abp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{pos_index}-abp".encode()), dtype="float32"))
            
            # Make arrays writable
            pos_abp = np.require(pos_abp, requirements=['O', 'W'])
            pos_abp.setflags(write=1)
            
             # Cast to torch tensor
            pos_annotation = torch.tensor(pos_abp)
        else:
            pos_sbp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{pos_index}-sbp".encode()), dtype="float32"))
            pos_dbp = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{pos_index}-dbp".encode()), dtype="float32"))
            pos_map = np.squeeze(np.frombuffer(
                self.lmdbtxn.get(f"{pos_index}-map".encode()), dtype="float32"))
            
            # Make arrays writable
            pos_sbp = np.require(pos_sbp, requirements=['O', 'W'])
            pos_sbp.setflags(write=1)
            pos_dbp = np.require(pos_dbp, requirements=['O', 'W'])
            pos_dbp.setflags(write=1)
            pos_map = np.require(pos_map, requirements=['O', 'W'])
            pos_map.setflags(write=1)
            
            # Cast to torch tensor
            pos_annotation = torch.stack([
                torch.tensor(pos_sbp), 
                torch.tensor(pos_dbp), 
                torch.tensor(pos_map)
            ], dim=-1)
        
        # ---------------------------------------------------------
        # === Return original and positive pair ===
        # Contrastive loss will treat all embeddings
        # from the same subject_id as positives
        # ---------------------------------------------------------
        return (signals, annotation, bp_cat), (pos_signals, pos_annotation, bp_cat)


def parseargs():
    parser = argparse.ArgumentParser(description="Dataset overview")

    parser.add_argument('--dataset_folder', default='./lmdb', type=str, help='path to the dataset to analyze')
    parser.add_argument('--name', default='test', type=str, help='name of the processed dataset')
    parser.add_argument('--save_path', default='./data_figs', type=str, help='where to save graphs from dataset analysis')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--pretraining_tr_val_tt_split_ratio', default='0.7,0.1,0.2', type=str, help='ratio for train, validation, and test split, comma separated')
    parser.add_argument('--min_subject_sample_number', default=0, type=int, help='minimum number of samples per subject to consider it valid, 0 means no limit')
    parser.add_argument('--mix_pretraining_subject_samples', default='True', type=lambda x: bool(strtobool(x)), help='whether to mix pretraining subject samples among train/val/test or not')
    parser.add_argument('--plot', default='False', type=lambda x: bool(strtobool(x)), help='plot dataset overview or not (# subjects per pretraining/personalization steps, # samples in pretraining splits)')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--contrastive', default='False', type=lambda x: bool(strtobool(x)), help='whether to load the positive pair of a subject sample or not')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--plot_aug', default='False', type=lambda x: bool(strtobool(x)), help='plot signal augmentations or not')
    parser.add_argument('--loader_worker', default=4, type=int, help='number of loader workers')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()  
    
    root_figs_folder = os.path.join(args.save_path, args.name) 
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)
    
    dataset = PhysioDataset(
        seed=args.seed,
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        pretraining_split_ratio=list(map(float, args.pretraining_tr_val_tt_split_ratio.split(','))),
        mix_pretraining_subject_samples=args.mix_pretraining_subject_samples,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        sig2sig=args.sig2sig,
        contrastive=args.contrastive,
        min_subject_sample_number=args.min_subject_sample_number,
        plot=args.plot, 
        savepath=root_figs_folder
    )
    
    # Pretraining & personalization datasets statistics
    (train_sampler, val_sampler, test_sampler) = dataset.get_pretraining_samplers()
    
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)    
    valid_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    
    if args.mix_pretraining_subject_samples:
        calculate_dataloaders_mean_std(
            dataloaders=[train_dataloader, valid_dataloader, test_dataloader], 
            dataloaders_names=['Pretraining-Train', 'Pretraining-Val', 'Pretraining-Test'], 
            savepath=root_figs_folder
            ) 
    else:
        calculate_dataloaders_mean_std(
            dataloaders=[train_dataloader, valid_dataloader, test_dataloader], 
            dataloaders_names=['No-Mix-Pretraining-Train', 'No-Mix-Pretraining-Val', 'No-Mix-Pretraining-Test'], 
            savepath=root_figs_folder
            )
    
    input_batch = next(iter(train_dataloader))
    sig = input_batch[0]
    sig = sig.unsqueeze(-1) if len(sig.shape) == 2 else sig
    idx = np.random.randint(0, sig.shape[0])
    
    annotation = input_batch[1]
    subject_ids = input_batch[2]
    
    print(f"Input batch shape: {sig.shape}, Annotation batch shape: {annotation.shape}, Subject IDs shape: {subject_ids.shape}")
    print(f"Subject ID: {subject_ids[idx]}, Input signal shape: {sig[idx].shape}, Annotation shape: {annotation[idx].shape}")
    
    if args.sig2sig:
        sig = sig[idx, :, :].squeeze().numpy()
        abp = annotation[idx, :].squeeze().numpy()
    else:
        sbp_val = annotation[idx, 0].squeeze().numpy()
        dbp_val = annotation[idx, 1].squeeze().numpy()
        map_val = annotation[idx, 2].squeeze().numpy()
    
    # Note that the train_dataloader will already return the required signals specified by the conditions
    if args.ecg:
        
        if args.sig2sig:
            sigs = np.concatenate((sig, abp[:, np.newaxis]), axis=1)
            plot_signals(
                sigs.T, 
                fs=args.fs, 
                labels=['PPG', 'ECG', 'ABP'], 
                title=f'Input: PPG + ECG, Output: ABP', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV', 'mmHg']
            )
        else:
            plot_signals(
                sig[idx, :, :].T, fs=args.fs, 
                labels=['PPG', 'ECG'], 
                title=f'Input: PPG + ECG, Output: [SBP {sbp_val:.2f} - DBP {dbp_val:.2f} - MAP {map_val:.2f}]', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV']
                )
            
    else:
        
        if args.sig2sig:
            sigs = np.concatenate((sig, abp[:, np.newaxis]), axis=1)
            plot_signals(
                sigs.T, 
                fs=args.fs, 
                labels=['PPG', 'ABP'], 
                title=f'Input: PPG, Output: ABP', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mmHg']
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