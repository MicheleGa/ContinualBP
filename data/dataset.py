import os 
import argparse
import pickle
import numpy as np
import random
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from preprocessing_utils.data_visualization import (
    plot_signals, plot_subject_sample_distribution, plot_age_gender_distribution,
    plot_train_val_test_samples_distribution, calculate_dataloaders_mean_std,     
)
from preprocessing_utils.split import split_train_val_test


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
        self.total_subject_n = len(self.subjects_for_pretraining)
        self.pretraining_split_ratio = pretraining_split_ratio # To divide pretraining from personalization, and then to divide the pretraining dataset
        self.meta_split_ratio = meta_split_ratio

        if self.plot:
            if self.min_subject_sample_number > 0:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, f'pretraining_subject_sample_distribution_min_sample_{self.min_subject_sample_number}.jpg'))
            else:
                plot_subject_sample_distribution(self.index_by_subject_id, self.subjects_for_pretraining, savepath=os.path.join(savepath, 'pretraining_subject_sample_distribution.jpg'))
        
        print(f"[PhysioDataset] Initialized with following configuration")
        print(f"\t-Total Subjects: {len(self.subjects_for_pretraining)}")
        print(f"\t-Total Samples: {len(self.index_by_sample_id)}")
                    
    def check_subjects_list(self, min_subject_sample_number=0):
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
        """Get train/val/test samplers for pretraining dataset"""
            
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
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--plot_aug', action=argparse.BooleanOptionalAction, default=False, help='plot signal augmentations or not')
    parser.add_argument('--loader_worker', default=8, type=int, help='number of loader workers')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()  
    
    root_figs_folder = os.path.join(args.figs_folder, args.name) 
    if not os.path.exists(root_figs_folder):
        os.makedirs(root_figs_folder)
      
    dataset = PhysioDataset(
        seed=args.seed,
        lmdb_folder=os.path.join(args.dataset_folder, args.name),
        pretraining_split_ratio=list(map(float, args.pretraining_tr_val_tt_split_ratio.split(','))),
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        ecg=args.ecg,
        min_subject_sample_number=args.min_subject_sample_number,
        plot=args.plot, 
        savepath=root_figs_folder
    )
    
    # Pretraining & personalization datasets statistics
    (supervised_train_sampler, _,  val_sampler, test_sampler) = dataset.get_pretraining_samplers()
    
    supervised_train_dataloader = DataLoader(dataset, sampler=supervised_train_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)    
    valid_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.loader_worker, pin_memory=True)

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
            
        if args.mix_pretraining_subject_samples:
            calculate_dataloaders_mean_std(
                dataloaders=[supervised_train_dataloader, valid_dataloader, test_dataloader], 
                dataloaders_names=['Pretraining-Train', 'Pretraining-Val', 'Pretraining-Test'], 
                savepath=root_figs_folder
                ) 
        else:
            calculate_dataloaders_mean_std(
                dataloaders=[supervised_train_dataloader, valid_dataloader, test_dataloader], 
                dataloaders_names=['No-Mix-Pretraining-Train', 'No-Mix-Pretraining-Val', 'No-Mix-Pretraining-Test'], 
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