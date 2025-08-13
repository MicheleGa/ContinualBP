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
from preprocessing_utils.data_visualization import plot_signals, plot_pretraining_personalization_subjects_distribution, plot_subject_sample_distribution, plot_train_val_test_samples_distribution, calculate_dataloaders_mean_std, calculate_personalization_subjects_mean_std
from preprocessing_utils.augmentations import RandomAugmentor, Identity, Jitter, TimeWarp, Scaling, MagnitudeWarp, Flip
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
                 resp=False, 
                 sig2sig=False,
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
                
        # Which input data to load (PPG, PPG + ECG, PPG + ECG + RESP), PPG is always loaded
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        self.fs = fs
        self.input_seq_len_s = input_seq_len_s
        
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

    def __len__(self):
        return len(self.index_by_sample_id)
    
    def __getitem__(self, index):
        sample = dict()

        # Input data        
        if self.ecg and not self.resp:
            sample['ppg'] = np.frombuffer(self.lmdbtxn.get("{}-ppg".format(index).encode()), dtype="float32")
            sample['ecg'] = np.frombuffer(self.lmdbtxn.get("{}-ecg".format(index).encode()), dtype="float32")
            
            sample['sig'] = np.concatenate(
                (
                    np.expand_dims(sample['ppg'], axis=-1), 
                    np.expand_dims(sample['ecg'], axis=-1)
                ), 
                axis=-1)
        elif self.ecg and self.resp:
            sample['ppg'] = np.frombuffer(self.lmdbtxn.get("{}-ppg".format(index).encode()), dtype="float32")
            sample['ecg'] = np.frombuffer(self.lmdbtxn.get("{}-ecg".format(index).encode()), dtype="float32")
            sample['resp'] = np.frombuffer(self.lmdbtxn.get("{}-resp".format(index).encode()), dtype="float32")
            
            sample['sig'] = np.concatenate(
                (
                    np.expand_dims(sample['ppg'], axis=-1), 
                    np.expand_dims(sample['ecg'], axis=-1),
                    np.expand_dims(sample['resp'], axis=-1)
                ), 
                axis=-1)
        else:
            sample['ppg'] = np.frombuffer(self.lmdbtxn.get("{}-ppg".format(index).encode()), dtype="float32")
            
            sample['sig'] = sample['ppg']
            
        # Annotation
        if self.sig2sig:
            sample['abp'] = np.squeeze(np.frombuffer(self.lmdbtxn.get("{}-abp".format(index).encode()), dtype="float32"))
            
            # Ensure that the signal and annotation are numpy arrays with writeable flags
            for k in sample:
                sample[k] = np.require(sample[k], requirements=['O', 'W'])
                sample[k].setflags(write=1)
            
            # Cast to torch tensor
            signals = torch.tensor(sample['sig'])
            abp = torch.tensor(sample['abp'])
            
            return signals, abp
        else:
            sample['sbp'] = np.squeeze(np.frombuffer(self.lmdbtxn.get("{}-sbp".format(index).encode()), dtype="float32"))
            sample['dbp'] = np.squeeze(np.frombuffer(self.lmdbtxn.get("{}-dbp".format(index).encode()), dtype="float32"))
        
            for k in sample:
                sample[k] = np.require(sample[k], requirements=['O', 'W'])
                sample[k].setflags(write=1)
            
            # Cast to torch tensor
            signals = torch.tensor(sample['sig'])
            sbp_val = torch.tensor(sample['sbp']).unsqueeze(-1)
            dbp_val = torch.tensor(sample['dbp']).unsqueeze(-1)
            
            return signals, [sbp_val, dbp_val]
    

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
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
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
        resp=args.resp,
        sig2sig=args.sig2sig,
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
    annotation = input_batch[1]
    
    idx = np.random.randint(0, sig.shape[0])
    if args.sig2sig:
        sig = sig[idx, :, :].squeeze().numpy()
        abp = annotation[idx, :].squeeze().numpy()
    else:
        sbp_val = annotation[0][idx].squeeze().numpy()
        dbp_val = annotation[1][idx].squeeze().numpy()
    
    # Note that the train_dataloader will already return the required signals specified by the conditions
    if args.ecg and not args.resp:
        
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
                title=f'Input: PPG + ECG, Output: [SBP {sbp_val:.2f} - DBP {dbp_val:.2f}]', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV']
                )
        
        if args.plot_aug:
            augments = RandomAugmentor(
                [
                    Identity(prob=0.2),
                    Jitter(prob=0.2),
                    Scaling(prob=0.2),
                    MagnitudeWarp(prob=0.2),
                    Flip(prob=0.2)
                ]
            )
            inp_sigs_augs = []
            for inp_mod in range(sig.shape[-1]):
                sig_aug = augments(sig[:, :, inp_mod].squeeze())
                sig_aug = sig_aug.unsqueeze(-1) if len(sig_aug.shape) == 2 else sig
                inp_sigs_augs.append(sig_aug)
            inp_sigs_augs = torch.cat(inp_sigs_augs, dim=-1)
            plot_signals(
                inp_sigs_augs[idx, :, :].T, 
                fs=args.fs, 
                labels=['PPG', 'ECG'], 
                title=f'Augmented PPG + ECG', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mmV']
            )
            
    elif args.ecg and args.resp:
        
        if args.sig2sig:
            sigs = np.concatenate((sig, abp[:, np.newaxis]), axis=1)
            plot_signals(
                sigs.T, 
                fs=args.fs, 
                labels=['PPG', 'ECG', 'RESP', 'ABP'], 
                title=f'Input: PPG + ECG + RESP, Output: ABP', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV', 'pm', 'mmHg']
            )
        else:    
            plot_signals(
                sig[idx, :, :].T, 
                fs=args.fs, 
                labels=['PPG', 'ECG', 'RESP'], 
                title=f'Input: PPG + ECG + RESP, Output: [SBP {sbp_val:.2f} - DBP {dbp_val:.2f}]', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV', 'pm']
                )
        
        if args.plot_aug:
            augments = RandomAugmentor(
                [
                    Identity(prob=0.2),
                    Jitter(prob=0.2),
                    Scaling(prob=0.2),
                    MagnitudeWarp(prob=0.2),
                    Flip(prob=0.2)
                ]
            )
            inp_sigs_augs = []
            for inp_mod in range(sig.shape[-1]):
                sig_aug = augments(sig[:, :, inp_mod].squeeze())
                sig_aug = sig_aug.unsqueeze(-1) if len(sig_aug.shape) == 2 else sig
                inp_sigs_augs.append(sig_aug)
            inp_sigs_augs = torch.cat(inp_sigs_augs, dim=-1)
            plot_signals(
                inp_sigs_augs[idx, :, :].T, 
                fs=args.fs, 
                labels=['PPG', 'ECG', 'RESP'], 
                title=f'Augmented PPG + ECG + RESP', 
                savepath=root_figs_folder, 
                ylabels=['a.u.', 'mV', 'pm']
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
                title=f'Input: PPG, Ouput: [SBP {sbp_val:.2f} - DBP {dbp_val:.2f}]', 
                savepath=root_figs_folder, 
                ylabels=['a.u.']
                )
        
        if args.plot_aug:
            augments = RandomAugmentor(
                [
                    Identity(prob=0.2),
                    Jitter(prob=0.2),
                    Scaling(prob=0.2),
                    MagnitudeWarp(prob=0.2),
                    Flip(prob=0.2)
                ]
            )
            inp_sigs_augs = []
            for inp_mod in range(sig.shape[-1]):
                sig_aug = augments(sig[:, :, inp_mod].squeeze())
                sig_aug = sig_aug.unsqueeze(-1) if len(sig_aug.shape) == 2 else sig
                inp_sigs_augs.append(sig_aug)
            inp_sigs_augs = torch.cat(inp_sigs_augs, dim=-1)
            plot_signals(
                inp_sigs_augs[idx, :, :].T, 
                fs=args.fs, 
                labels=['PPG'], 
                title=f'Augmented PPG', 
                savepath=root_figs_folder, 
                ylabels=['a.u.']
            )