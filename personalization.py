import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import shutil
import pprint
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from data.dataset import PhysioDataset
from data.preprocessing_utils.split import split_train_val_test_personalization
from models.trainer import personalization_training_validation_testing
from training_utils.helpers import fixseed, generate_runname, get_model_architecture, set_trainable_parameters, parseargs
from training_utils.metrics import call_metric


def copy_chekpoint_for_resume_training(checkpoint_path, save_name, subject_id, pretrained_filename):
    # Copy the pretrained_checkpoint to the new subject directory to continue training without interfering with the pretrained model
    # N.B. the ckpt_path override the default_root_dir behavior in the Trainer
    new_checkpoint_dir = os.path.join(checkpoint_path, save_name, str(subject_id), 'ckpt')
    os.makedirs(new_checkpoint_dir, exist_ok=True) 
    new_checkpoint_path = os.path.join(new_checkpoint_dir, f'{model_name}.ckpt')
    shutil.copy(pretrained_filename, new_checkpoint_path)
    
    # IMPORTANT: remember to add ", ckpt_path=new_checkpoint_path" when calling .fit() on the trainer

def personalization(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device, save_models=False):
    
    # Get test subjects
    personalization_subjects = dataset.subjects_for_personalization
    
    # Load pretrained model
    pretrained_model = get_model_architecture(config)
    checkpoint = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
    print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
    pretrained_model.load_state_dict(checkpoint['model'])
    pretrained_model.eval() # to avoind incosistent results as explicitly reported at https://pytorch.org/tutorials/beginner/saving_loading_models.html
    pretrained_model = pretrained_model.to(device)
    set_trainable_parameters(model=pretrained_model, tune='none', config=config)
    
    # Pretrained model evaluation
    # Record outputs and targets
    if config['sig2sig']:
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
    else:
        all_test_outputs = np.empty((0, 2), dtype=float)
        all_test_targets = np.empty((0, 2), dtype=float)
        
    for subject_counter, subject_id in enumerate(personalization_subjects):
        
        print(f"{subject_counter}/{len(personalization_subjects)} testing subject {subject_id} without personalization")

        # Get all samples of a subject
        subject_sample_ids = dataset.index_by_subject_id[subject_id]
        
        # Get first X samples for training, the remaining are divided in X% for valid and test (chronological split)
        # Note the minimum amount of samples a subject can have from the dataset plot in figs
        _, _, test_idx = split_train_val_test_personalization(
            subject_sample_ids, 
            fixed_train_size=dataset.personalization_sample_number
            )
        
        test_sampler = SubsetRandomSampler(test_idx)
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=config["batch_size"], num_workers=config["loader_worker"])

        for _, batch in enumerate(test_dataloader):
            
            # Prepare data
            signals, targets = batch
            signals = signals.to(device)
            if len(signals.shape) == 2:
                signals = signals.unsqueeze(-1)

            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
            
            # Perform forward pass
            outputs = pretrained_model(signals)
        
            all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
            all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)
        
    # Log test metrics and plot them
    print('Results w/o personalization')
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'test_pretraining'), plot=True)    
    
    # Deallocate the pretrained model to free memory
    del pretrained_model
    gc.collect()
    
    ## --- Personalization ~ Training/Val/Test ---
    
    # Record outputs and targets
    if config['sig2sig']:
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
    else:
        all_test_outputs = np.empty((0, 2), dtype=float)
        all_test_targets = np.empty((0, 2), dtype=float)
        
    for subject_counter, subject_id in enumerate(personalization_subjects):
        
        print(f"{subject_counter}/{len(personalization_subjects)} personalizing subject {subject_id}")

        # Get all samples of a subject
        subject_sample_ids = dataset.index_by_subject_id[subject_id]
        
        # Get first X samples for training, the remaining are divided in X% for val and test (chronological split)
        # Note the minimum amount of samples a subject can have from the dataset plot in figs
        train_idx, val_idx, test_idx = split_train_val_test_personalization(
            subject_sample_ids, 
            fixed_train_size=dataset.personalization_sample_number
            )
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=config["batch_size"], num_workers=config["loader_worker"])
        val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=config["batch_size"], num_workers=config["loader_worker"])
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=config["batch_size"], num_workers=config["loader_worker"])
        
        test_targets, test_outputs = personalization_training_validation_testing(
            save_name=save_name,
            checkpoint_path=checkpoint_path,
            tensorboard_path=tensorboard_path,
            model_name=model_name,
            subject_id=subject_id,
            dataloaders={
                'train': train_dataloader,
                'val': val_dataloader,
                'test': test_dataloader
            },
            config=config,
            device=device,
            save_models=save_models
        )
        
        all_test_outputs = np.concatenate((all_test_outputs, test_outputs), axis=0)
        all_test_targets = np.concatenate((all_test_targets, test_targets), axis=0)
        
    print('Results w/ personalization')
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'test_personalization'), plot=True)    
    
    print(f"Personalization completed")


if __name__ == "__main__":
    global args
    args = parseargs()

    ## Setup config
    # Load the model class dynamically
    imported_module = __import__(args.model)
    model_name = args.model.split(sep='.')[-1]
    target_model = imported_module.__dict__[model_name].__dict__[model_name]

    # Select the gpu to be usesd
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch._C.device("cuda:0")

    # Setup paths
    run_name = generate_runname(model_name=target_model.__name__, exp_name=args.expname)
    checkpoint_path = os.path.join("./checkpoints/", args.expname, run_name)
    tensorboard_path = os.path.join("./tensorboard/", args.expname, run_name)
    figure_path = os.path.join("./figs/", args.expname, run_name)
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    print(f'Dataset folder: {os.path.join(args.dataset_folder, args.dataset_name)}')
    print(f'Checkpoint folder: {checkpoint_path}')
    print(f'Tensorboard folder: {tensorboard_path}')
    print(f'Figure folder: {tensorboard_path}')

    # Load configuration into a dict
    config = dict()    
    config.update(args.__dict__)  # add argparse
    config['model_name'] = model_name # add model name
    config['checkpoint_path'] = checkpoint_path
    config['tensorboard_path'] = tensorboard_path
    config['figure_path'] = figure_path
    
    print('Configuration for the run:')
    pprint.pprint(config, width=1)
    
    # Seed everything
    fixseed(config['seed'])
    pl.seed_everything(config['seed'])
    
    ## Build the dataset
    
    # RESP is loaded only if ECG is also loaded
    if config['resp'] and not config['ecg']:
        raise ValueError('RESP can be loaded only along with ECG')
    
    # PPG derivatives/PPG EMD/PPG freqs are loaded only if ecg (and optionally resp) are not present
    if (config['ppg_derivatives'] or config['ppg_emd'] or config['ppg_freqs']) and config['ecg']: 
        raise ValueError('PPG derivatives/emd/scalogram can be loaded only without ECG (and optionally RESP)')  
    
    # Load data and annotation
    dataset = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_ratio=config['pretraining_ratio'],
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        personalization_sample_number=config['personalization_sample_number'],
        mix_pretraining_subject_samples=config['mix_pretraining_subject_samples'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        resp=config['resp'],
        sig2sig=config['sig2sig'],
        ppg_derivatives=config['ppg_derivatives'],
        ppg_emd=config['ppg_emd'],
        ppg_freqs=config['ppg_freqs']
    )
    
    ## Personalization
    personalization(args.expname, model_name, dataset, checkpoint_path, tensorboard_path, config, device)




