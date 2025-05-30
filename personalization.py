import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import shutil
import pprint
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from data.online_dataset import OnlinePhysioDataset, ContinualLearningDataset
from data.preprocessing_utils.split import split_train_val_test_personalization
from training_utils.helpers import fixseed, generate_runname, get_model_architecture, set_trainable_parameters, parseargs, configure_optimizer_and_scheduler, EarlyStopping
from training_utils.metrics import call_metric, AverageMeter, get_metric_values


def personalization(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device, save_models=False):
    
    # Get test subjects
    personalization_subjects = dataset.base_dataset.subject_list
    
    # Load pretrained model
    pretrained_model = get_model_architecture(config)
    checkpoint = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
    print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
    pretrained_model.load_state_dict(checkpoint['model'])
    pretrained_model.eval() # to avoid incosistent results as explicitly reported at https://pytorch.org/tutorials/beginner/saving_loading_models.html
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

        # N.B. access with subject counter and not the subject id, which is provided by the subject list
        online_subject_id = continual_ds.base_dataset.subject_list[subject_counter] 
        continual_ds.set_active_subject(subject_id)
        
        # --- Simulate Online Training Loop for a Subject --- 
        sample_count = 0
        while True:
            break
            sample_data = continual_ds.get_next_sample() # Returns a dict
            if sample_data is None:
                break # No more samples for this subject

            signals = sample_data['sig_tensor'] # Use the tensor version
            targets = sample_data['annotation_tensor'] # Use the tensor version
            abp_valid = sample_data['abp_valid']
        
            # TODO: semi-supervised logic on unlabeled samples tbd
            if not abp_valid:
                continue
            
            # Prepare input data
            signals = signals.to(device)            
            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((targets[1][0].to(device), targets[1][1].to(device)), dim=-1)

            # Add batch dimension
            signals = signals.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
            # Perform forward pass
            with torch.no_grad(): # No gradient calculation during testing
                outputs = pretrained_model(signals)
                
            sample_count += 1
        
            all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
            all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)
            
        print(f"Finished processing {sample_count} samples for Subject {online_subject_id}.")
        
    # Log test metrics and plot them
    #print('Results w/o personalization')
    #_ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'test_pretraining'), plot=True)    
    
    # Deallocate the pretrained model to free memory
    del pretrained_model
    gc.collect()
    
    ## --- Personalization ---
    
    # Record outputs and targets
    if config['sig2sig']:
        all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
    else:
        all_test_outputs = np.empty((0, 2), dtype=float)
        all_test_targets = np.empty((0, 2), dtype=float)
        
    for subject_counter, subject_id in enumerate(personalization_subjects):
        
        print(f"{subject_counter}/{len(personalization_subjects)} personalizing subject {subject_id}")

        # N.B. access with subject counter and not the subject id, which is provided by the subject list
        online_subject_id = continual_ds.base_dataset.subject_list[subject_counter] 
        continual_ds.set_active_subject(subject_id)
        
        # Load pretrained model
        pretrained_model = get_model_architecture(config)
        checkpoint = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
        print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
        pretrained_model.load_state_dict(checkpoint['model'])
        pretrained_model.eval() # to avoind incosistent results as explicitly reported at https://pytorch.org/tutorials/beginner/saving_loading_models.html
        set_trainable_parameters(model=pretrained_model, tune='none', config=config)
        
        # Copy pretrained model for fine-tuning to avoid touching the original model
        personalized_model = copy.deepcopy(pretrained_model)
        personalized_model = personalized_model.to(device)
        
        # Deallocate the pretrained model to free memory
        del pretrained_model
        gc.collect()
        
        # Logging to TensorBoard Summary Writer
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))
        
        # Set the model to training model
        # Different choice can be performed based on the model architecture
        set_trainable_parameters(model=personalized_model, tune='last_layer', config=config)
        
        # Print trainable and non-trainable parameters
        trainable_params = sum(p.numel() for p in personalized_model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in personalized_model.parameters() if not p.requires_grad)

        print(f"Trainable parameters: {trainable_params} / {trainable_params + non_trainable_params} ({trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
        print(f"Non-trainable parameters: {non_trainable_params} / {trainable_params + non_trainable_params} ({non_trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
        
        # Optimizer (TODO: no scheduler at the moment)        
        optim_sched = configure_optimizer_and_scheduler(personalized_model, config)
        optimizer = optim_sched['optimizer']
        
        # --- Simulate Online Training Loop for a Subject --- 
        sample_count = 0
        train_losses = AverageMeter(name='train/loss')
        while True:
            sample_data = continual_ds.get_next_sample() # Returns a dict
            if sample_data is None:
                break # No more samples for this subject

            signals = sample_data['sig_tensor'] # Use the tensor version
            targets = sample_data['annotation_tensor'] # Use the tensor version
            abp_valid = sample_data['abp_valid']
        
            # TODO: semi-supervised logic on unlabeled samples tbd
            if not abp_valid:
                continue
            
            # Prepare input data
            signals = signals.to(device)            
            if config['sig2sig']:
                targets = targets.to(device)
            else:
                targets = torch.cat((targets[1][0].to(device), targets[1][1].to(device)), dim=-1)

            # Add batch dimension
            signals = signals.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
            outputs = personalized_model(signals)

            # Supervised loss
            supervised_loss = 0.
            if config['lambda_supervised'] != 0.:
                if config['criterion'] == 'MSELoss':
                    supervised_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    supervised_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")

            # Total loss
            loss = config['lambda_supervised'] * supervised_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log training loss
            train_losses.update(loss.item(), signals.size(0))
            writer.add_scalar('train/loss', loss.item(), sample_count) # N.B. logging on each sample
            
            sample_count += 1
        
            all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
            all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)
        
        
        writer.close()
    
        # Remove the writer
        #shutil.rmtree(os.path.join(tensorboard_path, str(subject_id)))
        
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
    
    # Instantiate OnlinePhysioDataset (the base for continual learning)
    online_physio_dataset_base = OnlinePhysioDataset(
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        resp=config['resp'],
        sig2sig=config['sig2sig'],
        ppg_derivatives=config['ppg_derivatives'],
        ppg_emd=config['ppg_emd'],
        ppg_freqs=config['ppg_freqs']
    )
    
    # --- Continual Learning Setup ---
    continual_ds = ContinualLearningDataset(online_physio_dataset_base)

    
    ## Personalization
    personalization(args.expname, model_name, continual_ds, checkpoint_path, tensorboard_path, config, device)




