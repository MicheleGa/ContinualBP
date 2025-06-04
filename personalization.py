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
from data.preprocessing_utils.data_visualization import plot_signals


def process_batch_inference(signals_batch, targets_batch, model, device):
    """Process a batch for inference only. Assumes signals_batch and targets_batch are already batched tensors."""
    if signals_batch.size(0) == 0: # Check if batch is empty
        return None, None

    # Move to device
    batched_signals = signals_batch.to(device)
    batched_targets = targets_batch.to(device)

    # DataLoader or Dataset can sometimes return an additional singleton. Squeezing is the way
    # Check and squeeze extra dimension for signals_batch
    if batched_signals.dim() == 4 and batched_signals.size(1) == 1:
        batched_signals = batched_signals.squeeze(1) # Remove the singleton dimension

    # Check and squeeze extra dimension for targets_batch
    # Assuming the extra dimension, if present, is also at index 1 and has a size of 1
    if batched_targets.dim() == 3 and batched_targets.size(1) == 1:
        batched_targets = batched_targets.squeeze(1)

    # Perform forward pass (inference only)
    with torch.no_grad():
        outputs = model(batched_signals)

    return outputs, batched_targets

def process_batch_training(signals_batch, targets_batch, model, optimizer, config, device, writer, base_sample_count):
    """Process a batch for training (forward + backward pass). Allows multiple passes on the same batch."""
    if signals_batch.size(0) == 0: # Check if batch is empty
        return None, None, 0.0

    # Move to device
    batched_signals = signals_batch.to(device)
    batched_targets = targets_batch.to(device)

    # DataLoader or Dataset can sometimes return an additional singleton. Squeezing is the way
    # Check and squeeze extra dimension for signals_batch
    if batched_signals.dim() == 4 and batched_signals.size(1) == 1:
        batched_signals = batched_signals.squeeze(1) # Remove the singleton dimension

    # Check and squeeze extra dimension for targets_batch
    # Assuming the extra dimension, if present, is also at index 1 and has a size of 1
    if batched_targets.dim() == 3 and batched_targets.size(1) == 1:
        batched_targets = batched_targets.squeeze(1)

    total_loss_this_batch = 0.0
    num_passes = config.get('num_passes', 1) # Get num_passes from config, default to 1

    for pass_idx in range(num_passes):
        # Forward pass
        outputs = model(batched_signals)

        # Calculate loss
        supervised_loss = 0.
        if config['lambda_supervised'] != 0.:
            if config['criterion'] == 'MSELoss':
                supervised_loss = F.mse_loss(outputs, batched_targets)
            elif config['criterion'] == 'SmoothL1Loss':
                supervised_loss = F.smooth_l1_loss(outputs, batched_targets)
            else:
                raise ValueError("Invalid criterion ...")

        # Total loss
        loss = config['lambda_supervised'] * supervised_loss

        # Backpropagation and optimization
        optimizer.zero_grad() # Zero gradients for each pass
        loss.backward()
        optimizer.step()
        
        total_loss_this_batch += loss.item()

        # Log training loss for this pass (optional, can be average over passes)
        # For simplicity, we'll log the average loss for the batch after all passes
        # If you want to see per-pass loss, adjust base_sample_count or add a pass_idx to log tag
        
    avg_loss_this_batch = total_loss_this_batch / num_passes
    writer.add_scalar('train/loss', avg_loss_this_batch, base_sample_count)

    return outputs, batched_targets, avg_loss_this_batch

def pretrained_model_inference(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device, save_models=False):

    # Get test subjects
    personalization_subjects = dataset.base_dataset.subject_list

    # Load pretrained model
    pretrained_model = get_model_architecture(config)
    checkpoint = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
    print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
    pretrained_model.load_state_dict(checkpoint['model'])
    pretrained_model.eval() # to avoid inconsistent results as explicitly reported at https://pytorch.org/tutorials/beginner/saving_loading_models.html
    pretrained_model = pretrained_model.to(device)
    set_trainable_parameters(model=pretrained_model, tune='none', config=config)

    # Pretrained model evaluation
    # Use lists to collect outputs and targets, then concatenate once at the end
    all_test_outputs_list = []
    all_test_targets_list = []

    for subject_counter, subject_id in enumerate(personalization_subjects):

        print(f"{subject_counter}/{len(personalization_subjects)} testing subject {subject_id} without personalization")

        continual_ds.set_active_subject(subject_id) # Set the active subject for the dataset
        # Create a DataLoader for the current subject
        # Use num_workers from config
        subject_dataloader = DataLoader(continual_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['loader_worker'])

        sample_count = 0
        for batch_data in subject_dataloader:
            signals_batch = batch_data['sig_tensor']
            targets_batch = batch_data['annotation_tensor']
            abp_valid_batch = batch_data['abp_valid']

            # Filter out invalid ABP samples if needed
            valid_indices = abp_valid_batch.squeeze().bool()
            signals_batch = signals_batch[valid_indices]
            targets_batch = targets_batch[valid_indices]

            if signals_batch.size(0) == 0: # Skip if batch is empty after filtering
                continue

            outputs, targets = process_batch_inference(signals_batch, targets_batch, pretrained_model, device)
            if outputs is not None:
                all_test_outputs_list.append(outputs.detach().cpu().numpy())
                all_test_targets_list.append(targets.detach().cpu().numpy())
                sample_count += signals_batch.size(0) # Increment sample_count by batch size

        print(f"Finished processing {sample_count} samples for Subject {subject_id}.")
        
        if subject_counter == config['num_personalization_subjects']:
            break

    # Concatenate all collected arrays once
    if all_test_outputs_list:
        all_test_outputs = np.concatenate(all_test_outputs_list, axis=0)
        all_test_targets = np.concatenate(all_test_targets_list, axis=0)
    else:
        # Handle case where no samples were processed
        if config['sig2sig']:
            all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
            all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        else:
            all_test_outputs = np.empty((0, 2), dtype=float)
            all_test_targets = np.empty((0, 2), dtype=float)

    # Log test metrics and plot them
    print('Results w/o personalization')
    _ = call_metric(all_test_targets, all_test_outputs, config, figure_savepath=os.path.join(config['figure_path'], 'test_pretraining'), plot=True)

    # Deallocate the pretrained model to free memory
    del pretrained_model
    gc.collect()


def personalization(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device, save_models=False):

    ## --- Personalization ---

    # Get test subjects
    personalization_subjects = dataset.base_dataset.subject_list

    # Use lists to collect outputs and targets, then concatenate once at the end
    all_test_outputs_list = []
    all_test_targets_list = []

    for subject_counter, subject_id in enumerate(personalization_subjects):

        print(f"{subject_counter}/{len(personalization_subjects)} personalizing subject {subject_id}")

        continual_ds.set_active_subject(subject_id) # Set the active subject for the dataset

        # Set the number of training samples
        # Ensure that personalization_sample_number is not greater than available samples
        # If -1, use all samples for personalization training
        num_subject_samples = len(continual_ds)
        if config['personalization_sample_number'] == -1:
            effective_personalization_samples = num_subject_samples
        else:
            effective_personalization_samples = min(config['personalization_sample_number'], num_subject_samples)

        if effective_personalization_samples == 0:
            print(f'No samples available for subject {subject_id}, skipping ...')
            continue

        # Load pretrained model
        pretrained_model = get_model_architecture(config)
        checkpoint = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
        print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
        pretrained_model.load_state_dict(checkpoint['model'])
        pretrained_model.eval() # to avoid inconsistent results as explicitly reported at https://pytorch.org/tutorials/beginner/saving_loading_models.html
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
        personalized_model.train()
        if config['model_name'] == 'GRU':
            set_trainable_parameters(model=personalized_model, tune=config['tune'], config=config)
        elif config['model_name'] == 'UNet':
            set_trainable_parameters(model=personalized_model, tune=config['tune'], config=config)
        elif config['model_name'] == 'EUNet':
            set_trainable_parameters(model=personalized_model, tune=config['tune'], config=config)
        else:
            raise ValueError(f"Invalid model name: {config['model_name']}")

        # Print trainable and non-trainable parameters
        trainable_params = sum(p.numel() for p in personalized_model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in personalized_model.parameters() if not p.requires_grad)

        print(f"Trainable parameters: {trainable_params} / {trainable_params + non_trainable_params} ({trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
        print(f"Non-trainable parameters: {non_trainable_params} / {trainable_params + non_trainable_params} ({non_trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")

        # Optimizer
        optim_sched = configure_optimizer_and_scheduler(personalized_model, config)
        optimizer = optim_sched['optimizer']

        # --- Simulate Online Training Loop for a Subject with DataLoader ---
        # Use num_workers from config
        subject_dataloader = DataLoader(continual_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['loader_worker'])

        training_sample_count = 0  # Counter for samples used in training
        total_samples_processed = 0 # Total samples processed for this subject
        train_losses = AverageMeter(name='train/loss')

        for batch_data in subject_dataloader:
            signals_batch = batch_data['sig_tensor']
            targets_batch = batch_data['annotation_tensor']
            abp_valid_batch = batch_data['abp_valid']

            # Filter out invalid ABP samples if needed
            valid_indices = abp_valid_batch.squeeze().bool()
            signals_batch = signals_batch[valid_indices]
            targets_batch = targets_batch[valid_indices]

            current_batch_size = signals_batch.size(0)
            if current_batch_size == 0: # Skip if batch is empty after filtering
                continue

            # Determine if we're still in training phase or inference-only phase
            if training_sample_count < effective_personalization_samples:
                # Training phase: perform training on this batch
                outputs, targets, loss_val = process_batch_training(
                    signals_batch, targets_batch, personalized_model, optimizer,
                    config, device, writer, total_samples_processed # Pass current total samples for logging
                )
                if outputs is not None:
                    train_losses.update(loss_val, current_batch_size)
                    training_sample_count += current_batch_size
                    #print(f"Processed training batch of {current_batch_size} samples (total trained: {training_sample_count})")

                    # Check if we've reached the personalization limit
                    if training_sample_count >= effective_personalization_samples:
                        print(f"Reached personalization limit of {effective_personalization_samples} samples. Switching to inference-only mode.")
                        # Set model to eval mode for inference-only processing
                        personalized_model.eval()
                        set_trainable_parameters(model=personalized_model, tune='none', config=config)
            else:
                # Inference-only phase: no training, just forward pass
                outputs, targets = process_batch_inference(
                    signals_batch, targets_batch, personalized_model, device
                )
                #if outputs is not None:
                #    print(f"Processed inference batch of {current_batch_size} samples")

            # Add to results lists
            if outputs is not None:
                all_test_outputs_list.append(outputs.detach().cpu().numpy())
                all_test_targets_list.append(targets.detach().cpu().numpy())

            total_samples_processed += current_batch_size

        print(f"Finished processing {total_samples_processed} samples for Subject {subject_id}.")
        print(f"Training was performed on {min(training_sample_count, effective_personalization_samples)} samples.")

        writer.close()

        # Remove the writer folder
        shutil.rmtree(os.path.join(tensorboard_path, str(subject_id)))
        
        if subject_counter == config['num_personalization_subjects']:
            break

    # Concatenate all collected arrays once
    if all_test_outputs_list:
        all_test_outputs = np.concatenate(all_test_outputs_list, axis=0)
        all_test_targets = np.concatenate(all_test_targets_list, axis=0)
    else:
        # Handle case where no samples were processed
        if config['sig2sig']:
            all_test_outputs = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
            all_test_targets = np.empty((0, config['input_seq_len_s'] * config['fs']), dtype=float)
        else:
            all_test_outputs = np.empty((0, 2), dtype=float)
            all_test_targets = np.empty((0, 2), dtype=float)

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
    print(f'Figure folder: {figure_path}')

    # Load configuration into a dict
    config = dict()
    config.update(args.__dict__)  # add argparse
    config['model_name'] = model_name # add model name
    config['checkpoint_path'] = checkpoint_path
    config['tensorboard_path'] = tensorboard_path
    config['figure_path'] = figure_path
    
    # Add num_passes to config with a default value if not already present
    if 'num_passes' not in config:
        config['num_passes'] = 1 # Default to 1 pass if not specified

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
    # This remains unchanged, as it's the core LMDB interface.
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
    # ContinualLearningDataset now WRAPS the OnlinePhysioDataset instance.
    continual_ds = ContinualLearningDataset(online_physio_dataset_base)


    ## Personalization
    pretrained_model_inference(args.expname, model_name, continual_ds, checkpoint_path, tensorboard_path, config, device)
    personalization(args.expname, model_name, continual_ds, checkpoint_path, tensorboard_path, config, device)
    print(f"Personalization completed")