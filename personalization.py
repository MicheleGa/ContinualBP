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
from data.online_dataset import OnlineSubjectDataset
from data.preprocessing_utils.split import split_train_val_test_personalization
from training_utils.helpers import fixseed, generate_runname, get_model_architecture, set_trainable_parameters, parseargs, configure_optimizer_and_scheduler, EarlyStopping
from training_utils.metrics import call_metric, AverageMeter, get_metric_values
from data.preprocessing_utils.data_visualization import plot_signals
from models.trainer import MAMLLearner, BPRegressor, build_inner_optimizer


def pretty_print_metrics(metric_values, prefix=""):
    r"""
    Pretty print the metric_values dictionary in a readable format.

    Parameters
    ------------
    metric_values : dict
        Dictionary of metric names -> values.
    prefix : str, optional
        Optional string to prepend to each printed line (e.g., "Pre-Adaptation" or "Post-Adaptation").
    """
    print("\n" + "=" * 50)
    if prefix:
        print(f"{prefix} Metrics")
        print("-" * 50)

    for k, v in metric_values.items():
        print(f"{k:15s}: {v:.4f}")

    print("=" * 50 + "\n")
    

def personalization(save_name, model_name, dataset, checkpoint_path, tensorboard_path, config, device, save_models=False):

    ## --- Personalization ---
    
    # Load pretrained model
    model = get_model_architecture(config)
    
    feat_dim = model.embed_dim
    output_dim = 3  # SBP, DBP, MAP
    bp_regressor = BPRegressor(feat_dim, output_dim)
    
    learner = MAMLLearner(model, bp_regressor)
    
    ckpt = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
    learner.load_state_dict(ckpt['learner_state_dict'])
    learner = learner.to(device)
    learner.eval()
    
    # Model/Regressor number of parameters
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    regressor_trainable_params = sum(p.numel() for p in bp_regressor.parameters() if p.requires_grad)
    regressor_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Model parameters: trainable {model_trainable_params}/ non trainable {model_non_trainable_params}")
    print(f"Regressor parameters: trainable {regressor_trainable_params}/ non trainable {regressor_non_trainable_params}")
    print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
    
    # Get test subjects
    personalization_subjects = dataset.subjects_for_personalization

    # Record outputs and targets
    if config['sig2sig']:
        out_shape = config['input_seq_len_s'] * config['fs'] # Full waveform
    else:
        out_shape = 3 # SBP/DBP/MAP

    all_test_outputs_non_adapted = np.empty((0, out_shape), dtype=float)
    all_test_targets_non_adapted = np.empty((0, out_shape), dtype=float)
    all_test_outputs_adapted = np.empty((0, out_shape), dtype=float)
    all_test_targets_adapted = np.empty((0, out_shape), dtype=float)

    # Loop over subjects
    for subject_counter, subject_id in enumerate(personalization_subjects):

        print(f"{subject_counter}/{len(personalization_subjects)} personalizing subject {subject_id}")

        # Activate subject in dataset
        dataset.set_active_subject(subject_counter)

        # Logging to TensorBoard Summary Writer
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))

        # --- Get subject runs ---
        subject_runs = dataset.get_subject_runs(subject_id, training_samples=config["training_samples"])

        for run_idx, run in enumerate(subject_runs):
            
            # --- Build train/test sets from sample IDs ---
            train_samples = [dataset.__getitem__(dataset.active_subject_samples.index(sid)) for sid in run["train"]]
            test_samples = [dataset.__getitem__(dataset.active_subject_samples.index(sid)) for sid in run["test"]]

            train_signals = torch.stack([s["sig_tensor"] for s in train_samples]).to(device)
            train_targets = torch.stack([s["annotation_tensor"] for s in train_samples]).to(device)

            test_signals = torch.stack([s["sig_tensor"] for s in test_samples]).to(device)
            test_targets = torch.stack([s["annotation_tensor"] for s in test_samples]).to(device)

            # --- Deepcopy to avoid polluting outer model ---
            adapted_model = copy.deepcopy(learner.model).to(device)
            adapted_regressor = copy.deepcopy(learner.regressor).to(device)
            
            # -- Test before adaptation ---
            adapted_model.eval()
            adapted_regressor.eval()
            with torch.no_grad():
                outputs = adapted_regressor(adapted_model(test_signals))
                loss = F.smooth_l1_loss(outputs, test_targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, test_targets)
                
                metric_values = get_metric_values(loss, outputs, test_targets, config)

                all_test_outputs_non_adapted = np.concatenate(
                    (all_test_outputs_non_adapted, outputs.detach().cpu().numpy()), axis=0
                )
                all_test_targets_non_adapted = np.concatenate(
                    (all_test_targets_non_adapted, test_targets.detach().cpu().numpy()), axis=0
                )
                
                pretty_print_metrics(metric_values, prefix="Pre-Adaptation")
            
            # --- Adaptation ---
            inner_opt = build_inner_optimizer(
                adapted_model, adapted_regressor,
                base_lr=config['personalization_lr'],
                config=config
            )
            
            adapted_model.train()
            adapted_regressor.train()

            for step in range(config['personalization_steps']):
                outputs = adapted_regressor(adapted_model(train_signals))
                loss = F.smooth_l1_loss(outputs, train_targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, train_targets)

                inner_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(adapted_model.parameters()) + list(adapted_regressor.parameters()),
                    config['grad_clip']
                )
                inner_opt.step()
                
                writer.add_scalar(f"{subject_id}/train_loss", loss.item(), step)
            
            # --- Test after adaptation ---
            adapted_model.eval()
            adapted_regressor.eval()
            with torch.no_grad():
                outputs = adapted_regressor(adapted_model(test_signals))
                loss = F.smooth_l1_loss(outputs, test_targets) if config['criterion'] == 'SmoothL1Loss' else F.mse_loss(outputs, test_targets)
                
                metric_values = get_metric_values(loss, outputs, test_targets, config)

                all_test_outputs_adapted = np.concatenate(
                    (all_test_outputs_adapted, outputs.detach().cpu().numpy()), axis=0
                )
                all_test_targets_adapted = np.concatenate(
                    (all_test_targets_adapted, test_targets.detach().cpu().numpy()), axis=0
                )
                
                pretty_print_metrics(metric_values, prefix="Post-Adaptation")

            print(f"Finished processing samples for Subject {subject_id} - run {run_idx}/{len(subject_runs)}.")

        writer.close()
        #   shutil.rmtree(os.path.join(tensorboard_path, str(subject_id)))
        
        print(f"Finished processing samples for Subject {subject_id}. Perfomance have been evaluated over {all_test_targets_adapted.shape[1]} samples.")
        
        break # should be removed it is here just for debugging

    print('Results w/o personalization')
    _ = call_metric(all_test_targets_non_adapted, all_test_outputs_non_adapted, config,
                    figure_savepath=os.path.join(config['figure_path'], 'test_without_personalization'), plot=True)

    print('Results w/ personalization')
    _ = call_metric(all_test_targets_adapted, all_test_outputs_adapted, config,
                    figure_savepath=os.path.join(config['figure_path'], 'test_personalization'), plot=True)

    print(f"Personalization completed")
    
    
if __name__ == "__main__":
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

    print('Configuration for the run:')
    pprint.pprint(config, width=1)

    # Seed everything
    fixseed(config['seed'])
    pl.seed_everything(config['seed'])

    ## Build the dataset

    # Instantiate OnlineSubjectDataset
    online_physio_dataset = OnlineSubjectDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        sig2sig=config['sig2sig'],
        min_run_length=config['min_run_length']
    )

    ## Personalization
    personalization(args.expname, model_name, online_physio_dataset, checkpoint_path, tensorboard_path, config, device)