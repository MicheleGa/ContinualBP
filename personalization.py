import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import shutil
import pprint
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from data.online_dataset import OnlineSubjectDataset
from training_utils.helpers import fixseed, generate_runname, get_model_architecture, parseargs
from training_utils.metrics import call_metric
from data.preprocessing_utils.data_visualization import plot_subject_annotation_runs
from models.trainer import MAMLLearner, BPRegressor, build_inner_optimizer


# ---------- simple embedding-based drift detector ----------
class EmbeddingDriftDetector:
    """
    Lightweight detector using running mean/std of model embeddings for a subject.
    - baseline_mean/std: established from initial run or a short calibration window.
    - trigger when L2 distance of current batch mean to baseline_mean > threshold * baseline_std_l2.
    """
    def __init__(self, model, device, buffer_init=None, threshold=5.0):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        # baseline mean & std vectors
        self.baseline_mean = None
        self.baseline_std = None
        if buffer_init is not None:
            self._init_from_buffer(buffer_init)

    def _embed_batch(self, signals):
        self.model.eval()
        with torch.no_grad():
            z = self.model(signals.to(self.device))
            # if model returns regressor outputs or dict, adapt accordingly
            if isinstance(z, tuple) or isinstance(z, list):
                z = z[0]
            return z.detach().cpu().numpy()

    def _init_from_buffer(self, buffer_samples):
        # buffer_samples: list of signals (np arrays or torch)
        zs = []
        batch = torch.stack([torch.tensor(s) for s in buffer_samples]).to(self.device)
        emb = self._embed_batch(batch)
        self.baseline_mean = np.mean(emb, axis=0)
        self.baseline_std = np.std(emb, axis=0)
        # prevent zero std
        self.baseline_std[self.baseline_std==0] = 1e-6
        # precompute l2 of baseline std for scaling
        self.baseline_std_l2 = np.linalg.norm(self.baseline_std)

    def should_trigger(self, signals_batch):
        z = self._embed_batch(signals_batch)
        mean_z = np.mean(z, axis=0)
        dist = np.linalg.norm(mean_z - self.baseline_mean)
        scaled = dist / (self.baseline_std_l2 + 1e-9)
        return scaled > self.threshold, float(scaled)
    

# ---------- error helpers ----------
def compute_error(outputs, targets, metric='mae'):
    """
    outputs, targets: numpy arrays (N, out_dim) or torch tensors
    returns scalar error (lower better)
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if metric == 'mse':
        return float(np.mean((outputs - targets)**2))
    elif metric == 'rmse':
        return float(np.sqrt(np.mean((outputs - targets)**2)))
    elif metric == 'mae':
        return float(np.mean(np.abs(outputs - targets)))
    else:
        raise ValueError("Unknown metric")


# ---------- evaluation on a given run list ----------
def eval_model_on_runset(model, regressor, dataset, run_samples, device, config):
    """
    run_samples: list of sampleIDs for the run's *test* set
    returns: error scalar and optionally concatenated outputs/targets
    """
    model.eval()
    regressor.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        # iterate in small batches if run large
        B = config['personalization_batch_size']
        for i in range(0, len(run_samples), B):
            batch_ids = run_samples[i:i+B]
            batch = [dataset.__getitem__(sid) for sid in batch_ids]
            signals = torch.stack([s for s,_ in batch]).to(device)
            targets = torch.stack([t for _,t in batch]).to(device)
            outputs = regressor(model(signals))
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    error = compute_error(all_outputs, all_targets, metric='mae') # N.B.: computed MAE, but is it the right metric?
    return error, all_outputs, all_targets


def evaluate_subject_runs_sequence(pretrained_learner, dataset, subject_id, tensorboard_path, device, config):
    """
    Run the sequential continual learning protocol for one subject
    under either Setup 1 (fixed interleaved) or Setup 2 (drift-triggered).
    Returns blockwise errors, outputs, and CL metrics.
    """
    runs = dataset.get_subject_runs(
        subject_id, 
        adapt_size=config['personalization_batch_size'], 
        val_size=config['personalization_batch_size'], 
        min_block_length=config['min_run_length']
    )

    if config['plot_personalization']:
        subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        if os.path.exists(subj_dir):
            shutil.rmtree(subj_dir)
        os.makedirs(subj_dir)
        plot_subject_annotation_runs(
            dataset, subject_id, runs,
            show_bp_plot=True,
            savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_runs.jpg")
        )

    if len(runs) < 1:
        return None, None, runs, None

    learner = copy.deepcopy(pretrained_learner)
    model = learner.model.to(device)
    regressor = learner.regressor.to(device)

    drift_detector = None
    if config.get("setup_type") == "drift":
        drift_detector = EmbeddingDriftDetector(model, device)

    subject_outputs_noadapt, subject_targets_noadapt = [], []
    subject_outputs_adapt, subject_targets_adapt = [], []
    err_after_list = []

    writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))
    
    # --- Evaluate pretrained before adaptation (test set)
    for block in runs:
        run_idx = block['r_idx']
        block_idx = block['b_idx']
        block_id = f"{run_idx+1}.{block_idx+1}"
        
        print(f"Testing pretrained model on subject {subject_id}, run {run_idx+1}, block with {block_idx+1}/{len(runs)} blocks")

        err_before, outs_before, tgts_before = eval_model_on_runset(
            model, regressor, dataset, block["test"], device, config
        )
        
        subject_outputs_noadapt.append(outs_before)
        subject_targets_noadapt.append(tgts_before)
        
        writer.add_scalar(f"{subject_id}_run_{block_id}/val_err_before", err_before)

    for block in runs:
        run_idx = block['r_idx']
        block_idx = block['b_idx']
        block_id = f"{run_idx+1}.{block_idx+1}"
        
        print(f"Processing subject {subject_id}, run {run_idx+1}, block with {block_idx+1}/{len(runs)} blocks")

        # ---- Decide whether to adapt ----
        do_adapt = True
        if config.get("setup_type") == "drift":
            
            do_adapt = False
            
            if drift_detector is None:
                raise ValueError("Drift detector not initialized but setup_type is 'drift'")
            
            batch = [dataset.__getitem__(sid) for sid in block["train"]]
            signals = torch.stack([s for s, _ in batch])
            
            triggered, score = drift_detector.should_trigger(signals)
            
            writer.add_scalar(f"{subject_id}_run_{block_id}/drift_score", score)
            
            if triggered:
                do_adapt = True

        # ---- Adaptation ----
        if do_adapt:
            
            inner_opt = build_inner_optimizer(model, regressor,
                base_lr=config['personalization_lr'], config=config)
            
            model.train(), regressor.train()
            B = config.get('personalization_batch_size')
            for step in range(config['personalization_steps']):
                for i in range(0, len(block['train']), B):
                    batch_ids = block['train'][i:i+B]
                    batch = [dataset.__getitem__(sid) for sid in batch_ids]
                    
                    signals = torch.stack([s for s, _ in batch]).to(device)
                    targets = torch.stack([t for _, t in batch]).to(device)
                    
                    outputs = regressor(model(signals))
                    
                    loss = F.smooth_l1_loss(outputs, targets) if config['criterion']=='SmoothL1Loss' else F.mse_loss(outputs, targets)
                    
                    inner_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(regressor.parameters()),
                        config['grad_clip']
                    )
                    inner_opt.step()
                    
                    writer.add_scalar(f"{subject_id}_run_{block_id}/train_loss", loss.item(), step)
            
            model.eval(), regressor.eval()

        # --- Evaluate after adaptation
        err_after, outs_after, tgts_after = eval_model_on_runset(
            model, regressor, dataset, block["test"], device, config
        )
        subject_outputs_adapt.append(outs_after)
        subject_targets_adapt.append(tgts_after)
        err_after_list.append(err_after)
        
        writer.add_scalar(f"{subject_id}_run_{block_id}/val_err_after", err_after)

    writer.close()

    outs_and_tgts = (
        np.concatenate(subject_outputs_noadapt, axis=0),
        np.concatenate(subject_targets_noadapt, axis=0),
        np.concatenate(subject_outputs_adapt, axis=0),
        np.concatenate(subject_targets_adapt, axis=0),
    )

    # ---- Compute CL metrics (AA, BWT) ----
    metrics = compute_transfer_metrics_blockwise(err_after_list)
    print(f"Subject {subject_id} metrics: {metrics}")

    return metrics, runs, outs_and_tgts


# ---------- transfer metrics ----------
def compute_transfer_metrics_blockwise(err_after_list):
    """
    Compute CL metrics (BWT, FWT, AA) for block-based continual learning.

    Parameters
    ----------
    err_after_list : list of float
        Validation errors after adaptation for each block.

    Returns
    -------
    metrics : dict
        {
          "BWT": float,
          "AA": float
        }
    """
    B = len(err_after_list)
    metrics = {}

    # Average accuracy (mean error after adaptation)
    metrics["AA"] = float(np.mean(err_after_list))

    # Backward Transfer (forgetting)
    if B > 1:
        final_errs = err_after_list[-1]
        bwt_vals = [err_after_list[i] - final_errs for i in range(B-1)]
        metrics["BWT"] = float(np.mean(bwt_vals))
    else:
        metrics["BWT"] = None

    return metrics


def personalization(dataset, tensorboard_path, config, device):
    
    ## --- Personalization ---

    # Load pretrained model
    model = get_model_architecture(config)
    feat_dim = model.embed_dim

    out_shape = config['input_seq_len_s'] * config['fs'] if config['sig2sig'] else 3  # Full waveform or SBP/DBP/MAP
    bp_regressor = BPRegressor(feat_dim, out_shape)

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

    # Before loop
    pretrained_learner_copy = copy.deepcopy(learner)

    # Record outputs/targets for all subjects
    all_test_outputs_non_adapted = np.empty((0, out_shape), dtype=float)
    all_test_targets_non_adapted = np.empty((0, out_shape), dtype=float)
    all_test_outputs_adapted = np.empty((0, out_shape), dtype=float)
    all_test_targets_adapted = np.empty((0, out_shape), dtype=float)

    # Store per-subject metrics
    subject_metrics_list = []

    for subject_counter, subject_id in enumerate(personalization_subjects):

        print(f"{subject_counter}/{len(personalization_subjects)} personalizing subject {subject_id}")

        dataset.set_active_subject(subject_id)

        metrics, runs, outs_and_tgts = evaluate_subject_runs_sequence(
            pretrained_learner_copy, dataset, subject_id, tensorboard_path, device, config
        )

        if metrics is None:
            print(f"Subject {subject_id} has insufficient runs/blocks. Skipping.")
            continue

        print("Subject metrics:", metrics)
        subject_metrics_list.append(metrics)

        subject_all_test_outputs_non_adapted = outs_and_tgts[0]
        subject_all_test_targets_non_adapted = outs_and_tgts[1]
        subject_all_test_outputs_adapted = outs_and_tgts[2]
        subject_all_test_targets_adapted = outs_and_tgts[3]

        if config['plot_personalization']:
            print(f'Subject {subject_id} results w/o personalization')
            _ = call_metric(subject_all_test_targets_non_adapted, subject_all_test_outputs_non_adapted, config,
                            figure_savepath=os.path.join(config['figure_path'], f"subject_{subject_id}",
                                                        f'{subject_id}_test_without_personalization'), plot=True)

            print(f'Subject {subject_id} results w/ personalization')
            _ = call_metric(subject_all_test_targets_adapted, subject_all_test_outputs_adapted, config,
                            figure_savepath=os.path.join(config['figure_path'], f"subject_{subject_id}",
                                                        f'{subject_id}_test_with_personalization'), plot=True)

        all_test_outputs_non_adapted = np.concatenate(
            (all_test_outputs_non_adapted, subject_all_test_outputs_non_adapted), axis=0
        )
        all_test_targets_non_adapted = np.concatenate(
            (all_test_targets_non_adapted, subject_all_test_targets_non_adapted), axis=0
        )
        all_test_outputs_adapted = np.concatenate(
            (all_test_outputs_adapted, subject_all_test_outputs_adapted), axis=0
        )
        all_test_targets_adapted = np.concatenate(
            (all_test_targets_adapted, subject_all_test_targets_adapted), axis=0
        )

        if subject_counter > config['num_personalization_subjects']:
            print(f"Reached the maximum number of subjects for personalization {config['num_personalization_subjects']}. Stopping.")
            break

        print(f"Finished processing Subject {subject_id} with {len(runs)} runs.")

    # --- Aggregate metrics across subjects
    if subject_metrics_list:
        agg_metrics = {k: np.nanmean([m[k] for m in subject_metrics_list if m[k] is not None])
                    for k in subject_metrics_list[0].keys()}
        print("Aggregate CL metrics across subjects:", agg_metrics)
        
        # Save agg_metrics to CSV
        csv_save_path = os.path.join(config['figure_path'], "aggregated_CL_metrics.csv")
        with open(csv_save_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'value'])
            for k, v in agg_metrics.items():
                writer.writerow([k, v])
        print(f"Aggregate metrics saved to {csv_save_path}")

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
    personalization(online_physio_dataset,tensorboard_path, config, device)