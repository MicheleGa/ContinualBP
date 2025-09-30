import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import copy
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from training_utils.helpers import get_encoder_architecture, get_prediction_head_architecture, build_inner_optimizer, count_parameters
from training_utils.metrics import call_metric, compute_transfer_metrics_blockwise
from data.online_dataset import OnlineSubjectDataset
from data.preprocessing_utils.data_visualization import plot_subject_annotation_runs
from models.MAMLLearner import MAMLLearner
from models.DriftDetector import EmbeddingDriftDetector
import matplotlib.pyplot as plt
    

def eval_model_on_runset(encoder, prediction_head, dataset, run_samples, device, config):
    """
    run_samples: list of sampleIDs for the run's *test* set
    returns: error scalar, std of absolute error, and optionally concatenated outputs/targets
    """
    
    all_outputs = []
    all_targets = []
    encoder.eval(); prediction_head.eval()
    with torch.no_grad():
        
        # Iterate in small batches if run large
        B = config['personalization_batch_size']
        for i in range(0, len(run_samples), B):
            batch_ids = run_samples[i:i+B]
            batch = [dataset.__getitem__(sid) for sid in batch_ids]
            signals = torch.stack([s for s, _, _ in batch]).to(device)
            targets = torch.stack([t for _, t, _ in batch]).to(device)
            outputs = prediction_head(encoder(signals))
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
    
    if len(all_outputs) == 0:
        return None, None, None, None
    
    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # MAE error
    abs_errs = np.abs(all_outputs - all_targets)
    # keep per-sample mean (in case multi-dim output)
    per_sample_mae = np.mean(abs_errs, axis=1)
    error = float(np.mean(per_sample_mae))
    error_std = float(np.std(per_sample_mae))
    
    return error, error_std, all_outputs, all_targets


def personalization_on_subject_runs_sequence(pretrained_learner, dataset, subject_id, tensorboard_path, device, config):
    """
    Run the sequential continual learning protocol for one subject
    under three baselines:
      - 'no_adapt' : pretrained model evaluated throughout (no updates)
      - 'first_batch_finetune' : fine-tune only on the first training batch of the first run then freeze
      - 'online_adapt' : adapt on every training batch as in original script (optionally drift-triggered)

    Returns per-baseline blockwise errors and outputs/targets and CL metrics per baseline.
    """
    
    runs = dataset.get_subject_runs(
        subject_id, 
        window_length=config['input_seq_len_s'],
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
        raise ValueError("Runs number should be grater than 2 ...")

    # Create three learners
    learner_noadapt = copy.deepcopy(pretrained_learner)
    learner_first = copy.deepcopy(pretrained_learner)
    learner_online = copy.deepcopy(pretrained_learner)

    enc_no = learner_noadapt.encoder.to(device); ph_no = learner_noadapt.prediction_head.to(device)
    enc_first = learner_first.encoder.to(device); ph_first = learner_first.prediction_head.to(device)
    enc_online = learner_online.encoder.to(device); ph_online = learner_online.prediction_head.to(device)

    # Drift detector used by online baseline optionally
    drift_detector = None
    if config.get("setup_type") == "drift":
        drift_detector = EmbeddingDriftDetector(enc_online, device)
        drift_detector._init_baseline(path=config['pretraining_feats_stats'])

    # Records per-baseline
    baselines = ['no_adapt', 'first_batch_finetune', 'online_adapt']
    baseline_block_mae = {b: [] for b in baselines}
    baseline_block_std = {b: [] for b in baselines}
    baseline_outputs = {b: [] for b in baselines}
    baseline_targets = {b: [] for b in baselines}

    # Writer to log to Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))

    # --- Evaluate pretrained before adaptation (test set) for all baselines
    for block in runs:
        for b in baselines:
            err_before, err_std_before, outs_before, tgts_before = eval_model_on_runset(
                *( (enc_no, ph_no, dataset, block["test"], device, config) if b=='no_adapt' else
                   (enc_first, ph_first, dataset, block["test"], device, config) if b=='first_batch_finetune' else
                   (enc_online, ph_online, dataset, block["test"], device, config) )
            )
            # store
            baseline_block_mae[b].append(err_before)
            baseline_block_std[b].append(err_std_before)
            if outs_before is not None:
                baseline_outputs[b].append(outs_before)
                baseline_targets[b].append(tgts_before)

    # --- Process blocks and perform adaptations according to baseline rules ---
    # For first_batch_finetune we will apply updates only once: on the first training batch
    first_finetuned = False

    for block_idx, block in enumerate(runs):
        run_idx = block['r_idx']
        b_idx = block['b_idx']
        block_id = f"{run_idx+1}.{b_idx+1}"

        print(f"[Personalization] Processing subject {subject_id}, run {run_idx+1}, block {b_idx+1}/{len(runs)}")

        # --- NO ADAPT : nothing to do ---

        # --- FIRST_BATCH_FINETUNE ---
        if not first_finetuned:
            # fine-tune only on the first training batch of the first block
            if len(block['train']) > 0:
                # take the first training batch (i.e., contiguous B samples provided by dataset)
                batch_ids = block['train'][:config['personalization_batch_size']]
                batch = [dataset.__getitem__(sid) for sid in batch_ids]
                signals = torch.stack([s for s, _, _ in batch]).to(device)
                targets = torch.stack([t for _, t, _ in batch]).to(device)

                enc_first.train(); ph_first.train()
                inner_opt = build_inner_optimizer(enc_first, ph_first, base_lr=config['personalization_lr'], config=config)
                loss = 0.0
                for step in range(config['personalization_steps']):
                    outputs = ph_first(enc_first(signals))
                    loss = F.smooth_l1_loss(outputs, targets) if config['criterion']=='SmoothL1Loss' else F.mse_loss(outputs, targets)
                    inner_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(enc_first.parameters()) + list(ph_first.parameters()), config['grad_clip'])
                    inner_opt.step()
                    writer.add_scalar(f"{subject_id}_first_finetune/train_loss", float(loss.item()), step)

                # Freeze parameters after fine-tuning
                for p in enc_first.parameters():
                    p.requires_grad = False
                for p in ph_first.parameters():
                    p.requires_grad = False

                enc_first.eval(); ph_first.eval()
                first_finetuned = True
                print(f"Applied first-batch fine-tuning for subject {subject_id} (loss={loss.item() if loss is not None else 'n/a'})")

        # --- ONLINE ADAPT ---
        # Decide whether to adapt this block
        do_adapt = True
        if config.get("setup_type") == "drift":
            do_adapt = False
            if drift_detector is None:
                raise ValueError("Drift detector not initialized but setup_type is 'drift'")
            batch = [dataset.__getitem__(sid) for sid in block["train"]]
            signals = torch.stack([s for s, _, _ in batch])
            triggered, score = drift_detector.should_trigger(signals)
            writer.add_scalar(f"{subject_id}_run_{block_id}/drift_score", score)
            if triggered:
                do_adapt = True

        if do_adapt:
            inner_opt = build_inner_optimizer(enc_online, ph_online, base_lr=config['personalization_lr'], config=config)
            enc_online.train(); ph_online.train()
            B = config.get('personalization_batch_size')
            for step in range(config['personalization_steps']):
                for i in range(0, len(block['train']), B):
                    batch_ids = block['train'][i:i+B]
                    batch = [dataset.__getitem__(sid) for sid in batch_ids]
                    signals = torch.stack([s for s, _, _ in batch]).to(device)
                    targets = torch.stack([t for _, t, _ in batch]).to(device)
                    outputs = ph_online(enc_online(signals))
                    loss = F.smooth_l1_loss(outputs, targets) if config['criterion']=='SmoothL1Loss' else F.mse_loss(outputs, targets)
                    inner_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(enc_online.parameters()) + list(ph_online.parameters()), config['grad_clip'])
                    inner_opt.step()
                    writer.add_scalar(f"{subject_id}_run_{block_id}/online_train_loss", loss.item(), step)
            enc_online.eval(); ph_online.eval()

        # --- Evaluate after possible adaptations for this block for each baseline ---
        for b in baselines:
            enc = enc_no if b=='no_adapt' else (enc_first if b=='first_batch_finetune' else enc_online)
            ph = ph_no if b=='no_adapt' else (ph_first if b=='first_batch_finetune' else ph_online)
            err_after, err_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block["test"], device, config)
            baseline_block_mae[b][block_idx] = err_after
            baseline_block_std[b][block_idx] = err_std_after
            # replace outputs/targets arrays (we stored before) with after-adapt versions
            baseline_outputs[b][block_idx] = outs_after
            baseline_targets[b][block_idx] = tgts_after

            writer.add_scalar(f"{subject_id}_run_{block_id}/{b}_val_err_after", err_after)
            writer.add_scalar(f"{subject_id}_run_{block_id}/{b}_val_err_after_std", err_std_after)

    writer.close()

    # Concatenate outputs/targets per baseline
    outs_and_tgts = {}
    for b in baselines:
        if len(baseline_outputs[b]) > 0:
            outs_and_tgts[b] = (np.concatenate([o for o in baseline_outputs[b] if o is not None], axis=0),
                                np.concatenate([t for t in baseline_targets[b] if t is not None], axis=0))
        else:
            outs_and_tgts[b] = (np.empty((0,)), np.empty((0,)))

    # Compute CL metrics (AA, BWT) per baseline
    baseline_metrics = {}
    for b in baselines:
        err_list = [v for v in baseline_block_mae[b] if v is not None]
        if len(err_list) == 0:
            baseline_metrics[b] = {"AA": None, "BWT": None}
        else:
            baseline_metrics[b] = compute_transfer_metrics_blockwise(err_list)

    # Also prepare per-block MAE/STD arrays for plotting
    per_block_stats = {b: {'mae': baseline_block_mae[b], 'std': baseline_block_std[b]} for b in baselines}

    return per_block_stats, outs_and_tgts


def personalization(tensorboard_path, config, device):
    
    ## --- Personalization ---
    
    # Initialize dataset
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
    
    # Initialize encoder
    encoder = get_encoder_architecture(config)
    print(f"[Personalization] Encoder weights initialized ✅")
    
    # Initialize prediction head
    prediction_head = get_prediction_head_architecture(config)
    print(f"[Personalization] Prediction head weights initialized ✅")

    learner = MAMLLearner(encoder, prediction_head)

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    learner.load_state_dict(ckpt['learner_state_dict'])
    learner = learner.to(device)
    learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✅")

    # Count trainable parameters
    learner_trainable, learner_non_trainable = count_parameters(learner)
    print(f"[Personalization] Parameter count:")
    print(f"\t- MAML learner: {learner_trainable:,} trainable, {learner_non_trainable:,} non-trainable")

    # Get test subjects
    personalization_subjects = online_physio_dataset.subjects_for_personalization

    # Before loop
    pretrained_learner_copy = copy.deepcopy(learner)
    
    global_outs_and_tgts = {b: [] for b in ['no_adapt','first_batch_finetune','online_adapt']}

    for subject_counter, subject_id in enumerate(personalization_subjects):
        print(f"[Personalization] {subject_counter}/{len(personalization_subjects)} personalizing model on subject {subject_id}")
        
        per_block_stats, outs_and_tgts = personalization_on_subject_runs_sequence(pretrained_learner_copy, online_physio_dataset, subject_id, tensorboard_path, device, config)
        
        if outs_and_tgts is None:
            raise ValueError("Outputs and targets for a subject cannot be None ...")

        subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        blocks = len(per_block_stats['no_adapt']['mae'])
        x = np.arange(1, blocks + 1)
        plt.figure(figsize=(12,8))
        for b in per_block_stats.keys():
            mae_arr = per_block_stats[b]['mae']
            std_arr = per_block_stats[b]['std']
            mae_plot = np.array([np.nan if v is None else v for v in mae_arr])
            std_plot = np.array([np.nan if v is None else v for v in std_arr])
            plt.errorbar(x, mae_plot, yerr=std_plot, label=b, marker='o')
        plt.xlabel('Block index (chronological)')
        plt.ylabel('MAE (with STD errorbars)')
        plt.title(f'Subject {subject_id} - Blockwise test MAE per baseline')
        plt.legend()
        savepath = os.path.join(subj_dir, f'subject_{subject_id}_blockwise_mae.png')
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()

        # ---- Call metric plots for each baseline ----
        for b, (outs, tgts) in outs_and_tgts.items():
            if outs.shape[0] > 0:
                print(f"[Personalization] {subject_counter}/{len(personalization_subjects)} Results for {subject_id} with baseline {b}")
                call_metric(tgts, outs, config, os.path.join(subj_dir, f"subject_{subject_id}_{b}_metrics.png"), plot=True)
                global_outs_and_tgts[b].append((outs, tgts))
                
        print(f"[Personalization] Personalization on {subject_counter}/{len(personalization_subjects)} subject completed ✅")
            
        if config['num_personalization_subjects'] > 0 and subject_counter > config['num_personalization_subjects']:
            break 
    
    # ---- Aggregate call_metric across subjects ----
    agg_dir = os.path.join(config['figure_path'], 'aggregate_metrics')
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)
    for b, data_list in global_outs_and_tgts.items():
        if len(data_list) > 0:
            print(f'[Personalization] Aggregated personalization results for the baseline {b}')
            all_outs = np.concatenate([o for o, _ in data_list], axis=0)
            all_tgts = np.concatenate([t for _, t in data_list], axis=0)
            call_metric(all_tgts, all_outs, config, os.path.join(agg_dir, f"aggregate_{b}_metrics.png"), plot=True)

    print(f"[Personalization] Personalization completed ✅")
    
