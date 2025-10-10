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
from training_utils.metrics import call_metric, compute_transfer_metrics_from_matrix
from data.online_dataset import OnlineSubjectDataset
from data.preprocessing_utils.data_visualization import plot_subject_annotation_runs, plot_blockwise_mae
from models.MAMLLearner import MAMLLearner
from models.DriftDetector import EmbeddingDriftDetector
import matplotlib.pyplot as plt
import pandas as pd


class ReservoirReplayBuffer:
    """Reservoir sampling based buffer for storing (feature, target) pairs.
    Uses reservoir sampling to maintain a uniform sample of seen examples under limited capacity.
    Stores tensors on CPU detached to avoid GPU memory explosion.
    """
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self.features = []
        self.targets = []
        self.n_seen = 0  # total seen examples

    def add(self, feat_t, tgt_t):
        # feat_t, tgt_t are torch tensors (batch x D) (batch x out_dim)
        feat_cpu = feat_t.detach().cpu()
        tgt_cpu = tgt_t.detach().cpu()
        for i in range(feat_cpu.shape[0]):
            self.n_seen += 1
            if len(self.features) < self.max_size:
                self.features.append(feat_cpu[i])
                self.targets.append(tgt_cpu[i])
            else:
                # reservoir sampling: replace with probability max_size / n_seen
                j = np.random.randint(0, self.n_seen)
                if j < self.max_size:
                    self.features[j] = feat_cpu[i]
                    self.targets[j] = tgt_cpu[i]

    def sample(self, k):
        if len(self.features) == 0:
            return None, None
        k = min(k, len(self.features))
        idxs = np.random.choice(len(self.features), size=k, replace=False)
        feats = torch.stack([self.features[i] for i in idxs]).to(torch.float)
        tgts = torch.stack([self.targets[i] for i in idxs]).to(torch.float)
        return feats, tgts

    def __len__(self):
        return len(self.features)


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


def personalization_on_subject_runs_sequence(pretrained_learner, fresh_learner, dataset, subject_id, tensorboard_path, device, config):
    """
    Performs personalization for one subject with 5 baselines:
      - no_adapt
      - first_batch_finetune
      - online_adapt
      - online_from_scratch
      - continual_replay
    Computes error matrices and continual learning metrics (AA, BWT) per baseline,
    and saves per-baseline error matrices as CSVs.
    """
    # ---- Prepare runs ----
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['personalization_batch_size'],
        min_block_length=config['min_run_length']
    )
    if len(runs) < 1:
        raise ValueError("Subject must have at least one run with valid blocks.")

    subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
    if os.path.exists(subj_dir):
        shutil.rmtree(subj_dir)
    os.makedirs(subj_dir, exist_ok=True)

    # Optional visualization
    if config.get('plot_personalization', False):
        plot_subject_annotation_runs(
            dataset, subject_id, runs,
            show_bp_plot=False,
            savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_runs.jpg")
        )

    # ---- Initialize learners for all baselines ----
    baselines = ['no_adapt', 'first_batch_finetune', 'online_adapt', 'online_from_scratch', 'continual_replay']

    learner_noadapt = copy.deepcopy(pretrained_learner)
    learner_first = copy.deepcopy(pretrained_learner)
    learner_online = copy.deepcopy(pretrained_learner)
    learner_online_scratch = copy.deepcopy(fresh_learner)
    learner_cl = copy.deepcopy(pretrained_learner)

    enc_no, ph_no = learner_noadapt.encoder.to(device), learner_noadapt.prediction_head.to(device)
    enc_first, ph_first = learner_first.encoder.to(device), learner_first.prediction_head.to(device)
    enc_online, ph_online = learner_online.encoder.to(device), learner_online.prediction_head.to(device)
    enc_scratch, ph_scratch = learner_online_scratch.encoder.to(device), learner_online_scratch.prediction_head.to(device)
    enc_cl, ph_cl = learner_cl.encoder.to(device), learner_cl.prediction_head.to(device)

    # ---- Drift detector (optional) ----
    drift_detector = None
    if config.get("setup_type") == "drift":
        drift_detector = EmbeddingDriftDetector(enc_online, device)
        drift_detector._init_baseline(path=config['pretraining_feats_stats'])
        
    # ---- Prepare structures ----
    T = len(runs)
    errors_matrices = {b: np.full((T, T), np.nan, dtype=float) for b in baselines}
    baseline_block_mae = {b: [] for b in baselines}
    baseline_block_std = {b: [] for b in baselines}
    baseline_outputs = {b: [] for b in baselines}
    baseline_targets = {b: [] for b in baselines}

    writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))

    # ---- Setup continual replay ----
    replay_buffer = ReservoirReplayBuffer(max_size=config.get('replay_buffer_size'))
    replay_batch_size = config.get('replay_batch_size', 16)
    first_finetuned = False

    # ---- MAIN LOOP ----
    for block_idx, block in enumerate(runs):
        run_idx, b_idx = block['r_idx'], block['b_idx']
        block_id = f"{run_idx + 1}.{b_idx + 1}"
        print(f"[Personalization] Subject {subject_id}: Run {run_idx + 1}, Block {b_idx + 1}/{len(runs)}")

        # ---------- FIRST BATCH FINETUNE ----------
        if not first_finetuned and len(block['train']) > 0:
            batch_ids = block['train'][:config['personalization_batch_size']]
            batch = [dataset.__getitem__(sid) for sid in batch_ids]
            signals = torch.stack([s for s, _, _ in batch]).to(device)
            targets = torch.stack([t for _, t, _ in batch]).to(device)
            enc_first.train(); ph_first.train()
            opt = build_inner_optimizer(enc_first, ph_first, base_lr=config['personalization_lr'], config=config)
            for step in range(config['personalization_steps']):
                out = ph_first(enc_first(signals))
                loss = F.smooth_l1_loss(out, targets) if config['criterion']=='SmoothL1Loss' else F.mse_loss(out, targets)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(enc_first.parameters()) + list(ph_first.parameters()), config['grad_clip'])
                opt.step()
            for p in list(enc_first.parameters()) + list(ph_first.parameters()):
                p.requires_grad = False
            enc_first.eval(); ph_first.eval()
            first_finetuned = True
            print(f"[Personalization] First-batch fine-tune done (loss={loss.item():.4f})")

        # ---------- Decide adaptation ----------
        do_adapt = True
        if config.get("setup_type") == "drift":
            do_adapt = False
            if drift_detector is None:
                raise ValueError("Drift detector missing for drift setup.")
            batch = [dataset.__getitem__(sid) for sid in block["train"]]
            signals = torch.stack([s for s, _, _ in batch])
            triggered, score = drift_detector.should_trigger(signals)
            writer.add_scalar(f"{subject_id}_run_{block_id}/drift_score", score)
            if triggered:
                do_adapt = True
                print(f"[Personalization] Drift detected (score={score:.4f} > threshold={drift_detector.threshold}), adapting!")
            else:
                print(f"[Personalization] No drift detected (score={score:.4f} <= threshold={drift_detector.threshold}), skipping adaptation.")

        # ---------- Adapt each baseline ----------
        if do_adapt:
            # Online adaptation (pretrained)
            for (enc, ph, name) in [(enc_online, ph_online, "online_adapt"), (enc_scratch, ph_scratch, "online_from_scratch"), (enc_cl, ph_cl, "continual_replay")]:
                opt = build_inner_optimizer(enc, ph, base_lr=config['personalization_lr'], config=config)
                enc.train(); ph.train()
                for step in range(config['personalization_steps']):
                    for i in range(0, len(block['train']), config['personalization_batch_size']):
                        batch_ids = block['train'][i:i+config['personalization_batch_size']]
                        batch = [dataset.__getitem__(sid) for sid in batch_ids]
                        signals = torch.stack([s for s, _, _ in batch]).to(device)
                        targets = torch.stack([t for _, t, _ in batch]).to(device)

                        if name == "continual_replay":
                            cur_feats = enc(signals)
                            replay_feats, replay_tgts = replay_buffer.sample(replay_batch_size)
                            if replay_feats is not None:
                                replay_feats = replay_feats.to(device).detach()
                                replay_tgts = replay_tgts.to(device)
                                feats = torch.cat([cur_feats, replay_feats], dim=0)
                                tgts = torch.cat([targets, replay_tgts], dim=0)
                            else:
                                feats, tgts = cur_feats, targets
                            out = ph(feats)
                        else:
                            out = ph(enc(signals))
                            tgts = targets

                        loss = F.smooth_l1_loss(out, tgts) if config['criterion']=='SmoothL1Loss' else F.mse_loss(out, tgts)
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(ph.parameters()), config['grad_clip'])
                        opt.step()
                enc.eval(); ph.eval()

            # Update continual replay buffer
            with torch.no_grad():
                all_batch = [dataset.__getitem__(sid) for sid in block['train']]
                if len(all_batch) > 0:
                    signals_all = torch.stack([s for s, _, _ in all_batch]).to(device)
                    targets_all = torch.stack([t for _, t, _ in all_batch]).to(device)
                    feats_all = enc_cl(signals_all)
                    replay_buffer.add(feats_all.detach().cpu(), targets_all.detach().cpu())

        # ---------- Evaluate after adaptation ----------
        for b in baselines:
            enc = enc_no if b=='no_adapt' else (enc_first if b=='first_batch_finetune' else (enc_online if b=='online_adapt' else (enc_scratch if b=='online_from_scratch' else enc_cl)))
            ph = ph_no if b=='no_adapt' else (ph_first if b=='first_batch_finetune' else (ph_online if b=='online_adapt' else (ph_scratch if b=='online_from_scratch' else ph_cl)))
            enc.eval(); ph.eval()
            for i in range(block_idx+1):
                err_i, err_std_i, outs_i, tgts_i = eval_model_on_runset(enc, ph, dataset, runs[i]['test'], device, config)
                errors_matrices[b][block_idx, i] = err_i
            err_after, err_std_after, outs_after, tgts_after = eval_model_on_runset(enc, ph, dataset, block['test'], device, config)
            baseline_block_mae[b].append(err_after)
            baseline_block_std[b].append(err_std_after)
            baseline_outputs[b].append(outs_after)
            baseline_targets[b].append(tgts_after)
            writer.add_scalar(f"{subject_id}_run_{block_id}/{b}_val_mae", err_after)

    writer.close()

    # ---- Compute CL metrics (AA, BWT) ----
    baseline_metrics = {b: compute_transfer_metrics_from_matrix(errors_matrices[b]) for b in baselines}

    # ---- Save error matrices as CSV ----
    for b in baselines:
        df_err = pd.DataFrame(errors_matrices[b])
        df_err.to_csv(os.path.join(subj_dir, f"subject_{subject_id}_{b}_error_matrix.csv"), index=False)
    print(f"[Personalization] Saved error matrices for subject {subject_id} ✅")

    # ---- Prepare per-block stats ----
    per_block_stats = {b: {'mae': baseline_block_mae[b], 'std': baseline_block_std[b]} for b in baselines}

    # ---- Prepare concatenated outputs/targets ----
    outs_and_tgts = {}
    for b in baselines:
        if len(baseline_outputs[b]) > 0:
            outs_and_tgts[b] = (
                np.concatenate([o for o in baseline_outputs[b] if o is not None], axis=0),
                np.concatenate([t for t in baseline_targets[b] if t is not None], axis=0)
            )
        else:
            outs_and_tgts[b] = (np.empty((0,)), np.empty((0,)))

    return per_block_stats, outs_and_tgts, baseline_metrics


def personalization(tensorboard_path, config, device):
    
    ## --- Personalization ---
    
    # Initialize dataset
    online_physio_dataset = OnlineSubjectDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        sig2sig=config['sig2sig'],
        min_run_length=config['min_run_length']
    )
    
    # Initialize encoder and prediction head (fresh weights)
    encoder = get_encoder_architecture(config)
    prediction_head = get_prediction_head_architecture(config)
    fresh_learner = MAMLLearner(copy.deepcopy(encoder), copy.deepcopy(prediction_head))
    print(f"[Personalization] Fresh encoder & head initialized ✅")
    
    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    learner = MAMLLearner(encoder_pre, prediction_head_pre)

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    learner.load_state_dict(ckpt['learner_state_dict'])
    learner = learner.to(device)
    learner.eval()
    print(f"[Personalization] MAML Learner pre-trained ckpt loaded ✅")

    # Count trainable parameters
    learner_trainable, learner_non_trainable = count_parameters(learner)
    print(f"[Personalization] Parameter count:")
    print(f"	- MAML learner: {learner_trainable:,} trainable, {learner_non_trainable:,} non-trainable")

    # Get test subjects
    personalization_subjects = online_physio_dataset.subjects_for_personalization

    # Before loop
    pretrained_learner_copy = copy.deepcopy(learner)
    fresh_learner_copy = copy.deepcopy(fresh_learner)
    
    global_outs_and_tgts = {b: [] for b in ['no_adapt','first_batch_finetune','online_adapt','online_from_scratch','continual_replay']}

    for subject_counter, subject_id in enumerate(personalization_subjects):
        print(f"[Personalization] {subject_counter}/{len(personalization_subjects)} personalizing model on subject {subject_id}")
        
        per_block_stats, outs_and_tgts, baseline_metrics = personalization_on_subject_runs_sequence(pretrained_learner_copy, fresh_learner_copy, online_physio_dataset, subject_id, tensorboard_path, device, config)
        
        if outs_and_tgts is None:
            raise ValueError("Outputs and targets for a subject cannot be None ...")

        subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        # --- Save baseline_metrics as CSV (AA and BWT per baseline) ---
        df_metrics = pd.DataFrame.from_dict(baseline_metrics, orient='index')
        csv_path = os.path.join(subj_dir, f'subject_{subject_id}_baseline_metrics.csv')
        df_metrics.to_csv(csv_path, index=True)
        print(f"[Personalization] Saved baseline metrics CSV to {csv_path} ✅")
        
        # Plot and save MAE per block using the standalone function
        plot_blockwise_mae(per_block_stats, subject_id, savepath=os.path.join(subj_dir, f'subject_{subject_id}_blockwise_mae.png'))

        # ---- Call metric plots for each baseline ----
        for b, (outs, tgts) in outs_and_tgts.items():
            if outs.shape[0] > 0:
                print(f"[Personalization] {subject_counter}/{len(personalization_subjects)} Results for {subject_id} with baseline {b}")
                call_metric(tgts, outs, config, os.path.join(subj_dir, f"subject_{subject_id}_{b}_metrics.png"), plot=True)
                global_outs_and_tgts[b].append((outs, tgts))
                
        print(f"[Personalization] Personalization on {subject_counter}/{len(personalization_subjects)} subject completed ✅")
            
        if config['num_personalization_subjects'] > 0 and subject_counter > config['num_personalization_subjects']:
            break 
        
    agg_dir = os.path.join(config['figure_path'], 'aggregate_metrics')
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)
    
    # ---- Aggregate AA and BWT across subjects ----
    all_metrics = {b: {"AA": [], "BWT": []} for b in global_outs_and_tgts.keys()}
    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_baseline_metrics.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            for b in df.index:
                if "AA" in df.columns and not pd.isna(df.loc[b, "AA"]):
                    all_metrics[b]["AA"].append(df.loc[b, "AA"])
                if "BWT" in df.columns and not pd.isna(df.loc[b, "BWT"]):
                    all_metrics[b]["BWT"].append(df.loc[b, "BWT"])
    # Compute mean/std across subjects for each baseline
    agg_rows = []
    for b, vals in all_metrics.items():
        if len(vals["AA"]) > 0:
            aa_mean, aa_std = np.mean(vals["AA"]), np.std(vals["AA"])
            bwt_mean, bwt_std = np.mean(vals["BWT"]), np.std(vals["BWT"])
            agg_rows.append({"Baseline": b, "AA_mean": aa_mean, "AA_std": aa_std,
                             "BWT_mean": bwt_mean, "BWT_std": bwt_std})
            print(f'[Personalization] Aggregated personalization results (AA/BWT) for the baseline {b}')
            print(f' \t- AA: {aa_mean:.4f} ± {aa_std:.4f} /  - BWT: {bwt_mean:.4f} ± {bwt_std:.4f}')
            
    df_agg = pd.DataFrame(agg_rows)
    agg_csv_path = os.path.join(agg_dir, 'aggregate_baseline_metrics.csv')
    df_agg.to_csv(agg_csv_path, index=False)
    
    # ---- Aggregate call_metric across subjects ----
    for b, data_list in global_outs_and_tgts.items():
        if len(data_list) > 0:
            print(f'[Personalization] Aggregated personalization results (BHS/AAMI/Bland-Altman/R²) for the baseline {b}')
            all_outs = np.concatenate([o for o, _ in data_list], axis=0)
            all_tgts = np.concatenate([t for _, t in data_list], axis=0)
            call_metric(all_tgts, all_outs, config, os.path.join(agg_dir, f"aggregate_{b}_metrics.png"), plot=True)

    print(f"[Personalization] Saved aggregated metrics to {agg_dir} ✅")
    
    print(f"[Personalization] Personalization completed ✅")
