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
from models.maml import MAML
from component_factory import ReservoirReplayBuffer, EmbeddingDriftDetector, Model
from training_utils.helpers import get_encoder_architecture, get_prediction_head_architecture, build_inner_optimizer, count_parameters
from training_utils.metrics import call_metric, compute_transfer_metrics_from_matrix
from data.online_dataset import OnlineSubjectDataset
from data.preprocessing_utils.data_visualization import plot_subject_annotation_runs, plot_blockwise_mae, plot_update_summary_table
import pandas as pd


def eval_model_on_runset(encoder, prediction_head, dataset, run_samples, device, config):
    r"""
    run_samples: list of sampleIDs for the run's *test* set
    returns: error scalar, std of absolute error, and optionally concatenated outputs/targets
    """

    all_outputs = []
    all_targets = []
    encoder.eval(); prediction_head.eval()
    with torch.no_grad():

        # Iterate in small batches if run large
        B = config['validation_batch_size']
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


def personalization_on_subject_runs_sequence(pretrained_learner, fresh_model, dataset, subject_id, tensorboard_path, device, config):
    r"""
    Performs personalization for one subject with baselines:
      - no_adapt
      - first_batch_finetune
      - online_adapt
      - online_from_scratch
      - continual_replay (from pretrained)
      - continual_replay_from_scratch (new baseline: replay but starting from fresh)
    """

    # ---- Prepare runs ----
    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])
    runs = dataset.get_subject_runs(
        subject_id,
        window_length=config['input_seq_len_s'],
        adapt_size=config['personalization_batch_size'],
        val_size=config['validation_batch_size'],
        block_length=block_length
    )

    assert len(runs) >= 1, "Subject must have at least one run with valid blocks."

    subj_dir = os.path.join(config['figure_path'], f"subject_{subject_id}")
    if os.path.exists(subj_dir):
        shutil.rmtree(subj_dir)
    os.makedirs(subj_dir, exist_ok=True)

    # Optional visualization
    if config.get('plot_personalization'):
        plot_subject_annotation_runs(
            dataset,
            subject_id,
            runs,
            show_bp_plot=False,
            savepath=os.path.join(subj_dir, f"subject_{subject_id}_annotation_runs.jpg")
        )

    # ---- Initialize learners for all baselines ----
    baselines = [
        'no_adapt',
        'first_batch_finetune',
        'online_adapt',
        'online_from_scratch',
        'continual_replay',
        'continual_replay_from_scratch'   # NEW baseline
    ]

    learner_noadapt = copy.deepcopy(pretrained_learner)
    learner_first = copy.deepcopy(pretrained_learner)
    learner_online = copy.deepcopy(pretrained_learner)
    learner_online_scratch = copy.deepcopy(fresh_model)
    learner_cl = copy.deepcopy(pretrained_learner)
    learner_cl_scratch = copy.deepcopy(fresh_model)  # NEW: continual replay starting from scratch

    enc_no, ph_no = learner_noadapt.encoder.to(device), learner_noadapt.prediction_head.to(device)
    enc_first, ph_first = learner_first.encoder.to(device), learner_first.prediction_head.to(device)
    enc_online, ph_online = learner_online.encoder.to(device), learner_online.prediction_head.to(device)
    enc_scratch, ph_scratch = learner_online_scratch.encoder.to(device), learner_online_scratch.prediction_head.to(device)
    enc_cl, ph_cl = learner_cl.encoder.to(device), learner_cl.prediction_head.to(device)
    enc_cl_scratch, ph_cl_scratch = learner_cl_scratch.encoder.to(device), learner_cl_scratch.prediction_head.to(device)  # NEW

    # ---- Drift detectors per adaptive baseline (optional) ----
    detectors = {
        'online_adapt': None,
        'online_from_scratch': None,
        'continual_replay': None,
        'continual_replay_from_scratch': None  # NEW
    }
    if config.get("setup_type") == "drift":
        # create a detector for each adaptive baseline using its encoder
        detectors['online_adapt'] = EmbeddingDriftDetector(enc_online, device, threshold=config.get('drift_threshold'))
        detectors['online_from_scratch'] = EmbeddingDriftDetector(enc_scratch, device, threshold=config.get('drift_threshold'))
        detectors['continual_replay'] = EmbeddingDriftDetector(enc_cl, device, threshold=config.get('drift_threshold'))
        detectors['continual_replay_from_scratch'] = EmbeddingDriftDetector(enc_cl_scratch, device, threshold=config.get('drift_threshold'))

        # Initialize baseline stats from pretraining features stats file if provided
        path = config.get('pretraining_feats_stats', None)
        if path is not None:
            for det in detectors.values():
                if det is not None:
                    det._init_baseline(path)

    # ---- Prepare structures ----
    T = len(runs)
    errors_matrices = {b: np.full((T, T), np.nan, dtype=float) for b in baselines}
    baseline_block_mae = {b: [] for b in baselines}
    baseline_block_std = {b: [] for b in baselines}
    baseline_outputs = {b: [] for b in baselines}
    baseline_targets = {b: [] for b in baselines}

    # ---- Track updates per baseline ----
    updates_count = {b: 0 for b in baselines}

    writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))

    # ---- Setup continual replay buffers ----
    replay_buffer = ReservoirReplayBuffer(max_size=config.get('replay_buffer_size'))  # for pretrained continual_replay
    replay_buffer_scratch = ReservoirReplayBuffer(max_size=config.get('replay_buffer_size'))  # for new scratch continual_replay
    replay_batch_size = config.get('replay_batch_size')
    first_finetuned = False

    # ---- MAIN LOOP ----
    for block_idx, block in enumerate(runs):
        run_idx, b_idx = block['r_idx'], block['b_idx']
        block_id = f"{run_idx + 1}.{b_idx + 1}"
        print(f"[Personalization] Subject {subject_id}: Run {run_idx + 1}, Block {b_idx + 1}/{len(runs)}")
        
        # ---------- Evaluate before adaptation on training ----------
        # This evaluation is to ensure the system always produce an output given the stream of data
        # Create mapping baseline -> (enc, ph) to simplify selection
        baseline_encph = {
            'no_adapt': (enc_no, ph_no),
            'first_batch_finetune': (enc_first, ph_first),
            'online_adapt': (enc_online, ph_online),
            'online_from_scratch': (enc_scratch, ph_scratch),
            'continual_replay': (enc_cl, ph_cl),
            'continual_replay_from_scratch': (enc_cl_scratch, ph_cl_scratch)
        }

        for b in baselines:
            enc, ph = baseline_encph[b]

            enc.eval(); ph.eval()
            err_before, err_std_before, outs_before, tgts_before = eval_model_on_runset(enc, ph, dataset, block['train'], device, config)
            baseline_block_mae[b].append(err_before)
            baseline_block_std[b].append(err_std_before)
            baseline_outputs[b].append(outs_before)
            baseline_targets[b].append(tgts_before)
            writer.add_scalar(f"{subject_id}_run_{block_id}/{b}_val_mae", err_before)

        # ---------- FIRST BATCH FINETUNE ----------
        if not first_finetuned and len(block['train']) > 0:

            batch_ids = block['train'][:config['personalization_batch_size']]
            batch = [dataset.__getitem__(sid) for sid in batch_ids]

            signals = torch.stack([s for s, _, _ in batch]).to(device)
            targets = torch.stack([t for _, t, _ in batch]).to(device)

            # If LoRA was applied, there is no need to unmerge the LoRA weights here because this training loop 
            # will be performed only once after baselines intialization, when we are sure that LoRA weights are not merged 
            enc_first.train(); ph_first.train()
            opt = build_inner_optimizer(enc_first, ph_first, base_lr=config['personalization_lr'], config=config)
            for step in range(config['personalization_steps']):
                out = ph_first(enc_first(signals))
                loss = F.smooth_l1_loss(out, targets) if config['criterion']=='SmoothL1Loss' else F.mse_loss(out, targets)
                opt.zero_grad(); loss.backward(); opt.step()

            # Once the First-Batch-Fine-Tune baseline is adapted, it is frozen and online used in inference
            for p in list(enc_first.parameters()) + list(ph_first.parameters()):
                p.requires_grad = False
            enc_first.eval(); ph_first.eval()
            first_finetuned = True

            print(f"[Personalization] First-batch fine-tune done (loss={loss.item():.4f})")

        # ---------- Decide adaptation per baseline ----------
        # For baselines that can adapt: online_adapt, online_from_scratch, continual_replay*, continual_replay_from_scratch*
        do_adapt = {b: False for b in baselines}
        # If not drift setup: adapt by default (every block) for adaptive baselines
        if config.get("setup_type") != "drift":
            do_adapt['online_adapt'] = True
            do_adapt['online_from_scratch'] = True
            do_adapt['continual_replay'] = True
            do_adapt['continual_replay_from_scratch'] = True
        else:
            # Use detectors for each adaptive baseline
            # load the block's signals once (all train examples)
            batch = [dataset.__getitem__(sid) for sid in block["train"]]
            signals_block = torch.stack([s for s, _, _ in batch])

            # Evaluate detectors separately
            for name, det in [
                ('online_adapt', detectors['online_adapt']),
                ('online_from_scratch', detectors['online_from_scratch']),
                ('continual_replay', detectors['continual_replay']),
                ('continual_replay_from_scratch', detectors['continual_replay_from_scratch'])
            ]:
                if det is None:
                    raise ValueError("Drift Detector is None!")
                if signals_block.numel() == 0:
                    raise ValueError("Signals batch is empty!")

                if name == 'continual_replay':
                    replay_feats_local = torch.stack(replay_buffer.features) if len(replay_buffer.features) > 0 else None
                    triggered, _ = det.should_trigger_with_replay(signals_block, replay_feats=replay_feats_local)
                elif name == 'continual_replay_from_scratch':
                    replay_feats_local = torch.stack(replay_buffer_scratch.features) if len(replay_buffer_scratch.features) > 0 else None
                    triggered, _ = det.should_trigger_with_replay(signals_block, replay_feats=replay_feats_local)
                else:
                    triggered, _ = det.should_trigger(signals_block)
                if triggered:
                    do_adapt[name] = True
                else:
                    do_adapt[name] = False

        # ---------- Adapt each baseline independently ----------
        # We'll run adaptation separately per baseline when requested.
        adapt_tasks = [
            (enc_online, ph_online, "online_adapt"),
            (enc_scratch, ph_scratch, "online_from_scratch"),
            (enc_cl, ph_cl, "continual_replay"),
            (enc_cl_scratch, ph_cl_scratch, "continual_replay_from_scratch")  # NEW task
        ]

        for enc, ph, name in adapt_tasks:
            if not do_adapt.get(name):
                continue  # skip adaptation for this baseline on this block

            opt = build_inner_optimizer(enc, ph, base_lr=config['personalization_lr'], config=config)
            enc.train(); ph.train()
            # Training loop over personalization steps and batches
            for step in range(config['personalization_steps']):
                for i in range(0, len(block['train']), config['personalization_batch_size']):
                    batch_ids = block['train'][i:i+config['personalization_batch_size']]
                    batch = [dataset.__getitem__(sid) for sid in batch_ids]
                    signals = torch.stack([s for s, _, _ in batch]).to(device)
                    targets = torch.stack([t for _, t, _ in batch]).to(device)

                    if name in ["continual_replay", "continual_replay_from_scratch"]:
                        cur_feats = enc(signals)
                        # select appropriate buffer
                        if name == "continual_replay":
                            replay_feats, replay_tgts = replay_buffer.sample(replay_batch_size)
                        else:
                            replay_feats, replay_tgts = replay_buffer_scratch.sample(replay_batch_size)
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
                    opt.zero_grad(); loss.backward(); opt.step()

            if name == "continual_replay":
                # Update continual replay buffer (pretrained version)
                with torch.no_grad():
                    all_batch = [dataset.__getitem__(sid) for sid in block['train']]
                    if len(all_batch) > 0:
                        signals_all = torch.stack([s for s, _, _ in all_batch]).to(device)
                        targets_all = torch.stack([t for _, t, _ in all_batch]).to(device)
                        feats_all = enc_cl(signals_all)
                        replay_buffer.add(feats_all.detach().cpu(), targets_all.detach().cpu())
            elif name == "continual_replay_from_scratch":
                # Update continual replay buffer for scratch baseline
                with torch.no_grad():
                    all_batch = [dataset.__getitem__(sid) for sid in block['train']]
                    if len(all_batch) > 0:
                        signals_all = torch.stack([s for s, _, _ in all_batch]).to(device)
                        targets_all = torch.stack([t for _, t, _ in all_batch]).to(device)
                        feats_all = enc_cl_scratch(signals_all)
                        replay_buffer_scratch.add(feats_all.detach().cpu(), targets_all.detach().cpu())

            # After training, increment updates counter:
            updates_count[name] += 1
            writer.add_scalar(f"{subject_id}_run_{block_id}/{name}_updates_count", updates_count[name])

        # ---------- Evaluate after adaptation ----------
        # Create mapping baseline -> (enc, ph) to simplify selection
        baseline_encph = {
            'no_adapt': (enc_no, ph_no),
            'first_batch_finetune': (enc_first, ph_first),
            'online_adapt': (enc_online, ph_online),
            'online_from_scratch': (enc_scratch, ph_scratch),
            'continual_replay': (enc_cl, ph_cl),
            'continual_replay_from_scratch': (enc_cl_scratch, ph_cl_scratch)
        }

        for b in baselines:
            enc, ph = baseline_encph[b]

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

    # Insert updates_count into baseline_metrics for easy reporting
    for b in baselines:
        baseline_metrics.setdefault(b, {})
        baseline_metrics[b]['num_updates'] = updates_count.get(b, 0)

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

    block_length = config['num_train_val'] * (config['personalization_batch_size'] + config['validation_batch_size'])

    # Initialize dataset
    online_physio_dataset = OnlineSubjectDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        block_length=block_length,
        valid_runs_number=config['valid_runs_number']
    )

    # Initialize encoder and prediction head (fresh weights)
    encoder = get_encoder_architecture(config)
    prediction_head = get_prediction_head_architecture(config)
    fresh_model = Model(copy.deepcopy(encoder), copy.deepcopy(prediction_head))
    print(f"[Personalization] Fresh encoder & head initialized ✅")

    # Initialize pretrained learner (will be loaded from ckpt)
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    learner = Model(encoder_pre, prediction_head_pre)

    # N.B. MAML checkpoint saved with learn2learn wrapper (in maml.py script)
    # thereby MAML init must match that of the pretrainer.py script
    learner = MAML(
        learner,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head'),
        lora=config['use_lora']
    )

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
    fresh_model_copy = copy.deepcopy(fresh_model)

    # Include the new baseline key here as well so aggregation handles it
    global_outs_and_tgts = {
        b: [] for b in [
            'no_adapt',
            'first_batch_finetune',
            'online_adapt',
            'online_from_scratch',
            'continual_replay',
            'continual_replay_from_scratch'  # NEW
        ]
    }

    for subject_counter, subject_id in enumerate(personalization_subjects):
        print(f"[Personalization] {subject_counter + 1}/{len(personalization_subjects)} personalizing model on subject {subject_id}")

        per_block_stats, outs_and_tgts, baseline_metrics = personalization_on_subject_runs_sequence(pretrained_learner_copy, fresh_model_copy, online_physio_dataset, subject_id, tensorboard_path, device, config)

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
                print(f"[Personalization] {subject_counter + 1}/{len(personalization_subjects)} Results for {subject_id} with baseline {b}")
                call_metric(tgts, outs, config, os.path.join(subj_dir, f"subject_{subject_id}_{b}_metrics.png"), plot=True)
                # append to global for aggregated metrics later
                if b in global_outs_and_tgts:
                    global_outs_and_tgts[b].append((outs, tgts))

        # --- Extract and plot number of updates per baseline ---
        updates_per_baseline = {b: metrics.get('num_updates') for b, metrics in baseline_metrics.items()}
        update_fig_path = os.path.join(subj_dir, f"subject_{subject_id}_update_summary.png")
        plot_update_summary_table(updates_per_baseline, update_fig_path)
        print(f"[Personalization] Saved update summary figure for subject {subject_id} ✅")

        print(f"[Personalization] Personalization on {subject_counter + 1}/{len(personalization_subjects)} subject completed ✅")

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
            print(f'\t- AA: {aa_mean:.4f} ± {aa_std:.4f}')
            print(f'\t- BWT: {bwt_mean:.4f} ± {bwt_std:.4f}')

    df_agg = pd.DataFrame(agg_rows)
    agg_csv_path = os.path.join(agg_dir, 'aggregate_baseline_metrics.csv')
    df_agg.to_csv(agg_csv_path, index=False)

    # ---- Aggregate number of updates across subjects ----
    updates_agg = {}

    for subject_id in personalization_subjects:
        subj_csv = os.path.join(config['figure_path'], f"subject_{subject_id}", f"subject_{subject_id}_baseline_metrics.csv")
        if os.path.exists(subj_csv):
            df = pd.read_csv(subj_csv, index_col=0)
            if 'num_updates' in df.columns:
                for baseline in df.index:
                    updates_agg.setdefault(baseline, []).append(df.loc[baseline, 'num_updates'])

    # Compute mean and std of updates per baseline
    agg_update_rows = []
    for b, counts in updates_agg.items():
        if len(counts) > 0:
            mean_count, std_count = np.mean(counts), np.std(counts)
            agg_update_rows.append({'Baseline': b, 'mean_updates': mean_count, 'std_updates': std_count})
            print(f'[Personalization] Aggregated update counts for {b}: {mean_count:.2f} ± {std_count:.2f}')

    df_updates_agg = pd.DataFrame(agg_update_rows)
    agg_update_csv = os.path.join(agg_dir, 'aggregate_update_counts.csv')
    df_updates_agg.to_csv(agg_update_csv, index=False)

    # Plot aggregated update counts
    update_mean_dict = {row['Baseline']: row['mean_updates'] for _, row in df_updates_agg.iterrows()}
    update_fig_path = os.path.join(agg_dir, "aggregate_update_summary.png")
    plot_update_summary_table(update_mean_dict, update_fig_path)
    print(f"[Personalization] Saved aggregated update summary figure to {update_fig_path} ✅")

    # ---- Aggregate call_metric across subjects ----
    for b, data_list in global_outs_and_tgts.items():
        if len(data_list) > 0:
            print(f'[Personalization] Aggregated personalization results (BHS/AAMI/Bland-Altman/R²) for the baseline {b}')
            all_outs = np.concatenate([o for o, _ in data_list], axis=0)
            all_tgts = np.concatenate([t for _, t in data_list], axis=0)
            call_metric(all_tgts, all_outs, config, os.path.join(agg_dir, f"aggregate_{b}_metrics.png"), plot=True)

    print(f"[Personalization] Saved aggregated metrics to {agg_dir} ✅")

    print(f"[Personalization] Personalization completed ✅")
