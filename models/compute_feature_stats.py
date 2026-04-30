import os
import sys
folders_to_add = ['data', 'models', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import pprint
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from models.maml import MAML
from models.component_factory import RidgeHead
from component_factory import Model
from data.dataset import PhysioDataset
from data.meta_dataloaders import MetaTaskDataset
from training_utils.helpers import get_encoder_architecture, get_prediction_head_architecture

# ---------------------------
# FEATURE STATISTICS FUNCTION
# ---------------------------

def compute_embedding_stats_full(
    encoder,
    supervised_loader,
    meta_loader,
    device,
    shrinkage=1e-4
):
    """
    Compute mean and full covariance over:
        - supervised pretraining data
        - meta-training data

    Returns:
        mean (128,)
        covariance (128,128)
        covariance_inverse (128,128)
        drift_scores (list)  # for threshold calibration
    """

    encoder.eval()

    n_samples = 0
    mean = None
    M2 = None

    def process_embeddings(z_batch):
        nonlocal n_samples, mean, M2

        # L2-normalization as per: https://github.com/mueller-mp/maha-norm/tree/main
        z_batch = torch.nn.functional.normalize(z_batch, p=2, dim=1)
        z_batch = z_batch.detach().cpu().numpy()
        
        if mean is None:
            D = z_batch.shape[1]
            mean = np.zeros(D, dtype=np.float64)
            M2 = np.zeros((D, D), dtype=np.float64)

        for x in z_batch:
            n_samples += 1
            delta = x - mean
            mean += delta / n_samples
            delta2 = x - mean
            M2 += np.outer(delta, delta2)

    # ------------------------
    # Supervised training data
    # ------------------------
    print("[Stats] Processing supervised pretraining data...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(supervised_loader):
            signals, _ = batch
            signals = signals.to(device).float()
            z = encoder(signals)
            process_embeddings(z)

    # ------------------
    # Meta-training data
    # ------------------
    print("[Stats] Processing meta-training data...")
    with torch.no_grad():
        for batch_idx, meta_batch in enumerate(meta_loader):
            (Xs, _), (_, _), _ = meta_batch
            meta_batch_size = Xs.shape[0]

            for t in range(meta_batch_size):
                sX = Xs[t].to(device).float()
                z = encoder(sX)
                process_embeddings(z)

    print(f"[Stats] Total embeddings processed: {n_samples}")

    covariance = M2 / (n_samples - 1)

    # Shrinkage regularization
    covariance += shrinkage * np.eye(covariance.shape[0])

    cov_inv = np.linalg.pinv(covariance)
    
    print("\n[Diagnostics] Covariance inverse")

    identity_test = covariance @ cov_inv
    identity_error = np.linalg.norm(identity_test - np.eye(identity_test.shape[0]))

    print("Inverse check ||C*C_inv - I||:", identity_error)
    
    print("\n[Diagnostics] Embedding statistics")

    print("Mean shape:", mean.shape)
    print("Covariance shape:", covariance.shape)

    print("Mean abs value:", np.mean(np.abs(mean)))
    print("Mean std estimate:", np.mean(np.sqrt(np.diag(covariance))))

    print("Covariance diagonal min:", np.min(np.diag(covariance)))
    print("Covariance diagonal max:", np.max(np.diag(covariance)))

    print("Covariance condition number:", np.linalg.cond(covariance))

    print("Any NaN in covariance:", np.isnan(covariance).any())
    print("Any Inf in covariance:", np.isinf(covariance).any())

    return (
        mean.astype(np.float32),
        covariance.astype(np.float32),
        cov_inv.astype(np.float32)
    )


# ---------------------
# THRESHOLD CALIBRATION
# ---------------------

def calibrate_thresholds(
    encoder,
    supervised_loader,
    meta_loader,
    mean,
    cov_inv,
    device
):
    """
    Compute Mahalanobis scores on pretraining data
    to derive drift thresholds.
    """

    encoder.eval()
    scores = []

    def compute_score(z_batch):
        
        # L2-normalization
        z_batch = torch.nn.functional.normalize(z_batch, p=2, dim=1)
        z_batch = z_batch.detach().cpu().numpy()

        # Difference from global mean
        deltas = z_batch - mean   # shape: [B, 128]

        # Mahalanobis distance per sample
        sample_mahalanobis = np.einsum(
            "bi,ij,bj->b",
            deltas,
            cov_inv,
            deltas
        )

        # Robust aggregation over the batch
        score = np.quantile(sample_mahalanobis, 0.95)

        return float(score)

    # Supervised
    with torch.no_grad():
        for batch_idx, batch in enumerate(supervised_loader):
            signals, _ = batch
            signals = signals.to(device).float()
            z = encoder(signals)
            score = compute_score(z)
            scores.append(score)

    # Meta-training
    with torch.no_grad():
        for batch_idx, meta_batch in enumerate(meta_loader):
            (Xs, _), (_, _), _ = meta_batch
            meta_batch_size = Xs.shape[0]

            for t in range(meta_batch_size):
                sX = Xs[t].to(device).float()
                z = encoder(sX)
                score = compute_score(z)
                scores.append(score)

    scores = np.array(scores)
    
    print("[Stats] Extreme score check")
    
    top10 = np.sort(scores)[-10:]
    print("[Stats]\tTop 10 scores:", top10)

    bottom10 = np.sort(scores)[:10]
    print("[Stats]\tBottom 10 scores:", bottom10)
    
    print("[Stats] Scores statistics:")
    print("\tMean:", scores.mean())
    print("\tStd:", scores.std())
    print("\tMin:", scores.min())
    print("\tMax:", scores.max())

    thr_01 = np.percentile(scores, 0.1)
    thr_02 = np.percentile(scores, 0.2)
    thr_03 = np.percentile(scores, 0.3)
    thr_04 = np.percentile(scores, 0.4)
    thr_05 = np.percentile(scores, 0.5)
    
    print(f"[Stats] Drift threshold (0.1th): {thr_01:.4f}")
    print(f"[Stats] Drift threshold (0.2th): {thr_02:.4f}")
    print(f"[Stats] Drift threshold (0.3th): {thr_03:.4f}")
    print(f"[Stats] Drift threshold (0.4th): {thr_04:.4f}")
    print(f"[Stats] Drift threshold (0.5th): {thr_05:.4f}")
    
    return thr_01, thr_02, thr_03, thr_04, thr_05


def pretrain_ridge_head(train_loader, val_loader, meta_train_loader, encoder, config, device):

    all_train_feats = []
    all_train_tgts = []
    all_val_feats = []
    all_val_tgts = []
    
    encoder.eval()

    def process_embeddings(z_batch, target_batch, train=False):
        nonlocal all_train_feats, all_train_tgts, all_val_feats, all_val_tgts

        # L2-normalization as per: https://github.com/mueller-mp/maha-norm/tree/main
        z_batch = torch.nn.functional.normalize(z_batch, p=2, dim=1)
        if train:
            all_train_feats.append(z_batch)
            all_train_tgts.append(target_batch)
        else:
            all_val_feats.append(z_batch)
            all_val_tgts.append(target_batch)

    # ------------------------
    # Supervised training data
    # ------------------------
    print("[Stats] Processing supervised pretraining data...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            signals, targets = batch
            signals, targets = signals.to(device).float(), targets.to(device).float()
            z = encoder(signals)
            process_embeddings(z, targets, train=True)

    # ------------------
    # Meta-training data
    # ------------------
    print("[Stats] Processing meta-training data...")
    with torch.no_grad():
        for batch_idx, meta_batch in enumerate(meta_train_loader):
            (Xs, Ys), (_, _), _ = meta_batch
            meta_batch_size = Xs.shape[0]

            for t in range(meta_batch_size):
                sX, sY = Xs[t].to(device).float(), Ys[t].to(device).float()
                z = encoder(sX)
                process_embeddings(z, sY, train=True)
                
    # ------------------
    # Validation data
    # ------------------
    print("[Stats] Processing validation data...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            signals, targets = batch
            signals, targets = signals.to(device).float(), targets.to(device).float()
            z = encoder(signals)
            process_embeddings(z, targets, train=False)

    print("[Ridge] Fitting ridge regression head...")

    X_all_train = torch.cat(all_train_feats, dim=0)
    Y_all_train = torch.cat(all_train_tgts, dim=0)
    
    X_all_val = torch.cat(all_val_feats, dim=0)
    Y_all_val = torch.cat(all_val_tgts, dim=0)

    print(f"[Ridge] X shape: {X_all_train.shape}")
    print(f"[Ridge] Y shape: {Y_all_train.shape}")
    print(f"[Ridge] X val shape: {X_all_val.shape}")
    print(f"[Ridge] Y val shape: {Y_all_val.shape}")

    # ------------------------------------------
    # Alpha validation: select best ridge alpha
    # ------------------------------------------
    ridge_alphas = config["ridge_alphas"]

    criterion = (
        F.smooth_l1_loss
        if config["criterion"] == "SmoothL1Loss"
        else F.mse_loss
    )

    best_val_loss = float("inf")
    best_state = None
    best_alpha = None

    for alpha in ridge_alphas:
        print(f"[Ridge] Trying alpha={alpha}...")

        ridge = RidgeHead(
            in_dim=X_all_train.shape[1],
            out_dim=Y_all_train.shape[1],
            alpha=alpha,
            fit_intercept=True,
            normalize=True
        ).to(device)

        # Train on training features
        ridge.fit(X_all_train, Y_all_train)

        # Validate: run inference on validation features
        with torch.no_grad():
            X_val_tensor = X_all_val.to(device) 
            Y_val_tensor = Y_all_val.to(device) 

            Y_val_pred = ridge(X_val_tensor)
            val_loss = criterion(Y_val_pred, Y_val_tensor)

        print(f"[Ridge] alpha={alpha} → val_loss={val_loss.item():.7f}")

        # Select model with the lowest validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = ridge.state_dict()
            best_alpha = alpha

    print(f"[Ridge] Best alpha={best_alpha} with val_loss={best_val_loss:.7f} ✓")

    return best_state
    


def compute_feature_stats(device, config):
    
    # ---------------------------
    # Load PhysioDataset
    # ---------------------------
    dataset = PhysioDataset(
        seed=config['seed'],
        lmdb_folder=os.path.join(config['dataset_folder'], config['dataset_name']),
        pretraining_split_ratio=list(map(float, config['pretraining_tr_val_tt_split_ratio'].split(','))),
        meta_split_ratio=config['meta_train_split_ratio'],
        drift_aware=config['drift_aware'],
        fs=config['fs'],
        input_seq_len_s=config['input_seq_len_s'],
        ecg=config['ecg'],
        min_subject_sample_number=config['min_subject_sample_number']
    )

    supervised_train_sampler, supervised_val_sampler, _, _ = dataset.get_pretraining_samplers()

    supervised_train_loader = DataLoader(
        dataset,
        sampler=supervised_train_sampler,
        batch_size=config['batch_size'],
        num_workers=config['loader_worker'],
        pin_memory=True
    )
    
    supervised_val_loader = DataLoader(
        dataset,
        sampler=supervised_val_sampler,
        batch_size=config['batch_size'],
        num_workers=config['loader_worker'],
        pin_memory=True
    )

    # ---------------------------
    # Meta-training loader
    # ---------------------------
    _ = dataset.get_pretraining_samplers()

    meta_train_ids = dataset.meta_learning_subjects

    meta_train_ds = MetaTaskDataset(
        base_dataset=dataset,
        patient_ids=meta_train_ids,
        k_support=config['k_support'],
        k_query=config['k_query'],
        window_length=config['input_seq_len_s']
    )

    meta_train_loader = DataLoader(
        meta_train_ds,
        batch_size=config['meta_batch_size'],
        shuffle=False,
        num_workers=config['loader_worker'],
        pin_memory=True
    )

    # ---------------------------
    # Load pretrained model
    # ---------------------------
    encoder_pre = get_encoder_architecture(config)
    prediction_head_pre = get_prediction_head_architecture(config)
    pretrained_model = Model(encoder_pre, prediction_head_pre)

    maml = MAML(
        pretrained_model,
        lr=config['inner_lr'],
        first_order=True,
        anil=(config['inner_adapt'] == 'head')
    )

    ckpt = torch.load(config['pretrained_model_ckpt_path'], weights_only=False)
    maml.load_state_dict(ckpt['learner_state_dict'])
    maml = maml.to(device)
    maml.eval()

    encoder = maml.module.encoder

    print("[Stats] Pretrained encoder loaded ✓")

    # ---------------------------
    # Compute statistics
    # ---------------------------
    mean, cov, cov_inv = compute_embedding_stats_full(
        encoder,
        supervised_train_loader,
        meta_train_loader,
        device
    )

    thr_01, thr_02, thr_03, thr_04, thr_05 = calibrate_thresholds(
        encoder,
        supervised_train_loader,
        meta_train_loader,
        mean,
        cov_inv,
        device
    )
    
    # ---------------------------
    # Pretrain Ridge Head
    # ---------------------------
    ridge_head_state_dict = pretrain_ridge_head(supervised_train_loader, supervised_val_loader, meta_train_loader, encoder, config, device)

    # ---------------------------
    # Save statistics & Ridge Head
    # ---------------------------
    save_dict = {
        "mean": mean,
        "covariance": cov,
        "cov_inv": cov_inv,
        "threshold_0.1": thr_01,
        "threshold_0.2": thr_02,
        "threshold_0.3": thr_03,
        "threshold_0.4": thr_04,
        "threshold_0.5": thr_05,
        "ridge_head_state_dict": ridge_head_state_dict
    }

    save_path = config['pretraining_feats_stats']
    torch.save(save_dict, save_path)

    print(f"[Stats] Feature statistics & Ridge Head state dictsaved at {save_path} ✓")