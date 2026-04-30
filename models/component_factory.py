from collections import deque
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============
# Drift Detector
# ===============
class SBPDriftDetector:
    r"""
    Drift detector based ONLY on SBP annotation statistics.
    Designed for online personalization (per subject).
    """

    def __init__(
        self,
        window_size=5,
        eps=1e-6
    ):
        self.window_size = window_size
        self.eps = eps

        self.sbp_history = deque(maxlen=window_size)
        self.records = []

    def update(self, sbp_values: np.ndarray, step_idx: int):
        r"""
        Updates drift statistics and rolling history using a new batch of SBP training data.
        
        Parameters
        ------------
        sbp_values (np.ndarray): 
            An array of Systolic Blood Pressure values from the current training block.
            
        step_idx (int): 
            The current personalization block or time-step index.
            
        Returns
        ------------
        output param 1:
            A tuple containing the absolute drift (float) and the relative drift (float).   
        """
        if sbp_values.size == 0:
            return None

        batch_mean = float(np.mean(sbp_values))
        batch_var = float(np.var(sbp_values))

        if len(self.sbp_history) < 1:
            # First batch → no drift
            abs_drift = 0.0
            rel_drift = 0.0
            rolling_mean = batch_mean
            rolling_var = batch_var
        else:
            rolling_mean = float(np.mean(self.sbp_history))
            rolling_var = float(np.var(self.sbp_history))

            abs_drift = abs(batch_mean - rolling_mean)
            rel_drift = abs_drift / (rolling_mean + self.eps)

        self.sbp_history.append(batch_mean)

        record = {
            "step": step_idx,
            "batch_mean_sbp": batch_mean,
            "batch_var_sbp": batch_var,
            "rolling_mean_sbp": rolling_mean,
            "rolling_var_sbp": rolling_var,
            "abs_drift_sbp": abs_drift,
            "rel_drift_sbp": rel_drift,
        }

        self.records.append(record)
        
        return abs_drift, rel_drift

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
    

class FeatureDriftDetector:

    def __init__(
        self,
        stats_path,
        threshold,
        ema_alpha=1e-3,
        shrinkage=1e-4,
        device="cpu"
    ):

        stats = torch.load(stats_path, map_location=device)

        self.mean = torch.tensor(stats["mean"], device=device)
        self.cov = torch.tensor(stats["covariance"], device=device)

        self.cov_inv = torch.tensor(stats["cov_inv"], device=device)

        threshold_idx = str(threshold) if threshold < 1.0 else str(int(threshold))
        self.thr = float(stats[f"threshold_{threshold_idx}"])

        self.alpha = ema_alpha
        self.shrinkage = shrinkage

        self.device = device


    def mahalanobis_batch(self, features):
        
        # L2 normalization (must match pretraining)
        features = F.normalize(features, p=2, dim=1)

        delta = features - self.mean

        dist = torch.einsum(
            "bi,ij,bj->b",
            delta,
            self.cov_inv,
            delta
        )

        return dist


    def score(self, features):

        d = self.mahalanobis_batch(features)

        return torch.quantile(d, 0.95).item() # More reobust to outliers


    def update_stats(self, features):

        # L2 normalization (must match pretraining)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        mu_b = torch.mean(features, dim=0)

        cov_b = torch.cov(features.T)

        # EMA update
        self.mean = (1 - self.alpha) * self.mean + self.alpha * mu_b

        self.cov = (1 - self.alpha) * self.cov + self.alpha * cov_b

        # shrinkage regularization
        D = self.cov.shape[0]
        self.cov += self.shrinkage * torch.eye(D, device=self.device)

        # recompute inverse
        self.cov_inv = torch.linalg.pinv(self.cov)


    def update(self, features):

        score = self.score(features)

        drift = score > self.thr

        return score, drift


# ==============
# Replay Buffer
# ==============
class ReservoirReplayBuffer:
    r"""
    Reservoir sampling based buffer for storing (feature, target) pairs.
    Uses reservoir sampling to maintain a uniform sample of seen examples under limited capacity.
    Stores tensors on CPU (.detach()) to avoid GPU memory explosion.
    """
    def __init__(self, max_size=64):
        self.max_size = max_size
        self.features = []
        self.targets = []
        self.n_seen = 0  # total seen examples

    def add(self, feat_t, tgt_t):
        r"""
        Adds new examples to the buffer using reservoir sampling to maintain a uniform distribution.
        
        Parameters
        ------------
        feat_t (torch.Tensor): 
            Batch of input features.
            
        tgt_t (torch.Tensor): 
            Batch of corresponding targets/labels.
        """
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
        r"""
        Randomly selects a subset of stored features and targets from the buffer.
        
        Parameters
        ------------
        k (int): 
            The number of samples to retrieve.
            
        Returns
        ------------
        output param 1:
            A tuple of (features, targets) as torch Tensors, or (None, None) if empty.   
        """
        if len(self.features) == 0:
            return None, None
        k = min(k, len(self.features))
        idxs = np.random.choice(len(self.features), size=k, replace=False)
        feats = torch.stack([self.features[i] for i in idxs]).to(torch.float)
        tgts = torch.stack([self.targets[i] for i in idxs]).to(torch.float)
        return feats, tgts

    def __len__(self):
        return len(self.features)
    
    
# ==============
# Model Wrapper
# ==============
class Model(torch.nn.Module):
    def __init__(self, encoder, prediction_head):
        super().__init__()
        self.encoder = encoder
        self.prediction_head = prediction_head

    def forward(self, x):
        feats = self.encoder(x)
        return self.prediction_head(feats)
    

# ==================
# BP Prediction Head
# ==================
class BPRegressor(nn.Module):
    r"""
    Source: https://github.com/easyfan327/FewShotBP/blob/main/models/PPGECGNet_V0e2x1b.py
    """
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"BPRegressor expected input dims 2 [btach dimension, feature dimension], got {list(x.shape)}")
        return self.regressor(x)
    

# ==============================
# Shallow ML Heads for Regressor
# ==============================
class RidgeHead(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0, fit_intercept=True, normalize=True):
        """
        Drop-in ridge regression head.

        Args:
            in_dim (int): feature dimension
            out_dim (int): output dimension (e.g. 2 for SBP/DBP)
            alpha (float): ridge regularization strength
            fit_intercept (bool): whether to use bias term
            normalize (bool): whether to normalize features
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        self.register_buffer("W", None)        # [D(+1), O]
        self.register_buffer("mean", None)
        self.register_buffer("std", None)

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        ones = torch.ones(X.shape[0], 1, device=X.device)
        return torch.cat([X, ones], dim=1)

    def _normalize(self, X):
        if not self.normalize:
            return X

        if self.mean is None:
            self.mean = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True) + 1e-6

        return (X - self.mean) / self.std

    def fit(self, X, Y, alpha=None):
        """
        Fit ridge regression in closed form.

        X: [N, D]
        Y: [N, O]
        """
        if alpha is None:
            alpha = self.alpha
        alpha = max(alpha, 1e-6)

        # Normalize
        X = self._normalize(X)

        # Add bias
        X = self._add_bias(X)

        D = X.shape[1]

        sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=X.device))

        X_aug = torch.cat([
            X,
            sqrt_alpha * torch.eye(D, device=X.device)
        ], dim=0)

        Y_aug = torch.cat([
            Y,
            torch.zeros(D, Y.shape[1], device=Y.device)
        ], dim=0)

        self.W = torch.linalg.lstsq(X_aug, Y_aug).solution

    def forward(self, X):
        """
        Predict outputs.

        X: [N, D]
        """
        if self.W is None:
            raise RuntimeError("RidgeHead must be fitted before calling forward().")

        X = self._normalize(X)
        X = self._add_bias(X)

        return X @ self.W

    def reset(self):
        """Reset model (useful per subject or per block)."""
        self.W = None
        self.mean = None
        self.std = None