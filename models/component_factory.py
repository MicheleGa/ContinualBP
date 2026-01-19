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