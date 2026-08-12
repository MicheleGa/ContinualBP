import numpy as np
import torch
import torch.nn as nn


# ==============
# Replay Buffer
# ==============
class ReservoirReplayBuffer:
    r"""
    Reservoir sampling based buffer for storing (feature, target) pairs.
    Uses reservoir sampling to maintain a uniform sample of seen examples under limited capacity.
    Stores tensors on CPU (.detach()) to avoid GPU memory explosion.
    """
    def __init__(self, seed, max_size=64):
        self.max_size = max_size
        self.features = []
        self.targets = []
        self.n_seen = 0  # total seen examples
        self.rng = np.random.default_rng(seed)

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
                j = self.rng.integers(0, self.n_seen)
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
        idxs = self.rng.choice(len(self.features), size=k, replace=False)
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
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(input_dim // 2, output_dim),
        )
    
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"BPRegressor expected input dims 2 [btach dimension, feature dimension], got {list(x.shape)}")
        return self.regressor(x)
        
        
class DeeperBPRegressor(nn.Module):
    r"""
    Source: https://github.com/easyfan327/FewShotBP/blob/main/models/PPGECGNet_V0e2x1b.py
    """
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(input_dim // 4, output_dim),
        )
    
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"BPRegressor expected input dims 2 [btach dimension, feature dimension], got {list(x.shape)}")
        return self.regressor(x)
        