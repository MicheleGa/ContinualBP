from typing import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============
# LoRA Wrappers
# =============

def attach_lora(self, r=8, alpha=16):
    """
    Walk the module tree and wrap eligible 1x1 Conv1d layers with LoRA.
    Depthwise convs are excluded.
    """
    pass

def _inject_lora_recursive(self, module, r, alpha):
    """
    Recursively wrap Conv1d(kernel=1) layers with LoRAInjectedConv1d.
    """
    pass

# =========================================================================
# LoRA MERGE / UNMERGE ----------------------------------------------------
# =========================================================================
def merge_lora(self):
    """
    Merge LoRA weights into their base conv layers for deployment.
    """
    pass

def unmerge_lora(self):
    """
    Undo merge so LoRA can keep training.
    """
    pass

def param_breakdown(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_total = 0
    lora_trainable = 0
    base_total = 0
    base_trainable = 0

    lora_names = []
    base_names = []

    for name, p in model.named_parameters():
        if "lora_" in name.lower() or "lora" in name.lower():
            lora_total += p.numel()
            if p.requires_grad:
                lora_trainable += p.numel()
            lora_names.append((name, p.numel(), p.requires_grad))
        else:
            base_total += p.numel()
            if p.requires_grad:
                base_trainable += p.numel()
            base_names.append((name, p.numel(), p.requires_grad))

    print(f"TOTAL params ...........: {total}")
    print(f" --> base params .......: {base_total} (trainable {base_trainable})")
    print(f" --> LoRA params .......: {lora_total} (trainable {lora_trainable})")
    print(f"TRAINABLE total ........: {trainable}")
    print()
    print("Top LoRA parameter names (name, #params, requires_grad):")
    for t in lora_names:
        print("  ", t)
    print()
    print("Some base parameter names (name, #params, requires_grad):")
    for t in base_names[:30]:
        print("  ", t)

class LoRAInjectedLinear(nn.Module):
    r"""
    Additive LoRA wrapper for an existing nn.Linear layer.
    Keeps the base linear trainable weights and adds low-rank adapters.
    """
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r != 0 else 1.0
        self.dropout = nn.Dropout(p=dropout)
        self.merged = False

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Init
        if r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.merged or self.r == 0:
            return self.base(x)

        # base forward
        base_out = self.base(x)
        # lora update
        # lora_A: (r, in_features) treat like weight for linear with in_features
        # we want (batch, seq, out_features)
        lora_inter = F.linear(self.dropout(x), self.lora_A)  # -> (..., r)
        lora_out = F.linear(lora_inter, self.lora_B) * self.scaling
        return base_out + lora_out

    def merge(self):
        """Merge LoRA weights into the base linear for inference."""
        if not self.merged and self.r > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            # delta shape (out_features, in_features) matches base.weight
            self.base.weight.data += delta
            self.merged = True

    def unmerge(self):
        """Undo merge for further fine-tuning."""
        if self.merged and self.r > 0:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data -= delta
            self.merged = False
            

class LoRAInjectedConv1d(nn.Module):
    """
    Wrap a Conv1d(kernel_size=1) layer with LoRA low-rank adapters.
    Supports merge/unmerge for deployment.
    """

    def __init__(self, conv, r=4, alpha=16):
        super().__init__()

        assert isinstance(conv, nn.Conv1d)
        assert conv.kernel_size[0] == 1, "LoRAInjectedConv1d only supports pointwise Conv1d"

        self.conv = conv  # base conv (frozen)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_ch = conv.in_channels
        out_ch = conv.out_channels

        # Create LoRA A and B matrices as conv layers
        self.lora_A = nn.Conv1d(in_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(r, out_ch, kernel_size=1, bias=False)

        # Initialization: A ~ Kaiming, B ~ zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Track whether LoRA weights are merged into conv
        self.merged = False

        # Freeze the base conv weights
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        if self.merged:
            return self.conv(x)

        return self.conv(x) + self.scaling * self.lora_B(self.lora_A(x))

    # --------------------------- MERGE / UNMERGE ---------------------------

    def merge_lora(self):
        """
        Bake LoRA weights into conv.weight.
        """
        if self.merged:
            return

        # Compute delta W = B @ A (outer product style)
        delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        delta_w = delta_w.view_as(self.conv.weight)

        self.conv.weight.data += delta_w
        self.merged = True

    def unmerge_lora(self):
        """
        Restore original conv.weight by subtracting LoRA update.
        """
        if not self.merged:
            return

        delta_w = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        delta_w = delta_w.view_as(self.conv.weight)

        self.conv.weight.data -= delta_w
        self.merged = False

            

# ===============
# Drift Detector
# ===============
class EmbeddingDriftDetector:
    r"""
    Lightweight embedding-based drift detector for on-device personalization.
    
    - Computes running embedding mean/std from a pretrained or personalized model.
    - Optionally uses replay buffer embeddings for hybrid drift detection.
    - Triggers adaptation only when both baseline and replay distances exceed thresholds.
    """

    def __init__(self, model, device, threshold=0.5, replay_threshold=0.35):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.replay_threshold = replay_threshold
        
        # baseline stats
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_std_l2 = None

    def _embed_batch(self, signals):
        """Return model embeddings for a batch as numpy array."""
        self.model.eval()
        with torch.no_grad():
            z = self.model(signals.to(self.device))
            return z.detach().cpu().numpy()

    def _compute_centroid_and_std(self, feats):
        """Compute mean and std vector of replay embeddings."""
        if feats is None or len(feats) == 0:
            return None, None
        feats = np.array(feats)
        centroid = np.mean(feats, axis=0).astype(np.float64)
        std_vec = np.sqrt(np.maximum(np.var(feats, axis=0, ddof=1), 1e-12)).astype(np.float64)
        return centroid, std_vec

    def _init_baseline(self, path):
        """
        Initialize baseline embedding stats from a .npz file containing:
        - baseline_mean
        - baseline_std
        """
        stats = np.load(path)
        self.baseline_mean = stats["baseline_mean"].astype(np.float32)
        self.baseline_std = stats["baseline_std"].astype(np.float32)
        self.baseline_std_l2 = float(np.linalg.norm(self.baseline_std))
        
        if self.baseline_mean is None or self.baseline_std is None:
            raise ValueError(f"Failed to initialize baseline from {path}")
        
        print(f"[EmbeddingDriftDetector] Baseline initialized from {path}")

    def should_trigger(self, signals_batch):
        """
        Simple version: trigger if distance from baseline > threshold * baseline_std_l2.
        """
        z = self._embed_batch(signals_batch)
        mean_z = np.mean(z, axis=0)
        dist = np.linalg.norm(mean_z - self.baseline_mean)
        scaled = dist / (self.baseline_std_l2 + np.finfo(np.float32).eps)
        return scaled > self.threshold, float(scaled)

    def should_trigger_with_replay(self, signals_batch, replay_feats=None):
        """
        Hybrid replay-aware drift detection.
        Trigger when BOTH baseline and replay distances exceed thresholds.
        
        Parameters
        -------------------
        signals_batch : torch.Tensor
            Current personalization batch (B, ...)
        replay_feats : np.ndarray or None
            Features from the replay buffer (N, D)
        
        Returns
        -------------------
        (triggered: bool, scores: dict)
            scores = {'baseline_score': float, 'replay_score': float or None}
        """
        z = self._embed_batch(signals_batch)
        mean_z = np.mean(z, axis=0).astype(np.float64)

        # Baseline comparison
        dist_baseline = np.linalg.norm(mean_z - self.baseline_mean)
        scaled_baseline = dist_baseline / (self.baseline_std_l2 + np.finfo(np.float32).eps)

        # Replay comparison
        replay_score = None
        if replay_feats is not None and len(replay_feats) > 0:
            replay_centroid, replay_std = self._compute_centroid_and_std(replay_feats)
            if replay_centroid is not None:
                dist_replay = np.linalg.norm(mean_z - replay_centroid)
                denom = float(np.linalg.norm(replay_std) + np.finfo(np.float32).eps)
                replay_score = dist_replay / denom

        # Decision rule (hybrid): both must exceed thresholds
        if replay_score is not None:
            triggered = (scaled_baseline > self.threshold) and (replay_score > self.replay_threshold)
        else:
            triggered = scaled_baseline > self.threshold

        return bool(triggered), {
            "baseline_score": float(scaled_baseline),
            "replay_score": None if replay_score is None else float(replay_score),
        }
    

# ==============
# Replay Buffer
# ==============
class ReservoirReplayBuffer:
    r"""
    Reservoir sampling based buffer for storing (feature, target) pairs.
    Uses reservoir sampling to maintain a uniform sample of seen examples under limited capacity.
    Stores tensors on CPU detached to avoid GPU memory explosion.
    """
    def __init__(self, max_size=64):
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
  
    
# =================
# Attention Pooling
# =================
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, C)
        w = self.att(x).squeeze(-1)     # (B, T)
        w = torch.softmax(w, dim=1)     # (B, T)
        z = torch.sum(x * w.unsqueeze(-1), dim=1)  # (B, C)
        return z