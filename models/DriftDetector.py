import numpy as np
import torch


class EmbeddingDriftDetector:
    """
    Lightweight detector using running mean/std of model embeddings for a subject.
    - baseline_mean/std: established from initial run or a short calibration window.
    - trigger when L2 distance of current batch mean to baseline_mean > threshold * baseline_std_l2.
    """
    def __init__(self, model, device, threshold=0.5):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        # baseline mean & std vectors
        self.baseline_mean = None
        self.baseline_std = None
        
    def _embed_batch(self, signals):
        self.model.eval()
        with torch.no_grad():
            z = self.model(signals.to(self.device))
            # if model returns regressor outputs or dict, adapt accordingly
            if isinstance(z, tuple) or isinstance(z, list):
                z = z[0]
            return z.detach().cpu().numpy()

    def _init_baseline(self, path):
        """
        Initialize baseline statistics (mean and std) for embeddings
        from a saved .npz file generated during pretraining.

        Parameters
        ----------
        path : str
            Path to the .npz checkpoint file containing 'baseline_mean' and 'baseline_std'.
        """
        stats = np.load(path)
        
        # Load pre-saved mean and std
        self.baseline_mean = stats["baseline_mean"].astype(np.float32)
        self.baseline_std = stats["baseline_std"].astype(np.float32)
        
        # Precompute L2 norm of std for scaling
        self.baseline_std_l2 = float(np.linalg.norm(self.baseline_std))
        
        if self.baseline_mean is None or self.baseline_std is None:
            raise ValueError(f"Failed to initialize baseline from {path}")
        
        print(f"[EmbeddingDriftDetector] Baseline initialized from {path}")
        print(f"\tμ: {self.baseline_mean}, σ: {self.baseline_std}")

    def should_trigger(self, signals_batch):
        z = self._embed_batch(signals_batch)
        mean_z = np.mean(z, axis=0)
        
        dist = np.linalg.norm(mean_z - self.baseline_mean)
        scaled = dist / (self.baseline_std_l2 + np.finfo(np.float32).eps )
        
        return scaled > self.threshold, float(scaled)