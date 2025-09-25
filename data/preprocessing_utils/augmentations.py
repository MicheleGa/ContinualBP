import os
import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =============================
# Augmentation definitions
# =============================

class Jitter(nn.Module):
    def __init__(self, sigma=0.02, p=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return x + self.sigma * torch.randn_like(x)
        return x

class Scaling(nn.Module):
    def __init__(self, sigma=0.05, p=0.2):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            factor = torch.normal(1.0, self.sigma, size=(1,), device=x.device)
            return x * factor
        return x

class TimeShift(nn.Module):
    def __init__(self, max_shift_samples=10, p=0.2):
        super().__init__()
        self.max_shift = max_shift_samples
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            shift = random.randint(-self.max_shift, self.max_shift)
            return torch.roll(x, shifts=shift, dims=0)
        return x

class TimeWarp(nn.Module):
    def __init__(self, sigma=0.05, knots=4, p=0.2):
        super().__init__()
        self.sigma = sigma
        self.knots = knots
        self.p = p

    def forward(self, x):
        if random.random() >= self.p:
            return x

        L = x.shape[0]
        # Warp index positions
        orig_steps = np.arange(0, L)
        warp = np.linspace(0, L-1, self.knots)
        warp_noise = np.random.normal(0, self.sigma * L, size=self.knots)
        warp_steps = np.clip(warp + warp_noise, 0, L-1)
        interp = np.interp(orig_steps, warp, warp_steps)
        warped = np.interp(interp, np.arange(L), x.cpu().numpy())
        return torch.tensor(warped, dtype=x.dtype, device=x.device)

class RandomCrop(nn.Module):
    def __init__(self, crop_size=0.8, p=0.3):
        super().__init__()
        self.crop_size = crop_size
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            L = x.shape[0]
            new_L = int(L * self.crop_size)
            start = random.randint(0, L - new_L)
            cropped = x[start:start+new_L]
            # Pad back to length L
            pad_left = start
            pad_right = L - (start + new_L)
            cropped = F.pad(cropped, (pad_left, pad_right))
            return cropped
        return x

class RandomMasking(nn.Module):
    def __init__(self, mask_size=0.1, p=0.3):
        super().__init__()
        self.mask_size = mask_size
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            L = x.shape[0]
            mask_len = int(L * self.mask_size)
            start = random.randint(0, L - mask_len)
            x = x.clone()
            x[start:start+mask_len] = 0.0
            return x
        return x
    
class PowerlineNoise(nn.Module):
    def __init__(self, amplitude_range: Tuple[float, float] = (0.01, 0.05),
                 freq: float = 50.0, sampling_rate: float = 125.0, p: float = 0.25):
        super().__init__()
        self.amplitude_range = amplitude_range
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        seq_len = x.size(-1)
        amplitude = random.uniform(*self.amplitude_range)
        t = torch.linspace(0, seq_len / self.sampling_rate, seq_len, device=x.device)
        powerline = amplitude * torch.sin(2 * np.pi * self.freq * t)
        return x + powerline

class BaselineWander(nn.Module):
    def __init__(self, amplitude_range: Tuple[float, float] = (0.05, 0.15),
                 freq_range: Tuple[float, float] = (1.0, 2.0), sampling_rate: float = 125.0, p: float = 0.3):
        super().__init__()
        self.amplitude_range = amplitude_range
        self.freq_range = freq_range
        self.sampling_rate = sampling_rate
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        seq_len = x.size(-1)
        amplitude = random.uniform(*self.amplitude_range)
        frequency = random.uniform(*self.freq_range)
        t = torch.linspace(0, seq_len / self.sampling_rate, seq_len, device=x.device)
        baseline = amplitude * torch.sin(2 * np.pi * frequency * t)
        return x + baseline


# Random Choice Transform
class RandomChoice(nn.Module):
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p if p is not None else [1.0/len(transforms)] * len(transforms)
    
    def __call__(self, x):
        transform = np.random.choice(self.transforms, p=self.p)
        return transform(x)


# --- Wrapper ---
class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
    
# --- Contrastive wrapper ---
class ContrastiveTransformations:
    def __init__(self, ppg_transforms, ecg_transforms=None,  n_views=2):
        self.ppg_transforms = ppg_transforms
        self.ecg_transforms = ecg_transforms
        self.n_views = n_views

    def __call__(self, x):
        # x can be [time, 2]  (PPG, ECG) if self.ecg else [time,1] (PPG)
        # if x has ppg and ecg, apply the corresponding set of transforms 
        
        if self.ecg_transforms is not None:
            ppg_data = x[:, 0].unsqueeze(1)
            ecg_data = x[:, 1].unsqueeze(1)
            
            views = []
            for _ in range(self.n_views):
                # Apply PPG transforms to PPG data
                ppg_view = self.ppg_transforms(ppg_data.squeeze())
                # Apply ECG transforms to ECG data
                ecg_view = self.ecg_transforms(ecg_data.squeeze())
                # Combine the augmented views and add to the list
                views.append(torch.cat((ppg_view.unsqueeze(1), ecg_view.unsqueeze(1)), dim=1))
            return views
        else:
            # If only PPG is present
            return [self.ppg_transforms(x.squeeze()) for _ in range(self.n_views)]
    

# =============================
# Visualization utilities
# =============================

def plot_compare(signal, aug_signal, title, savepath):

    _, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(signal.numpy())
    axes[0].set_title("Original")

    axes[1].plot(aug_signal.numpy())
    axes[1].set_title(f"{title}")

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'{title}.png'), dpi=200)
    plt.close()
    
# =============================
# Modality-specific pipelines
# =============================

def get_ecg_augmentations(
    jitter_sigma=0.005, jitter_p=0.2,
    scaling_sigma=0.02, scaling_p=0.1,
    timeshift_max=3, timeshift_p=0.15,
    timewarp_sigma=0.02, timewarp_knots=2, timewarp_p=0.1,
    crop_size=0.95, crop_p=0.1,
    mask_size=0.03, mask_p=0.1,
    powerline_p=0.3, baseline_p=0.2,
):
    return Compose([
        Jitter(sigma=jitter_sigma, p=jitter_p),
        Scaling(sigma=scaling_sigma, p=scaling_p),
        TimeShift(max_shift_samples=timeshift_max, p=timeshift_p),
        TimeWarp(sigma=timewarp_sigma, knots=timewarp_knots, p=timewarp_p),
        RandomChoice([
            RandomCrop(crop_size=crop_size, p=crop_p),
            RandomMasking(mask_size=mask_size, p=mask_p)
        ]),
        PowerlineNoise(p=powerline_p),
        BaselineWander(p=baseline_p),
    ])


def get_ppg_augmentations(
    jitter_sigma=0.01, jitter_p=0.3,
    scaling_sigma=0.05, scaling_p=0.2,
    timeshift_max=3, timeshift_p=0.2,
    timewarp_sigma=0.05, timewarp_knots=3, timewarp_p=0.2,
    crop_size=0.9, crop_p=0.2,
    mask_size=0.05, mask_p=0.2,
    powerline_p=0.3, baseline_p=0.2,
):
    return Compose([
        Jitter(sigma=jitter_sigma, p=jitter_p),
        Scaling(sigma=scaling_sigma, p=scaling_p),
        TimeShift(max_shift_samples=timeshift_max, p=timeshift_p),
        TimeWarp(sigma=timewarp_sigma, knots=timewarp_knots, p=timewarp_p),
        RandomChoice([
            RandomCrop(crop_size=crop_size, p=crop_p),
            RandomMasking(mask_size=mask_size, p=mask_p)
        ]),
        PowerlineNoise(p=powerline_p),
        BaselineWander(p=baseline_p),
    ])


# =============================
# Tests
# =============================

def test_single_pipeline(pipeline, signal, signal_name, savepath):
    aug_list = []
    titles = []

    # Get the list of individual transforms from the pipeline
    transforms = pipeline.transforms
    
    # Iterate through each transform and apply it individually
    for t in transforms:
        # Check if the transform is a RandomChoice
        if isinstance(t, RandomChoice):
            # Apply each transform within the RandomChoice
            for sub_t in t.transforms:
                try:
                    aug_list.append(sub_t(signal.clone()))
                    titles.append(f"RandomChoice - {sub_t.__class__.__name__}")
                except Exception as e:
                    print(f"Skipping {sub_t.__class__.__name__} due to error: {e}")
        else:
            # Apply the single transform
            try:
                aug_list.append(t(signal.clone()))
                titles.append(t.__class__.__name__)
            except Exception as e:
                print(f"Skipping {t.__class__.__name__} due to error: {e}")

    # Apply the full pipeline
    aug_list.append(pipeline(signal.clone()))
    titles.append("Full pipeline")
    
    plot_compare(signal, aug_list, titles, savepath)

def test_augmentations():
    # --- Read Signals ---
    
    # Pick the first 10 seconds at 125hz of the first segenet
    ppg = torch.tensor(np.load('/data/users/mgaspari/data/raw_mimic_iii/ecg/p000188_ecg.npy')[0, :1250])
    ecg = torch.tensor(np.load('/data/users/mgaspari/data/raw_mimic_iii/ppg/p000188_ppg.npy')[0, :1250])
    fs = 125
    
    # --- Init. augmentations ---
    jitter = Jitter(sigma=0.02, p=1.0)
    scaling = Scaling(sigma=0.05, p=1.0)
    time_shift = TimeShift(max_shift_samples=10, p=1.0)
    time_warp = TimeWarp(sigma=0.05, knots=4, p=1.0)
    random_crop = RandomCrop(crop_size=0.8, p=1.0)
    random_masking = RandomMasking(mask_size=0.1, p=1.0)
    power_line_noise = PowerlineNoise(amplitude_range=(0.01, 0.05), freq= 50.0, sampling_rate=fs, p=1.0)
    baseline_wander = BaselineWander(amplitude_range=(0.05, 0.15), freq_range=(1.0, 2.0), sampling_rate=125.0, p=1.0)
    random_choice = RandomChoice([RandomCrop(crop_size=0.8, p=0.2), RandomMasking(mask_size=0.1, p=0.2)])
    ppg_compose = get_ppg_augmentations()
    ecg_compose = get_ecg_augmentations()
    
    ppg_jitter = jitter(ppg.clone())
    ecg_jitter = jitter(ecg.clone())
    
    ppg_scaling = scaling(ppg.clone())
    ecg_scaling = scaling(ecg.clone())

    ppg_time_shift = time_shift(ppg.clone())
    ecg_time_shift = time_shift(ecg.clone())

    ppg_time_warp = time_warp(ppg.clone())
    ecg_time_warp = time_warp(ecg.clone())

    ppg_random_crop = random_crop(ppg.clone())
    ecg_random_crop = random_crop(ecg.clone())

    ppg_random_masking = random_masking(ppg.clone())
    ecg_random_masking = random_masking(ecg.clone())

    ppg_power_line_noise = power_line_noise(ppg.clone())
    ecg_power_line_noise = power_line_noise(ecg.clone())

    ppg_baseline_wander = baseline_wander(ppg.clone())
    ecg_baseline_wander = baseline_wander(ecg.clone())
    
    ppg_random_choice_aug = random_choice(ppg.clone())
    ecg_random_choice_aug = random_choice(ecg.clone())
    
    ppg_compose_aug = ppg_compose(ppg.clone())
    ecg_compose_aug = ecg_compose(ecg.clone())

    plot_compare(signal=ppg, aug_signal=ppg_jitter, title='PPG Jitter', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_jitter, title='ECG Jitter', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_scaling, title='PPG Scaling', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_scaling, title='ECG Scaling', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_time_shift, title='PPG Time Shift', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_time_shift, title='ECG Time Shift', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_time_warp, title='PPG Time Warp', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_time_warp, title='ECG Time Warp', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_random_crop, title='PPG Random Crop', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_random_crop, title='ECG Random Crop', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_random_masking, title='PPG Random Masking', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_random_masking, title='ECG Random Masking', savepath='./')
    
    plot_compare(signal=ppg, aug_signal=ppg_power_line_noise, title='PPG Powerline Noise', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_power_line_noise, title='ECG Powerline Noise', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_baseline_wander, title='PPG Baseline Wander', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_baseline_wander, title='ECG Baseline Wander', savepath='./')
    
    plot_compare(signal=ppg, aug_signal=ppg_random_choice_aug, title='PPG Random Choice Augmentation', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_random_choice_aug, title='ECG Random Choice Augmentation', savepath='./')

    plot_compare(signal=ppg, aug_signal=ppg_compose_aug, title='PPG Compose Augmentation', savepath='./')
    plot_compare(signal=ecg, aug_signal=ecg_compose_aug, title='ECG Compose Augmentation', savepath='./')
    

if __name__ == "__main__":
    # Useful links to setup augmentations: 
    # - https://openreview.net/pdf?id=f8PIYPs-nB&
    # - https://www.mdpi.com/2079-9292/13/8/1599
    # - https://arxiv.org/abs/2206.07656
    
    test_augmentations()