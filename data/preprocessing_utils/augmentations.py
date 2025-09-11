import torch
import torch.nn as nn
import random

# --- Custom 1D augmentations ---
class Jitter(nn.Module):
    def __init__(self, sigma=0.03, p=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return x + torch.randn_like(x) * self.sigma
        return x


class Scaling(nn.Module):
    def __init__(self, sigma=0.1, p=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            factor = torch.normal(mean=1.0, std=self.sigma, size=(1,), device=x.device)
            return x * factor
        return x


class TimeWarp(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            # simple time warp: random shift
            shift = random.randint(-5, 5)  # adjust based on sampling rate
            return torch.roll(x, shifts=shift, dims=0)
        return x


# --- Compose style ---
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# --- Contrastive wrapper ---
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
