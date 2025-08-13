import numpy as np
import matplotlib.pyplot as plt

# Config
base_lr = 1e-3
mult = 1
backbone_lr = base_lr * mult
stage1_epochs, warmup1 = 20, 5
stage2_epochs, warmup2 = 100, 10

def warmup_lr(epoch, warmup_epochs, target_lr):
    if epoch >= warmup_epochs:
        return target_lr
    return target_lr * (0.1 + 0.9 * (epoch / warmup_epochs))

def cosine_lr(epoch, total_epochs, start_lr):
    # Cosine from start_lr to 0
    return start_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

# Stage 1 head LRs
head_stage1 = []
for e in range(stage1_epochs):
    if e < warmup1:
        lr = warmup_lr(e, warmup1, base_lr)
    else:
        lr = cosine_lr(e - warmup1, stage1_epochs - warmup1, base_lr)
    head_stage1.append(lr)

# Stage 2 head & backbone LRs
head_stage2, back_stage2 = [], []
for e in range(stage2_epochs):
    if e < warmup2:
        head_lr = warmup_lr(e, warmup2, base_lr)
        back_lr = warmup_lr(e, warmup2, backbone_lr)
    else:
        head_lr = cosine_lr(e - warmup2, stage2_epochs - warmup2, base_lr)
        back_lr = cosine_lr(e - warmup2, stage2_epochs - warmup2, backbone_lr)
    head_stage2.append(head_lr)
    back_stage2.append(back_lr)

# Combine for plotting
epochs_total = list(range(1, stage1_epochs + stage2_epochs + 1))
head_all = head_stage1 + head_stage2
back_all = [0.0]*stage1_epochs + back_stage2

plt.figure(figsize=(8,5))
plt.plot(epochs_total, head_all, label='Head LR')
plt.plot(epochs_total, back_all, label='Backbone LR')
plt.axvline(stage1_epochs, color='gray', linestyle='--', alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Two-Stage Fine-Tuning LR Schedule')
plt.legend()
plt.grid(True)
plt.show()
