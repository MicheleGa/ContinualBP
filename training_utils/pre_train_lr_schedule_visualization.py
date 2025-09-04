import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

# ====== Config from your training script ======
base_lr = 0.001
backbone_lr_mul = 1  # Assuming a multiplier for the backbone LR
weight_decay = 0.01

stage1_epochs = 10
stage2_epochs = 25
warmup_stage1 = 10
warmup_stage2 = 5

# ====== Helper functions from your training script ======
def linear_warmup(current_epoch, warmup_epochs, base_lr):
    """Calculates linear warmup LR."""
    if current_epoch >= warmup_epochs:
        return base_lr
    return base_lr * (0.1 + 0.9 * (current_epoch / warmup_epochs))

# ====== Dummy parameters for simulation ======
# In a real training loop, these would be the model's parameters
head_params = [torch.nn.Parameter(torch.randn(1))]
backbone_params = [torch.nn.Parameter(torch.randn(1))]

# ====== Stage 1 Simulation: Head-only ======
# Optimizer and scheduler setup for Stage 1, mimicking your code
optimizer_stage1 = torch.optim.AdamW(head_params, lr=base_lr, weight_decay=weight_decay)
scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_stage1, 
    T_max=stage1_epochs - warmup_stage1
)

head_stage1_lrs = []
# Loop through epochs to track LR
for epoch in range(stage1_epochs):
    # Manually set LR during warmup
    if epoch < warmup_stage1:
        for pg in optimizer_stage1.param_groups:
            pg['lr'] = linear_warmup(epoch, warmup_stage1, base_lr)
    
    # Store the current learning rate
    head_stage1_lrs.append(optimizer_stage1.param_groups[0]['lr'])

    # Step the scheduler only after warmup
    if epoch >= warmup_stage1 - 1:
        scheduler_stage1.step()

# ====== Stage 2 Simulation: Backbone + Head ======
lr_backbone = base_lr * backbone_lr_mul

# Optimizer with two parameter groups, mimicking your code
optimizer_stage2 = torch.optim.AdamW([
    {'params': backbone_params, 'lr': lr_backbone},
    {'params': head_params, 'lr': base_lr}
], weight_decay=weight_decay)

# Single scheduler for the multi-param-group optimizer
scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_stage2,
    T_max=stage2_epochs - warmup_stage2
)

head_stage2_lrs, back_stage2_lrs = [], []
# Loop through epochs to track LR
for epoch in range(stage2_epochs):
    # Manually set LR during warmup
    if epoch < warmup_stage2:
        for i, pg in enumerate(optimizer_stage2.param_groups):
            # Target LR for head (index 1) is base_lr, for backbone (index 0) is lr_backbone
            target_lr = base_lr if i == 1 else lr_backbone
            pg['lr'] = linear_warmup(epoch, warmup_stage2, target_lr)
    
    # Store the current learning rates
    back_stage2_lrs.append(optimizer_stage2.param_groups[0]['lr'])
    head_stage2_lrs.append(optimizer_stage2.param_groups[1]['lr'])

    # Step the scheduler only after warmup
    if epoch >= warmup_stage2 - 1:
        scheduler_stage2.step()

# ====== Plotting the schedules ======

epochs_total = list(range(1, stage1_epochs + stage2_epochs + 1))
head_all_lrs = head_stage1_lrs + head_stage2_lrs
back_all_lrs = [0.0] * stage1_epochs + back_stage2_lrs

plt.figure(figsize=(8, 5))
plt.plot(epochs_total, head_all_lrs, label='Head LR')
plt.plot(epochs_total, back_all_lrs, label='Backbone LR')
plt.axvline(stage1_epochs, color='gray', linestyle='--', alpha=0.6, label='Stage Transition')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Two-Stage Fine-Tuning LR Schedule (Simulated)')
plt.legend()
plt.grid(True)
plt.show()
