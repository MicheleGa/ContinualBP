import math
import matplotlib.pyplot as plt

# === Schedulers (copied & simplified from your trainer) ===

def get_meta_lr(epoch, config):
    meta_lr_schedule = config['meta_lr_schedule']
    meta_epochs = config['max_training_epochs']
    base_meta_lr = config['meta_lr']

    if meta_lr_schedule == 'constant':
        return base_meta_lr
    elif meta_lr_schedule == 'cosine':
        return base_meta_lr * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
    elif meta_lr_schedule == 'cosine_wr':
        # Extended version with decaying peaks and optional tail
        T0      = int(config.get('lr_scheduler_T0'))
        T_mult  = float(config.get('lr_scheduler_T_mult'))
        eta_min0 = float(config.get('lr_scheduler_eta_min'))
        gamma   = float(config.get('lr_scheduler_gamma'))
        min_gamma = float(config.get('lr_scheduler_min_gamma'))
        max_cycles = config.get('lr_scheduler_max_cycles')
        tail_mode  = config.get('lr_scheduler_tail')
        eta_floor  = float(config.get('lr_scheduler_eta_floor'))

        cycle = 0
        length = T0
        e = epoch
        while e >= length:
            e -= length
            cycle += 1
            length = int(length * T_mult)

        if (max_cycles is not None) and (cycle >= int(max_cycles)):
            # compute how many epochs since last cycle finished
            rem = epoch
            length = T0
            for _ in range(int(max_cycles)):
                rem -= length
                length = int(length * T_mult)
            tail_epoch = max(0, rem)
            total_epochs = int(config['max_training_epochs'])
            last_cycle_peak = base_meta_lr * (gamma ** (int(max_cycles)-1))

            if tail_mode == 'linear':
                progress = min(1.0, tail_epoch / max(1, total_epochs))
                return eta_floor + (last_cycle_peak - eta_floor) * (1.0 - progress)
            else:
                phase = min(1.0, tail_epoch / max(1, total_epochs))
                return eta_floor + 0.5 * (last_cycle_peak - eta_floor) * (1 + math.cos(math.pi * phase))

        eta_max_cycle = base_meta_lr * (gamma ** cycle)
        eta_min_cycle = eta_min0 * (min_gamma ** cycle)
        return eta_min_cycle + 0.5 * (eta_max_cycle - eta_min_cycle) * (1 + math.cos(math.pi * e / max(1, length)))
    else:
        return base_meta_lr


def get_inner_lr(epoch, config):
    schedule = config['inner_lr_schedule']
    meta_epochs = config['max_training_epochs']
    base_lr_inner = config['lr_inner']
    inner_lr_min = config['inner_lr_min']
    if schedule == 'constant':
        return base_lr_inner
    elif schedule == 'cosine':
        return inner_lr_min + (base_lr_inner - inner_lr_min) * 0.5 * (1 + math.cos(math.pi * epoch / meta_epochs))
    else:
        return base_lr_inner


def get_inner_steps(epoch, config):
    meta_epochs = config['max_training_epochs']
    schedule = config['inner_steps_schedule']
    base_steps = config['inner_steps']
    max_steps = config['inner_steps_max']
    if schedule == 'constant':
        return base_steps
    elif schedule == 'cosine':
        progress = epoch / meta_epochs
        return int(round(base_steps + 0.5 * (1 - math.cos(math.pi * progress)) * (max_steps - base_steps)))
    elif schedule == 'increasing':
        progress = epoch / meta_epochs
        return int(base_steps + progress * (max_steps - base_steps))
    else:
        return base_steps


# === Example config ===
config = {
    'max_training_epochs': 800,

    # Meta LR schedule
    'meta_lr': 0.001,
    'meta_lr_schedule': 'cosine_wr',
    'lr_scheduler_T0': 100,
    'lr_scheduler_T_mult': 1.5,
    'lr_scheduler_eta_min': 1e-5,
    'lr_scheduler_gamma': 0.7,
    'lr_scheduler_min_gamma': 1.0,
    'lr_scheduler_max_cycles': 3,
    'lr_scheduler_tail': 'cosine',
    'lr_scheduler_eta_floor': 1e-5,

    # Inner LR schedule
    'lr_inner': 0.01,
    'inner_lr_min': 0.001,
    'inner_lr_schedule': 'cosine',

    # Inner steps schedule
    'inner_steps': 4,
    'inner_steps_max': 8,
    'inner_steps_schedule': 'cosine',
}

# === Collect curves ===
epochs = range(config['max_training_epochs'])
meta_lrs = [get_meta_lr(e, config) for e in epochs]
inner_lrs = [get_inner_lr(e, config) for e in epochs]
inner_steps = [get_inner_steps(e, config) for e in epochs]

# === Plot ===
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(epochs, meta_lrs)
axs[0].set_ylabel("Meta LR")
axs[0].set_title("Meta LR schedule")

axs[1].plot(epochs, inner_lrs)
axs[1].set_ylabel("Inner LR")
axs[1].set_title("Inner LR schedule")

axs[2].plot(epochs, inner_steps)
axs[2].set_ylabel("Inner Steps")
axs[2].set_xlabel("Epoch")
axs[2].set_title("Inner Steps schedule")

plt.tight_layout()
plt.show()
