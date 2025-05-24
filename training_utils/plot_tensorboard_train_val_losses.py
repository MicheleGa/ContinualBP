import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_losses_from_tensorboard(log_dir, train_loss_tag='train/loss', val_loss_tag='val/loss', title='', savepath='./figs'):
    """
    Extracts and plots train and validation losses from TensorBoard logs.

    Args:
        log_dir (str): Path to the directory containing the TensorBoard log files
                         (the directory containing events.out.tfevents.*).
        train_loss_tag (str, optional): Tag name used for training loss in TensorBoard.
                                         Defaults to 'train_loss_epoch'.
        val_loss_tag (str, optional): Tag name used for validation loss in TensorBoard.
                                       Defaults to 'val_loss_epoch'.
        savepath (str, optional): File path to save the generated plot.
                                   Defaults to './figs/losses.png'.
    """
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory '{log_dir}' not found.")
        return

    event_acc = EventAccumulator(log_dir)
    try:
        event_acc.Reload()
    except Exception as e:
        print(f"Error loading TensorBoard data from '{log_dir}': {e}")
        return

    train_losses = []
    val_losses = []
    train_epochs = []
    val_epochs = []
    for event in event_acc.scalars.Items(train_loss_tag):
        train_epochs.append(event.step)
        train_losses.append(event.value)

    for event in event_acc.scalars.Items(val_loss_tag):
        val_epochs.append(event.step)
        val_losses.append(event.value)

    # Create the x-axis array based on the length of the loss arrays
    train_epochs = np.arange(len(train_losses))
    val_epochs = np.arange(len(val_losses))

    # Ensure val_epochs and val_losses have the same length (they should)
    min_len = min(len(train_epochs), len(val_epochs))
    train_epochs = train_epochs[:min_len]
    train_losses = train_losses[:min_len]
    val_epochs = val_epochs[:min_len]
    val_losses = val_losses[:min_len]

    df = pd.DataFrame({'epoch': train_epochs, 'train_loss': train_losses, 'val_loss': val_losses})

    plt.figure(figsize=(12, 7))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss (per epoch)')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss (per epoch)')

    if val_losses:
        min_val_loss = min(val_losses)
        min_val_loss_epoch_index = val_losses.index(min_val_loss)
        min_val_loss_epoch = val_epochs[min_val_loss_epoch_index]
        plt.axvline(x=min_val_loss_epoch, color='r', linestyle='--', label=f'Min Val Loss ({min_val_loss:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (mmHg$^2$)')
    plt.ylim(0, 300)
    plt.title(f'{title} - Train and Validation Loss Over Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savepath, f'{title}_train_val_losses.jpg'))
    plt.close()

# Example usage: Replace with the actual path to your log directory
log_directory = './tensorboard/raw_dataset_5_win_len_3_overlap_resnet_cosine_warmup/raw_dataset_5_win_len_3_overlap_resnet_cosine_warmup-ResNet-2025_03_27-20_22_09/lightning_logs/version_0' # Make sure this points to the directory containing the events.out.tfevents file(s)

# You might need to adjust the tag names if you used different ones in your LightningModule
plot_losses_from_tensorboard(log_directory, train_loss_tag='train/loss_epoch', val_loss_tag='val/loss', title='PPG')