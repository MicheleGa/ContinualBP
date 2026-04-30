import os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from preprocessing_utils.signal_processing import compute_sp_dp
from preprocessing_utils.split import split_train_val_test_personalization


def plot_bp_pattern_distribution(dataloaders, dataloaders_names, savepath='./figs/dataset'):
    r"""
    Calculates Hyper/Hypo/normo-tensive windows distributions of the train/val/test split
    accessed through the corresponding dataloaders. The function plots them with mean,
    standard deviation, and quartiles. This function should be called only from dataset.py.
    
    Parameters
    ----------
    dataloaders : list of DataLoader
        List of dataloaders containing the data to analyze.
    dataloaders_names : list of str
        List of names corresponding to each dataloader, used for labeling the plots.
    savepath : str, default './figs/dataset'
        Path to save the generated plots.
        
    Returns
    -------
    None (saves the plots to the specified savepath)
    """
    labels_map = {
        0: 'Hypotension', 
        1: 'Normal', 
        2: 'Elevated', 
        3: 'Stage 1 Hypertension', 
        4: 'Stage 2 Hypertension'
    }

    # Ensure the save directory exists
    os.makedirs(savepath, exist_ok=True)

    for dataloader, dataloader_name in zip(dataloaders, dataloaders_names):
        all_data = []
        labels = []
        
        for batch in dataloader:
            # Unpack the batch. The label is the second element when bp_pattern is True
            _, batch_labels, _ = batch
            labels.extend(batch_labels.tolist())

        if not labels:
            print(f"No data collected for dataloader '{dataloader_name}'. Skipping plot.")
            continue
        
        counts = Counter(labels)
        for label, count in counts.items():
            all_data.append({
                'BP Pattern': labels_map.get(label, 'Unknown'),
                'Count': count
            })

        df = pd.DataFrame(all_data)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='BP Pattern', y='Count', data=df)

        # Calculate and add statistics for each group
        total_count = df['Count'].sum()
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                percent = (height / total_count) * 100
                plt.gca().text(p.get_x() + p.get_width() / 2., height,
                               f'{height}\n({percent:.1f}%)',
                               ha='center', va='bottom', fontsize=10, color='black', weight='bold')

        plt.title(f'Blood Pressure Pattern Distribution for {dataloader_name}')
        plt.xlabel('Blood Pressure Pattern')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{dataloader_name}_bp_distribution.png'), dpi=1000)
        plt.close()
        

def _calculate_dataset_mean_std(sbp_values, dbp_values, map_values, name, savepath):
    r"""
    Calculates key descriptive statistics and generates a visualization of the 
    distribution for SBP, DBP, and optionally MAP.

    The function creates a density-normalized histogram with vertical markers for 
    the mean ($\mu$), standard deviation ($\sigma$), and interquartile range (Q1, Q3). 
    This is used to verify that the dataset split (Train/Val/Test) is balanced and 
    covers the expected physiological range of blood pressure.

    Parameters
    ------------
    sbp_values (list or np.array): 
        Array of Systolic Blood Pressure labels in mmHg.
    dbp_values (list or np.array): 
        Array of Diastolic Blood Pressure labels in mmHg.
    map_values (list or np.array): 
        Array of Mean Arterial Pressure labels. Can be an empty list.
    name (str): 
        Descriptive name of the dataset (e.g., 'Pretraining-Train') for titles and filenames.
    savepath (str): 
        Directory path where the generated PNG distribution plot will be saved.
    """
    sbp_values = np.array(sbp_values)
    dbp_values = np.array(dbp_values)
    map_values = np.array(map_values)

    # Calculate statistics
    sbp_mean = np.mean(sbp_values)
    sbp_std = np.std(sbp_values)
    sbp_q1 = np.quantile(sbp_values, 0.25)
    sbp_q3 = np.quantile(sbp_values, 0.75)

    dbp_mean = np.mean(dbp_values)
    dbp_std = np.std(dbp_values)
    dbp_q1 = np.quantile(dbp_values, 0.25)
    dbp_q3 = np.quantile(dbp_values, 0.75)
    
    if len(map_values) > 0:
        map_mean = np.mean(map_values)
        map_std = np.std(map_values)
        map_q1 = np.quantile(map_values, 0.25)
        map_q3 = np.quantile(map_values, 0.75)
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    sns.set_palette("pastel")

    # Plot SBP histogram
    plt.hist(sbp_values, bins=50, alpha=0.7, label='SBP', color='skyblue')

    # Plot DBP histogram
    plt.hist(dbp_values, bins=50, alpha=0.7, label='DBP', color='lightcoral')
    
    if len(map_values) > 0:
        # Plot MAP histogram
        plt.hist(map_values, bins=50, alpha=0.7, label='MAP', color='lightgreen')

    # Add vertical lines for SBP
    plt.axvline(sbp_mean, color='blue', linestyle='dashed', linewidth=1, 
                label=r'SBP $\mu$: {:.2f}, $\sigma$: {:.2f}'.format(sbp_mean, sbp_std))  
    plt.axvline(sbp_q1, color='blue', linestyle='dotted', linewidth=1, label=f'SBP Q1: {sbp_q1:.2f}')
    plt.axvline(sbp_q3, color='blue', linestyle='dotted', linewidth=1, label=f'SBP Q3: {sbp_q3:.2f}')

    # Add vertical lines for DBP
    plt.axvline(dbp_mean, color='red', linestyle='dashed', linewidth=1,
                label=r'DBP $\mu$: {:.2f}, $\sigma$: {:.2f}'.format(dbp_mean, dbp_std))
    plt.axvline(dbp_q1, color='red', linestyle='dotted', linewidth=1, label=f'DBP Q1: {dbp_q1:.2f}')
    plt.axvline(dbp_q3, color='red', linestyle='dotted', linewidth=1, label=f'DBP Q3: {dbp_q3:.2f}')
    
    if len(map_values) > 0:
        # Add vertical lines for MAP
        plt.axvline(map_mean, color='green', linestyle='dashed', linewidth=1,
                    label=r'MAP $\mu$: {:.2f}, $\sigma$: {:.2f}'.format(map_mean, map_std))
        plt.axvline(map_q1, color='green', linestyle='dotted', linewidth=1, label=f'MAP Q1: {map_q1:.2f}')
        plt.axvline(map_q3, color='green', linestyle='dotted', linewidth=1, label=f'MAP Q3: {map_q3:.2f}')
        
        plt.title(f'{name} SBP/DBP/MAP Distributions')
        plt.xlabel('mmHg')
        plt.ylabel('Density')
        plt.legend()  # Update legend to include lines
        plt.tight_layout()

        plt.savefig(os.path.join(savepath, f'{name}_sbp_dbp_map_distribution.png'), dpi=1000)
        plt.close()                    
    else:
        plt.title(f'{name} SBP and DBP Distributions')
        plt.xlabel('mmHg')
        plt.ylabel('Density')
        plt.legend()  # Update legend to include lines
        plt.tight_layout()

        plt.savefig(os.path.join(savepath, f'{name}_sbp_dbp_distribution.png'), dpi=1000)
        plt.close()
        
        
def calculate_dataloaders_mean_std(dataloaders, dataloaders_names, savepath=f'./figs/dataset', meta_dataloader=False):
    r"""
    Calculates SBP, DBP, and MAP distributions of the train/val/test split accessed through the corresponding dataloaders.
    The function plots them with mean, standard deviation, and quartiles.
    This function should be called only from dataset.py.
    
    Parameters
    ------------
    dataloaders : list of DataLoader
        List of dataloaders containing the data to analyze.
    dataloaders_names : list of str
        List of names corresponding to each dataloader, used for labeling the plots.
    savepath : str, default './figs/dataset'
        Path to save the generated plots.
    meta_dataloader : bool, default False
        Flag indicating whether the dataloader is a meta-dataloader.
        
    Returns
    ------------
    None (saves the plots to the specified savepath)   
    """

    for dataloader, dataloader_name in zip(dataloaders, dataloaders_names):
        sbp_values = []
        dbp_values = []
        map_values = []

        for batch in dataloader:
            if not meta_dataloader:
                _, annotation = batch
            else:
                (_, Ys), (_, Yq), _ = batch
                annotation = torch.cat([
                    Ys.float().view(-1, Ys.shape[-1]), 
                    Yq.float().view(-1, Yq.shape[-1])
                    ],
                    dim=0
                )
                
            sbp_values.extend(annotation[:, 0].flatten().tolist())
            dbp_values.extend(annotation[:, 1].flatten().tolist())
            map_values.extend(annotation[:, 2].flatten().tolist())
            
        _calculate_dataset_mean_std(
            sbp_values=sbp_values,
            dbp_values=dbp_values,
            map_values=map_values,
            name=dataloader_name,
            savepath=savepath
            )   

    
def calculate_personalization_subjects_mean_std(dataset, args, savepath=f'./figs/dataset'):
    r"""
    Calculates SBP, DBP, and MAP distributions of the personalzation dataset train/val/test splits.
    The function plots them with mean, standard deviation, and quartiles.
    This function should be called only from dataset.py.
    
    Parameters
    ------------
    dataset: PhysioDataset obejct
        Dataset with the list of personalization subjects to analyze.
    savepath : str, default './figs/dataset'
        Path to save the generated plots.
        
    Returns
    ------------
    None (saves the plots to the specified savepath)   
    """

    for subject in dataset.subjects_for_personalization:
                
        # Get all samples of a subject
        subject_sample_ids = dataset.index_by_subject_id[subject]
        train_idx, val_idx, test_idx = split_train_val_test_personalization(
            subject_sample_ids, 
            fixed_train_size=dataset.personalization_sample_number
            )
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.loader_worker)
        val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.loader_worker)
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.loader_worker)
        
        subject_fig_path = os.path.join(savepath, str(subject))
        os.makedirs(subject_fig_path, exist_ok=True)
        
        calculate_dataloaders_mean_std(
            dataloaders=[train_dataloader, val_dataloader, test_dataloader],
            dataloaders_names=[f'{str(subject)}-Train', f'{str(subject)}-Val', f'{str(subject)}-Test'],
            savepath=subject_fig_path
        )
         

def plot_subject_sample_distribution(subject_sample_dict, ids, savepath="subject_sample_distribution.png"):
    r"""
    Plots the distribution of sample counts per subject in a dataset.
    
    Parameters
    ------------
    subject_sample_dict : dict
        Dictionary where keys are subject IDs and values are lists of sample IDs for each subject.
    ids : list
        List of subject IDs to consider for the distribution plot.
    savepath : str, optional
        File path to save the generated plot. Defaults to "subject_sample_distribution.png".
    
    Returns
    ------------
    None (saves the plot to the specified savepath)
    """
    
    subject_sample_counts = [len(subject_sample_dict[id]) for id in ids]
            
    plt.figure(figsize=(10, 6))
    sns.histplot(subject_sample_counts, kde=False)  # kde=False removes the kernel density estimate line
    plt.title("Distribution of Sample Counts per Subject")
    plt.xlabel("Number of Samples")
    plt.ylabel("Number of Subjects")

    # Calculate and display mean, standard deviation, min, and max
    mean_samples = np.mean(subject_sample_counts)
    std_samples = np.std(subject_sample_counts)
    min_samples = np.min(subject_sample_counts)
    max_samples = np.max(subject_sample_counts)
    
    plt.text(0.95, 0.95, f'Mean: {mean_samples:.2f}\nStd: {std_samples:.2f}\nMin: {min_samples}\nMax: {max_samples}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(savepath, dpi=1000)
    plt.close()


def plot_train_val_test_samples_distribution(train_samples_list, val_samples_list, test_samples_list, title='', savepath='./figs'):
    r"""
    Plots the distribution of samples across train, validation, and test sets.

    Parameters
    ------------
    train_samples_list : list
        List of sample IDs for the training set.
    val_samples_list : list
        List of sample IDs for the validation set.
    test_samples_list : list
        List of sample IDs for the test set.
    savepath : str, optional
        File path to save the generated plot. Defaults to './figs/dataset_overview.png'.

    Returns
    ------------
    None (saves the plot to the specified savepath)
    """
    x_values = ['train', 'val', 'test']
    y_values = [len(train_samples_list), len(val_samples_list), len(test_samples_list)]
    
    # Create a Pandas DataFrame for Seaborn
    df = pd.DataFrame({
        'Split': x_values,
        'Samples': y_values
    })

    # Pastel color palette
    pastel_palette = sns.color_palette("pastel")

    # Plotting with Seaborn
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    sns.barplot(x='Split', y='Samples', hue='Split', data=df, ax=axes, palette=pastel_palette, legend=False)  # Added hue and legend=False
    axes.set_ylabel('# Samples')

    fig.suptitle(f'{title}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap with suptitle

    plt.savefig(savepath, dpi=1000)
    plt.close()


def plot_pretraining_personalization_subjects_distribution(pretraining_subjects, personalization_subjects, title='', savepath='./figs/two_series_distribution.png'):
    r"""
    Plots the distribution of counts for two series of values.

    Parameters
    ------------
    pretraining_subjects : list
        List of items representing the first series.
    personalization_subjects : list
        List of items representing the second series.
    title : str, optional
        Title of the plot. Defaults to ''.
    savepath : str, optional
        File path to save the generated plot. Defaults to './figs/two_series_distribution.png'.

    Returns
    ------------
    None (saves the plot to the specified savepath)
    """
    series_names = ['Pretraining', 'Personalization']
    series_counts = [len(pretraining_subjects), len(personalization_subjects)]

    # Create a Pandas DataFrame for Seaborn
    df = pd.DataFrame({
        'Split': series_names,
        'Count': series_counts
    })

    # Pastel color palette
    pastel_palette = sns.color_palette("pastel")

    # Plotting with Seaborn
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    sns.barplot(x='Split', y='Count', hue='Split', data=df, ax=axes, palette=pastel_palette, legend=False)  # Added hue and legend=False
    axes.set_ylabel('# subjects')

    fig.suptitle(f'{title}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap with suptitle

    plt.savefig(savepath, dpi=1000)
    plt.close()



def plot_signals(signals, labels=None, title="Signals Plot", fs=125, savepath='./figs', ylabels=None):
    r"""
    Plots multiple 1D signals on the same figure.

    Parameters
    ------------
    
    signals (list or numpy.ndarray): 
        A list of 1D NumPy arrays, or a 2D NumPy array where each row is a signal.
    labels (list, optional): 
        A list of strings, one for each signal, to use as labels in the legend. Defaults to None.
    title (str, optional): 
        The title of the plot. Defaults to "Signals Plot".
    fs (int, optional): 
        The sampling frequency of the signals. Defaults to 125.
    savepath (str, optional):
        The directory where the plot will be saved. Defaults to './figs'.
    ylabels (list, optional):
        A list of y-axis labels for each signal. If None, defaults to "Amplitude" for all signals.
        
    Returns
    ------------
    None: 
        The function saves the plot as a .png file in the specified savepath.    
    """

    num_signals = len(signals) if isinstance(signals, list) else signals.shape[0] # Handle list or 2D array input
    time = np.arange(signals[0].size if isinstance(signals, list) else signals.shape[1]) / fs  # Time vector in seconds

    plt.figure(figsize=(12, 2 * num_signals))  # Adjust figure size as needed
    plt.suptitle(title, fontsize=14)  # Overall plot title

    for i in range(num_signals):
        plt.subplot(num_signals, 1, i + 1)  # Create subplots

        signal = signals[i] if isinstance(signals, list) else signals[i, :] # Get the signal

        plt.plot(time, signal)  # Plot the signal
        if labels:
            plt.title(labels[i])  # Use provided labels
        else:
            plt.title(f"Signal {i+1}")  # Default title

        plt.xlabel("Time (seconds)")
        if ylabels is None:
            plt.ylabel("Amplitude")
        else:
            plt.ylabel(ylabels[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot params for title
    plt.savefig(os.path.join(savepath, f'{title}.png'), dpi=1000)
    plt.close()


def plot_abp(signal : np.array, fs : int, flat_locs_sig : np.array = None, peaks : np.array = None, valleys: np.array = None, title : str = 'ABP', save_path : str = './figs') -> None:
    r"""
    Handy function to plot a signal, with its peaks, valleys, flat parts, outliers, and lower/upper envelops (when provided).

    Parameters
    ------------

    signal: np.array,
        the signal to analyze
    fs: int,
        the sampling rate of the signal
    flat_locs_sig: np.array, default None,
        the locations of the flat lines
    peaks: np.array, default None,
        the locations of the peaks
    valleys: np.array, default None,
        the locations of the valleys
    title: str, default '',
        the title of the plot
    save_path: str, default './',
        where to save the image

    Returns
    ------------
    None, saves the plot as a .png file in the specified save_path.
    """

    # Seconds on the x-axis, amplitude on the y-axis
    t = np.arange(0, (len(signal) / fs), 1.0 / fs)
    
    plt.figure(figsize=(12, 4))  # Adjust figure size as needed
    plt.title(f'{title}')
    plt.xlabel('s')
    plt.ylabel('mmHg')
    plt.plot(t, signal, label='abp')

    if peaks is not None:
        # If peaks are provided, then also the upper envelope of the signal is plotted
        x_vals = np.arange(len(signal))
        up_env = np.interp(x_vals, peaks, signal[peaks])
        plt.plot(t, up_env, color='red', label='up envelope', marker='o', linestyle='dashed', linewidth=1, markersize=1)
        plt.scatter(t[peaks], signal[peaks], color='red', label='peaks')

    if valleys is not None:
        # If valleys are provided, then also the lower envelope of the signal is plotted
        x_vals = np.arange(len(signal))
        down_env = np.interp(x_vals, valleys, signal[valleys])
        plt.plot(t, down_env, color='blue', label='down envelope', marker='o', linestyle='dashed', linewidth=1, markersize=1)
        plt.scatter(t[valleys], signal[valleys], color='blue', label='valleys')

    if flat_locs_sig is not None:
        plt.scatter(t[flat_locs_sig], signal[flat_locs_sig], color='green', label='flat lines')

    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, f'{title}.png'), dpi=1000)
    plt.close()
    
    
def plot_augmented_views(aug_signal_0, aug_signal_1, title, savepath, ecg=False):
    r"""
    Generates a comparative plot of two augmented views of the same signal 
    segment for visual validation of augmentation strategies.

    This function is particularly useful when debugging contrastive learning 
    pipelines, where the goal is to ensure that augmentations are strong enough 
    to be challenging but not so destructive that the underlying physiological 
    features (like the PPG pulse or ECG R-peak) are lost.

    Parameters
    ------------
    aug_signal_0 (torch.Tensor): 
        The first augmented version of the signal. Expected shape is $(T, C)$.
    aug_signal_1 (torch.Tensor): 
        The second augmented version of the signal. Expected shape is $(T, C)$.
    title (str): 
        The filename and title for the generated plot.
    savepath (str): 
        Directory where the resulting PNG file will be stored.
    ecg (bool, optional): 
        If True, the function expects and plots both PPG and ECG channels. 
        If False, only the PPG channel is plotted. Defaults to False.
    """
    if ecg:
        _, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(aug_signal_0[:,0].numpy())
        axes[0].set_title("View 0 - PPG")
        axes[1].plot(aug_signal_0[:,1].numpy())
        axes[1].set_title("View 0 - ECG")

        axes[2].plot(aug_signal_1[:,0].numpy())
        axes[2].set_title("View 1 - PPG")
        axes[3].plot(aug_signal_1[:,1].numpy())
        axes[3].set_title("View 1 - ECG")

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{title}.png'), dpi=200)
        plt.close()
    else:
        _, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(aug_signal_0[:,0].numpy())
        axes[0].set_title("View 0 - PPG")
        axes[1].plot(aug_signal_1[:,0].numpy())
        axes[1].set_title("View 1 - PPG")

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{title}.png'), dpi=200)
        plt.close()
        

def plot_subject_validity_over_time(subject_id, subject_windows, window_length, fs, savepath):
    r"""
    Generates a longitudinal visualization of a subject's recording, overlaying 
    physiological trends with data quality classifications.

    The plot uses background color-coding (shading) to identify different data 
    regimes, which is essential for semi-supervised or self-supervised learning 
    strategies where unlabeled data must be distinguished from low-quality noise.

    Data Regimes:
    - **Supervised (Green)**: PPG, ECG, and ABP are all valid. Suitable for training.
    - **Unlabeled (Blue)**: PPG and ECG are valid, but ABP is missing or corrupted. 
      Suitable for self-supervised pre-training.
    - **Invalid (Red)**: The input sensors (PPG or ECG) are corrupted. Must be discarded.

    Parameters
    ------------
    subject_id (str/int): 
        Identifier for the patient.
    subject_windows (list): 
        A list of dictionaries containing windowed data and validity flags 
        ('abp_valid', 'ppg_valid', 'ecg_valid').
    window_length (float): 
        Duration of each window in seconds.
    fs (int): 
        Sampling frequency.
    savepath (str): 
        Directory where the resulting PNG plot will be saved.
    """
    if not subject_windows:
        print(f"No valid windows to plot for Subject {subject_id}.")
        return

    # Extract MAP, and validity flags
    map_values = []
    abp_validity = []
    ppg_validity = []
    ecg_validity = []

    for window_data in subject_windows:
        # Use raw ABP if available, otherwise just use a placeholder
        current_abp = window_data['abp_raw']
        if current_abp is not None and len(current_abp) > 0:
            map_val = np.mean(current_abp) # Use Mean Arterial Pressure (MAP) for overall trend
        else:
            map_val = np.nan # Use NaN for invalid ABP windows

        map_values.append(map_val)
        abp_validity.append(window_data['abp_valid'])
        ppg_validity.append(window_data['ppg_valid'])
        ecg_validity.append(window_data['ecg_valid'])

    num_windows = len(subject_windows)
    time_axis = np.arange(num_windows) * window_length # Time in seconds

    fig, ax = plt.subplots(figsize=(15, 6)) # Use ax for direct plotting
    ax.plot(time_axis, map_values, label='Mean ABP (MAP)', color='black', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Mean ABP (mmHg)')
    ax.set_title(f'Subject {subject_id} ABP and Signal Validity Over Time')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Mark regions based on validity flags
    current_region_type = None
    current_region_start_idx = 0
    current_region_type_color = None # Initialize color

    # Define color map for regions
    region_colors = {
        'supervised': 'green',
        'unlabeled': 'blue',
        'invalid': 'red',
        'unknown': 'white'
    }

    for i in range(num_windows):
        is_supervised = ppg_validity[i] and ecg_validity[i] and abp_validity[i]
        is_unlabeled_candidate = ppg_validity[i] and ecg_validity[i] and not abp_validity[i]
        is_invalid_input = not ppg_validity[i] or not ecg_validity[i]

        if is_supervised:
            region_type = 'supervised'
        elif is_unlabeled_candidate:
            region_type = 'unlabeled'
        elif is_invalid_input:
            region_type = 'invalid'
        else: # Should not happen if logic is correct, but for safety
            region_type = 'unknown'

        if current_region_type is None:
            current_region_type = region_type
            current_region_start_idx = i
            current_region_type_color = region_colors[region_type]
        elif region_type != current_region_type:
            # End of previous region, plot it
            ax.axvspan(time_axis[current_region_start_idx], time_axis[i],
                       facecolor=current_region_type_color, alpha=0.1)
            current_region_type = region_type
            current_region_start_idx = i
            current_region_type_color = region_colors[region_type]

    # Plot the last region
    if current_region_type is not None:
        ax.axvspan(time_axis[current_region_start_idx], time_axis[num_windows - 1] + window_length,
                   facecolor=current_region_type_color, alpha=0.1)

    # Custom legend for the shaded regions and the line plot
    # Get handles and labels for existing plot elements (the MAP line)
    handles, labels = ax.get_legend_handles_labels()

    # Create dummy patches for the region types for the legend
    legend_patches = [
        Patch(facecolor='green', alpha=0.1, label='Supervised (PPG,ECG,ABP Valid)'),
        Patch(facecolor='blue', alpha=0.1, label='Unlabeled (PPG,ECG Valid,ABP Invalid)'),
        Patch(facecolor='red', alpha=0.1, label='Invalid Input (PPG or ECG Invalid)')
    ]

    # Combine handles and labels. Filter out any duplicate labels if ax.axvspan also created labels
    # We explicitly define the labels for the patches to avoid duplicates from axvspan
    all_handles = handles + legend_patches
    all_labels = labels + [patch.get_label() for patch in legend_patches]

    # Create a new list of unique handles and labels in the desired order
    unique_legend_elements = {}
    for h, l in zip(all_handles, all_labels):
        # Prioritize the explicitly defined patch labels if a key exists
        if l not in unique_legend_elements:
            unique_legend_elements[l] = h
        # If the label already exists, ensure the handle is the one we prefer (e.g., the line vs. a small shaded region if there's overlap)
        # For this case, line handles are typically added first from ax.get_legend_handles_labels()
        # and then patches, so the line will be correctly picked for 'Mean ABP (MAP)'.

    ax.legend(handles=list(unique_legend_elements.values()), labels=list(unique_legend_elements.keys()))
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'subject_{subject_id}_validity_plot.png'), dpi=1000)
    plt.close() 
    

def plot_consecutive_runs_subject(runs, subject_id, savepath="subject_run_lengths.png"):
    r"""
    Plots the distribution of run lengths (consecutive windows) for a single subject.

    Parameters
    ------------
    runs : list of dict
        Output from `find_consecutive_runs`, where each dict contains:
            - "start_idx": start index in subject sample list
            - "end_idx": end index in subject sample list
            - "values": the consecutive values
            - "length": length of the run
    subject_id : int
        The subject identifier.
    savepath : str, optional
        Path to save the generated figure (default="subject_run_lengths.png").
    """
    run_lengths = [r["length"] for r in runs]
    run_positions = [r["start_idx"] for r in runs]

    mean_len = np.mean(run_lengths)
    min_len = np.min(run_lengths)
    max_len = np.max(run_lengths)

    plt.figure(figsize=(10, 6))
    plt.bar(run_positions, run_lengths, width=1.0, align="center", alpha=0.7)
    plt.xlabel("Run start index (in subject sample list)")
    plt.ylabel("Run length (# consecutive windows)")
    plt.title(f"Consecutive run lengths for Subject {subject_id}")

    # Display stats inside the plot
    plt.text(0.95, 0.95,
             f'Mean: {mean_len:.2f}\nMin: {min_len}\nMax: {max_len}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_consecutive_block_all(all_blocks, savepath="all_subjects_block_lengths.png"):
    r"""
    Plots the cumulative distribution of block lengths across all subjects.

    Parameters
    ------------
    all_blocks : list of list of dict
        A list where each element corresponds to one subject's blocks (output of `find_consecutive_blocks`).
        Example: [ [block_dict, block_dict, ...],   # subject 1
                   [block_dict, block_dict, ...],   # subject 2
                   ... ]
    savepath : str, optional
        Path to save the generated figure (default="all_subjects_block_lengths.png").
    """
    # Flatten block lengths across subjects
    all_lengths = [r["length"] for subj_blocks in all_blocks for r in subj_blocks]

    mean_len = np.mean(all_lengths)
    min_len = np.min(all_lengths)
    max_len = np.max(all_lengths)

    plt.figure(figsize=(10, 6))
    sns.histplot(all_lengths, bins=50, kde=False, edgecolor="black")
    plt.xlabel("Block length (# consecutive windows)")
    plt.ylabel("Count")
    plt.title("Distribution of consecutive block lengths across all subjects")

    # Display stats inside the plot
    plt.text(0.95, 0.95,
             f'Mean: {mean_len:.2f}\nMin: {min_len}\nMax: {max_len}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_subject_annotation_blocks(dataset, subject_id, blocks, savepath="subject_annotation_plots.jpg", show_bp_plot=False):
    """
    Plot annotation statistics for a subject with fixed interleaved blocks.

    Parameters
    ----------
    dataset : OnlineSubjectDataset
        Dataset instance (must allow __getitem__ access to annotation tensors).
    subject_id : int
        Subject identifier.
    blocks : list of dict
        Each dict contains sample IDs
    savepath : str
        File path to save the figure.
    show_bp_plot : bool, optional
        If True, also plot SBP/DBP/MAP evolution. Default=False.
    """
    
    # Activate subject
    full_index_list = dataset.index_by_subject_id[subject_id]

    # Helper: extract SBP/DBP/MAP
    def get_annotations(sample_ids):
        sbp_values, dbp_values, map_values = [], [], []
        for sid in sample_ids:
            _, ann, _ = dataset.__getitem__(sid)
            sbp_values.append(float(ann[0]))
            dbp_values.append(float(ann[1]))
            map_values.append(float(ann[2]))
        return sbp_values, dbp_values, map_values

    # Collect block-level data
    block_idxs_list, batch_idxs_list, set_list = [], [], []
    sbp_list, dbp_list, map_list, window_indices = [], [], [], []

    for block in blocks:
        
        # Adaptation windows
        train_sbp, train_dbp, train_map = get_annotations(block["sample_ids"])
        adapt_window_indices = [full_index_list.index(sid) for sid in block["sample_ids"]]

        batch_idxs_list.extend([block["batch_idx"]] * len(train_sbp))
        block_idxs_list.extend([block["block_idx"]] * len(train_sbp))
        set_list.extend(['batches'] * len(train_sbp))
        sbp_list.extend(train_sbp)
        dbp_list.extend(train_dbp)
        map_list.extend(train_map)
        window_indices.extend(adapt_window_indices)
        
    # Build DataFrame and sort chronologically
    df = pd.DataFrame({
        'window_index': window_indices,
        'block_idx': block_idxs_list,
        'batch_idx': batch_idxs_list,
        'set': set_list,
        'sbp': sbp_list,
        'dbp': dbp_list,
        'map': map_list
    }).sort_values(by='window_index').reset_index(drop=True)
    
    # Create the plot with three horizontal subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Subject {subject_id} - Annotation Statistics', fontsize=14, fontweight='bold')

    # Subplot 1: Window indices vs Block structure
    ax1 = axes[0]
    
    # Plot points for each run with different colors
    unique_blocks = df['block_idx'].unique()
    colors_blocks = plt.cm.tab10(np.linspace(0, 1, len(unique_blocks)))
    
    for i, block_idx in enumerate(unique_blocks):
        block_data = df[df['block_idx'] == block_idx]
        ax1.scatter(block_data['window_index'], block_data['block_idx'], 
                   c=[colors_blocks[i]], label=f'Block {block_idx}', alpha=0.7, s=30)
    
    ax1.set_ylabel('Block Number')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title('Window Index vs Block Number')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Subplot 2: Window indices vs SBP/DBP/MAP
    ax2 = axes[1]
    ax2.plot(df['window_index'], df['sbp'], 'o-', color='red', 
             label='SBP', alpha=0.7, markersize=4, linewidth=1)
    ax2.plot(df['window_index'], df['dbp'], 'o-', color='blue', 
             label='DBP', alpha=0.7, markersize=4, linewidth=1)
    ax2.plot(df['window_index'], df['map'], 'o-', color='green', 
             label='MAP', alpha=0.7, markersize=4, linewidth=1)
    
    ax2.set_ylabel('Blood Pressure (mmHg)')
    ax2.set_xlabel('Window Index')
    ax2.set_title('Window Index vs Blood Pressure Values')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(savepath)
    plt.close()


def plot_block_length_statistics(dataset, savepath='./data_figs', keep_longest=False):
    r"""
    Plot statistics of block lengths across subjects.

    Parameters
    ----------
    dataset : OnlineSubjectDataset
        Initialized dataset object.
    savepath : str or None
        Path to save plots. If None, plots are only shown.
    keep_longest : bool
        If True, only the longest block per subject is considered.
    """
    block_lengths_by_subject = {}

    for subj in dataset.subjects_for_personalization:
        blocks = dataset.find_consecutive_blocks(dataset.index_by_subject_id[subj], dataset.input_seq_len_s)
        lengths = [b["length"] for b in blocks]

        if not lengths:
            continue

        if keep_longest:
            block_lengths_by_subject[subj] = [max(lengths)]
        else:
            block_lengths_by_subject[subj] = lengths

    # Flatten all block lengths
    all_lengths = [l for lengths in block_lengths_by_subject.values() for l in lengths]

    # --- Summary statistics ---
    print(f"Total subjects: {len(block_lengths_by_subject)}")
    print(f"Total blocks: {len(all_lengths)}")
    print(f"Mean block length: {np.mean(all_lengths):.2f}")
    print(f"Median block length: {np.median(all_lengths):.2f}")
    print(f"Max block length: {np.max(all_lengths)}")
    print(f"Min block length: {np.min(all_lengths)}")

    # --- Histogram ---
    plt.figure(figsize=(12, 10))
    plt.hist(all_lengths, bins=50, color="steelblue", alpha=0.7)
    plt.xlabel("Block length (#windows)")
    plt.ylabel("Frequency")
    plt.title("Distribution of block lengths across all subjects")
    plt.savefig(os.path.join(savepath, "block_length_histogram.png"))
    
    # --- Boxplot per subject ---
    plt.figure(figsize=(20, 10))
    plt.boxplot([block_lengths_by_subject[subj] for subj in block_lengths_by_subject],
                showfliers=False)
    plt.xlabel("Subjects")
    plt.xticks(ticks=range(1, len(block_lengths_by_subject) + 1),
           labels=list(block_lengths_by_subject.keys()),
           rotation=90, ha='right') 
    plt.ylabel("Block length (#windows)")
    plt.title("Block length distribution per subject")
    plt.savefig(os.path.join(savepath, "block_length_boxplot.png"))
    plt.close()


def plot_meta_dataset_run_distribution(meta_ds, dataset_name="train", savepath='./data_figs', bin_width=5):
    r"""
    Plot the distribution of patients grouped by the number of runs they have.

    Each bin groups patients whose number of runs falls within a given range,
    e.g., 1-10, 11-20, etc.

    Parameters
    ----------
    meta_ds : MetaTaskDataset
        The MetaTaskDataset instance (train, val, or test) containing
        the precomputed 'runs_by_patient' dictionary.

    dataset_name : str, optional
        Name of the dataset split (e.g., "train", "val", "test").
        Used in the plot title and axis labels.

    bin_width : int, optional
        Width of each bin (range of number of runs grouped together).
        Default is 10 (i.e., bins like 1-10, 11-20, etc.).
    """
    # Extract number of runs per patient
    patient_ids = list(meta_ds.runs_by_patient.keys())
    num_runs = np.array([len(meta_ds.runs_by_patient[pid]) for pid in patient_ids], dtype=int)

    if len(num_runs) == 0:
        raise ValueError("No patient runs found in the provided MetaTaskDataset.")

    # Compute binned labels
    max_runs = num_runs.max()
    bin_edges = np.arange(0, max_runs + bin_width, bin_width)
    # Shift first bin to start from 1 instead of 0
    bin_edges[0] = 1
    bin_labels = [f"{int(b1)}-{int(b2)}" for b1, b2 in zip(bin_edges[:-1], bin_edges[1:])]

    # Assign each patient to a bin
    bins = pd.cut(num_runs, bins=bin_edges, labels=bin_labels, include_lowest=True, right=True)
    bin_counts = bins.value_counts().sort_index()

    # Create the plot
    sns.set_theme(style="whitegrid")
    pastel_blue = sns.color_palette("pastel")[0]  # soft blue tone

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        x=bin_counts.index,
        y=bin_counts.values,
        color=pastel_blue,
        edgecolor="black",
        ax=ax
    )

    ax.set_title(f"Distribution of Patients by Number of Runs", fontsize=14)
    ax.set_xlabel("Number of Runs (Grouped)", fontsize=12)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7, axis="y")

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add value annotations above bars
    for i, val in enumerate(bin_counts.values):
        ax.text(i, val + 0.5, str(val), ha='center', va='bottom', fontsize=10)
        
    save_file = os.path.join(savepath, f'{dataset_name}_run_length_distribution.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_blockwise_mae(per_block_stats, index_to_plot, subject_id, savepath):
    r"""
    Generates a chronological performance plot showing the Mean Absolute Error (MAE) 
    and variability (STD) for multiple baselines across sequential data blocks.

    Parameters
    ------------
    per_block_stats (dict): 
        A dictionary where keys are baseline names (e.g., 'EWC', 'MAML') and 
        values are sub-dictionaries containing lists of MAE and STD values 
        per chronological block.
    index_to_plot (str): 
        The specific metric to visualize, typically 'sbp', 'dbp', or 'map'.
    subject_id (str/int): 
        The unique identifier for the patient being analyzed.
    savepath (str): 
        The full destination path (including filename) for the output plot.
    """
    blocks = len(next(iter(per_block_stats.values()))[f'{index_to_plot}_mae'])
    x = np.arange(1, blocks + 1)
    plt.figure(figsize=(12,8))
    for b in per_block_stats.keys():
        mae_arr = per_block_stats[b][f'{index_to_plot}_mae']
        std_arr = per_block_stats[b][f'{index_to_plot}_std']
        mae_plot = np.array([np.nan if v is None else v for v in mae_arr])
        std_plot = np.array([np.nan if v is None else v for v in std_arr])
        plt.errorbar(x, mae_plot, yerr=std_plot, label=b, marker='o')
    plt.xlabel('Block index (chronological)')
    plt.ylabel('MAE (with STD errorbars)')
    plt.title(f'Subject {subject_id} - Blockwise {index_to_plot} test MAE per baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
        
        
def plot_age_gender_distribution(valid_subjects, index_file_path, savepath="./figs", filename=""):
    r"""
    Function description
    --------------------
    This function plots and saves two visualizations for the valid subjects:
    (1) An age distribution histogram.
    (2) A gender distribution donut chart.
    The figure is saved to the specified savepath folder.

    Parameters
    ------------
    valid_subjects : list
        List of valid subject IDs to include in the visualization.

    index_file_path : str
        Path to the CSV file containing subject metadata with columns:
        ['subject_id', 'age', 'gender'], where 'gender' is encoded as
        0 for female and 1 for male.

    savepath : str, optional
        Root folder where the figure will be saved. Defaults to "./figs".
    """

    # Load index file
    df = pd.read_csv(index_file_path)

    # Filter for valid subjects
    valid_df = df[df['subject_id'].isin(valid_subjects)]

    # Prepare figure
    plt.figure(figsize=(12, 5))
    pastel_blue = '#AEC6CF'
    pastel_pink = '#FFD1DC'
    
    # --- AGE DISTRIBUTION HISTOGRAM ---
    plt.subplot(1, 2, 1)
    n_bins = 10
    plt.hist(valid_df['age'], bins=n_bins, color=pastel_blue, edgecolor='gray', alpha=0.85)
    plt.xlabel('Age')
    plt.ylabel('Number of Subjects')
    plt.title('Dataset Subjects Age Distribution')
    plt.grid(alpha=0.3, linestyle='--')

    # --- GENDER DISTRIBUTION (DONUT PLOT) ---
    plt.subplot(1, 2, 2)
    gender_counts = valid_df['gender'].value_counts().sort_index()
    labels = ['Female', 'Male']
    sizes = [gender_counts.get(0, 0), gender_counts.get(1, 0)]
    colors = [pastel_pink, pastel_blue]

    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'color': 'black'}
    )

    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Dataset Subjects Gender Distribution')
    plt.axis('equal')

    plt.tight_layout()

    # Save figure
    filename += "age_gender_distribution.png"
    save_file = os.path.join(savepath, filename)
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'aurora' in index_file_path:
        # --- HEIGHT AND WEIGHT DISTRIBUTION (AURORA SPECIFIC) ---
        plt.figure(figsize=(12, 5))
        
        # --- HEIGHT DISTRIBUTION ---
        plt.subplot(1, 2, 1)
        plt.hist(valid_df['height'], bins=10, color='#B39EB5', edgecolor='gray', alpha=0.85) # Pastel Purple
        plt.xlabel('Height (m)')
        plt.ylabel('Number of Subjects')
        plt.title('Dataset Subjects Height Distribution')
        plt.grid(alpha=0.3, linestyle='--')

        # --- WEIGHT DISTRIBUTION ---
        plt.subplot(1, 2, 2)
        plt.hist(valid_df['weight'], bins=10, color='#77DD77', edgecolor='gray', alpha=0.85) # Pastel Green
        plt.xlabel('Weight (kg)')
        plt.ylabel('Number of Subjects')
        plt.title('Dataset Subjects Weight Distribution')
        plt.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save the aurora-specific figure
        aurora_filename = filename.replace("age_gender_distribution.png", "") + "height_weight_distribution.png"
        aurora_save_file = os.path.join(savepath, aurora_filename)
        plt.savefig(aurora_save_file, dpi=300, bbox_inches='tight')
        plt.close()
    

def plot_update_summary_table(updates_dict, save_path):
    r"""
    Plot and save a pastel-colored bar chart showing the number of adaptation updates
    performed by each baseline during online personalization.

    Parameters
    -------------------
    updates_dict : dict
        Dictionary mapping baseline names (str) → either:
          - int  (for per-subject number of updates)
          - dict with {"mean": float, "std": float} (for aggregated updates)

    save_path : str
        Path (including filename and extension) where the resulting figure will be saved.
        Example: "./figures/update_summary_subject_01.png"
    """
    # Detect whether input is aggregated (dicts with mean/std) or per-subject (ints)
    first_val = next(iter(updates_dict.values()))
    aggregated = isinstance(first_val, dict)

    if aggregated:
        df = pd.DataFrame([
            {"Baseline": k, "Mean Updates": v["mean"], "Std Updates": v["std"]}
            for k, v in updates_dict.items()
        ])
    else:
        df = pd.DataFrame([
            {"Baseline": k, "Number of Updates": v}
            for k, v in updates_dict.items()
        ])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    if aggregated:
        ax = sns.barplot(
            data=df,
            x="Baseline",
            y="Mean Updates",
            hue="Baseline",
            dodge=False,
            palette="pastel",
            legend=False,
            errorbar=None  # we'll handle error bars manually
        )
        # Add manual error bars (std)
        ax.errorbar(
            x=range(len(df)),
            y=df["Mean Updates"],
            yerr=df["Std Updates"],
            fmt="none",
            ecolor="gray",
            elinewidth=1.5,
            capsize=4,
            capthick=1.2
        )
        ax.set_title("Aggregated Adaptation Updates per Baseline", fontsize=14, weight="bold", pad=12)
        ax.set_ylabel("Mean Number of Updates ± Std", fontsize=12)
    else:
        ax = sns.barplot(
            data=df,
            x="Baseline",
            y="Number of Updates",
            hue="Baseline",
            dodge=False,
            palette="pastel",
            legend=False
        )
        ax.set_title("Adaptation Updates per Baseline", fontsize=14, weight="bold", pad=12)
        ax.set_ylabel("Number of Updates", fontsize=12)

    ax.set_xlabel("Baseline", fontsize=12)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    

def plot_feature_distance_over_time(feature_df, subject_id, baseline, savepath):
    r"""
    Visualizes the evolution of feature distances to detect physiological drift
    across a subject data stream.

    Parameters
    ------------
    feature_df (pd.DataFrame): 
        A DataFrame containing the calculated distances. It must include:
        - `train_scaled`: Scaled distances for training/calibration blocks.
        - `test_scaled`: Scaled distances for upcoming test blocks.
    subject_id (str/int): 
        The unique identifier for the patient.
    baseline (str): 
        The name of the model or method used to generate the features.
    savepath (str): 
        The full destination path where the plot will be saved.
    """
    x = np.arange(len(feature_df))

    plt.figure(figsize=(12, 8))
    plt.plot(x, feature_df["train_scaled"], label="Train", marker="o")
    plt.plot(x, feature_df["test_scaled"], label="Test", marker="x")
    plt.xlabel("Streamed block index")
    plt.ylabel("Scaled feature distance")
    plt.title(f"Feature drift — Subject {subject_id} — {baseline}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()
    

def plot_sbp_drift_distribution(drift_info, subject_ids, title, savepath):
    r"""
    Visualizes the distribution of within-subject SBP drift across a 
    specific group of subjects.

    This histogram helps identify whether the dataset consists mostly of 
    physiologically stable subjects or highly volatile ones. The inclusion 
    of quartile lines ($Q_{25}$, $Q_{50}$, $Q_{75}$) provides a clear 
    benchmark for categorizing subjects into different "drift regimes" 
    used in meta-learning or continual learning splits.

    Parameters
    ------------
    drift_info (dict): 
        A dictionary containing calculated drift metrics for all subjects, 
        specifically looking for the 'sbp_std_over_time' key.
    subject_ids (list): 
        The list of subject IDs to include in this specific distribution plot 
        (e.g., only the training set subjects).
    title (str): 
        The title of the plot and the base name for the saved file.
    savepath (str): 
        The directory path where the resulting PNG image will be stored.
    """
    values = np.array([
        drift_info[s]["sbp_std_over_time"]
        for s in subject_ids
        if s in drift_info
    ])

    q25, q50, q75 = np.percentile(values, [25, 50, 75])

    plt.figure(figsize=(10, 7))
    plt.hist(values, bins=30, alpha=0.7, edgecolor="black")

    for q, label in zip([q25, q50, q75], ["25%", "50%", "75%"]):
        plt.axvline(q, linestyle="--", label=label)

    plt.xlabel("SBP temporal drift (std over time)")
    plt.ylabel("Number of subjects")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{title}.png"), dpi=300)
    plt.close()


def plot_drift_regime_counts(drift_info, subject_ids, q25, q75, title, savepath):
    r"""
    Visualizes the count of subjects assigned to 'Low', 'Medium', and 'High' 
    drift regimes based on pre-defined SBP drift thresholds.

    This categorical visualization ensures that the data partitioning strategy 
    (especially for drift-aware training) has resulted in a balanced 
    distribution of physiological phenotypes. It helps researchers confirm 
    that the model is being exposed to enough challenging (High-drift) 
    scenarios during the pre-training or meta-learning phases.

    Parameters
    ------------
    drift_info (dict): 
        Dictionary containing the 'sbp_std_over_time' metric for all subjects.
    subject_ids (list): 
        The specific subset of subject IDs to be categorized and counted 
        (e.g., the meta-learning pool).
    q25 (float): 
        The 25th percentile threshold; subjects below this are labeled 'Low' drift.
    q75 (float): 
        The 75th percentile threshold; subjects above this are labeled 'High' drift.
    title (str): 
        Title for the plot and the resulting filename.
    savepath (str): 
        Directory where the PNG plot will be stored.
    """
    regimes = []

    for s in subject_ids:
        if s not in drift_info:
            raise ValueError(f"Subject {s} not found in drift_info.")

        v = drift_info[s]["sbp_std_over_time"]
        if v <= q25:
            regimes.append("Low")
        elif v <= q75:
            regimes.append("Medium")
        else:
            regimes.append("High")

    counts = Counter(regimes)

    plt.figure(figsize=(10, 8))
    plt.bar(counts.keys(), counts.values())
    plt.ylabel("Number of subjects")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{title}.png"), dpi=300)
    plt.close()
    
    
def plot_sbp_drift_over_time(df: pd.DataFrame, subject_id: str, savepath: str):
    r"""
    Generates a multi-panel visualization of Systolic Blood Pressure trends 
    and drift metrics over successive personalization steps.

    The plot tracks three key perspectives:
    1.  **Direct Trends**: Compares immediate batch averages against a smoothed 
        rolling average to identify local volatility versus long-term trends.
    2.  **Absolute Drift**: Measures the raw mmHg difference from the 
        starting point, indicating the magnitude of physiological change.
    3.  **Relative Drift**: Normalizes the change, which is used as 
        a trigger for adaptation.

    Parameters
    ------------
    df (pd.DataFrame): 
        A DataFrame containing the columns: 'batch_mean_sbp', 'rolling_mean_sbp', 
        'abs_drift_sbp', and 'rel_drift_sbp'.
    subject_id (str): 
        The unique identifier for the patient being visualized.
    savepath (str): 
        The full destination path where the plot image will be saved.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(df["batch_mean_sbp"], label="Batch mean SBP")
    axs[0].plot(df["rolling_mean_sbp"], label="Rolling mean SBP", linestyle="--")
    axs[0].set_ylabel("SBP (mmHg)")
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(df["abs_drift_sbp"], color="orange")
    axs[1].set_ylabel("Absolute drift (mmHg)")
    axs[1].grid(True)

    axs[2].plot(df["rel_drift_sbp"], color="red")
    axs[2].set_ylabel("Relative drift")
    axs[2].set_xlabel("Personalization step")
    axs[2].grid(True)
    
    fig.suptitle(f"SBP Drift - Patient {subject_id}")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_sbp_abs_rel_drift_distributions(
    master_df,
    savepath,
    filename_prefix="sbp_drift"
):
    r"""
    Calculates and visualizes the statistical distributions of SBP drift across 
    the entire dataset, identifying key population percentiles.

    This function generates two distinct plots:
    1.  **Absolute Drift**: A histogram of raw mmHg changes, useful for 
        understanding clinical variance.
    2.  **Relative Drift**: A histogram of normalized changes with specific 
        percentile markers ($P_{50}, P_{70}, P_{80}, P_{90}$). These markers 
        are often used as "trigger thresholds" for continual learning: for 
        example, updating a model only when a patient's drift exceeds the 
        80th percentile of the population.

    Parameters
    ------------
    master_df (pd.DataFrame): 
        The aggregated results DataFrame containing 'abs_drift_sbp' and 
        'rel_drift_sbp' columns for all subjects.
    savepath (str or Path): 
        The directory where the distribution plots will be saved.
    filename_prefix (str, optional): 
        A prefix for the saved PNG files to differentiate between experimental 
        runs. Defaults to "sbp_drift".
    """
    savepath = Path(savepath)

    abs_drift = master_df["abs_drift_sbp"].values
    rel_drift = master_df["rel_drift_sbp"].values

    # -----------------------------
    # Plot 1: Absolute drift
    # -----------------------------
    plt.figure(figsize=(10, 8))
    sns.histplot(abs_drift, bins=40, kde=True)
    plt.xlabel("Absolute SBP drift (mmHg)")
    plt.ylabel("Count")
    plt.title("Absolute SBP Drift Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath / f"{filename_prefix}_abs_drift.png", dpi=300)
    plt.close()

    # -----------------------------
    # Plot 2: Relative drift + percentiles
    # -----------------------------
    percentiles = [50, 70, 80, 90]
    pct_vals = np.percentile(rel_drift, percentiles)

    plt.figure(figsize=(10, 8))
    sns.histplot(rel_drift, bins=40, kde=True)

    for p, v in zip(percentiles, pct_vals):
        plt.axvline(v, linestyle="--", label=f"{p}th: {v:.3f}")

    plt.xlabel("Relative SBP drift")
    plt.ylabel("Count")
    plt.title("Relative SBP Drift Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath / f"{filename_prefix}_rel_drift.png", dpi=300)
    plt.close()
    

def plot_param_updates(df, subject_id, save_path):
    r"""
    Generates a bar chart visualizing the history of parameter updates across 
    adaptation steps for a specific subject.

    The plot highlights the "budget" or "depth" of adaptation by coloring bars 
    based on the `update_mode`. This is crucial for verifying that an adaptive 
    system is correctly choosing between "shallow" updates (e.g., just the head) 
    and "deep" updates (e.g., the whole model) based on the physiological drift.

    Parameters
    ------------
    df (pd.DataFrame): 
        DataFrame containing update logs. Must include:
        - `s_idx`: The global adaptation step index.
        - `fraction_updated`: The percentage of model parameters that were modified.
        - `update_mode`: The category of update ('head', 'temporal', or 'all').
    subject_id (str/int): 
        Identifier for the patient currently being analyzed.
    save_path (str): 
        The full destination path where the plot image will be saved.
    """
    MODE_COLORS = {
        "head": '#AEC6CF',       # pastel blue
        "temporal": '#A8E6CF',   # pastel green
        "all": '#FFD1DC'         # pastel pink
    }

    # Sort by true chronological order
    sort_cols = [c for c in ["batch_idx", "block_idx", "step_idx"] if c in df.columns]
    df = df.sort_values(sort_cols)

    x = df["step_idx"].values
    y = df["fraction_updated"].values
    modes = df["update_mode"].values
    colors = [MODE_COLORS[m] for m in modes]

    plt.figure(figsize=(12, 5))

    plt.bar(
        x,
        y,
        color=colors,
        edgecolor="black",
        linewidth=0.4
    )

    # Legend (manual)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m])
        for m in MODE_COLORS
    ]
    labels = list(MODE_COLORS.keys())
    plt.legend(handles, labels, title="Update mode")

    plt.xlabel("Adaptation step index (step_idx)")
    plt.ylabel("Updated parameters (%)")
    plt.title(f"Adaptive parameter updates — Subject {subject_id}")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pareto_frontier(
    df,
    pareto_df,
    selected,
    shift_stressed,
    calibration_stressed,
    savepath,
    min_patients=85
):
    r"""
    Visualizes the multi-objective optimization trade-off between subject 
    inclusion and data quality constraints.

    This scatter plot highlights configurations that are "non-dominated"—meaning 
    you cannot improve one metric without degrading another. It specifically 
    marks configurations used for different experimental settings (Mixed, 
    Abrupt, and Gradual shifts) against the backdrop of all possible parameter 
    combinations.

    Parameters
    ------------
    df (pd.DataFrame): 
        The full search space of all possible subject-inclusion configurations.
    pareto_df (pd.DataFrame): 
        The subset of configurations that lie on the Pareto frontier.
    selected (pd.Series/dict): 
        The specific configuration chosen for the 'Mixed-Shifts' experimental set.
    shift_stressed (pd.Series/dict): 
        The configuration chosen for the 'Abrupt-Shifts' (High volatility) set.
    calibration_stressed (pd.Series/dict): 
        The configuration chosen for the 'Gradual-Shifts' (Long-term stability) set.
    savepath (str): 
        The destination path where the plot image will be saved.
    min_patients (int, optional): 
        The threshold for feasibility (default=85). Only Pareto points meeting 
        this count are highlighted in red.
    """
    plt.figure(figsize=(10, 8))

    pareto_feasible = pareto_df[pareto_df["N_patients"] >= min_patients]

    # Background: all configurations
    plt.scatter(
        df["A"], df["B"],
        color="lightgray", alpha=0.4,
        label="All configurations"
    )

    # Feasible Pareto points
    plt.scatter(
        pareto_feasible["A"], pareto_feasible["B"],
        s=120, color="red",
        label="≥ 85 patients"
    )

    # Mixed-Shifts configuration
    plt.scatter(
        selected["A"], selected["B"],
        s=220, marker="^",
        color="blue",
        label="Mixed-Shifts Set"
    )

    # Abrupt-Shifts configuration
    plt.scatter(
        shift_stressed["A"], shift_stressed["B"],
        s=220, marker="^",
        color="green",
        label="Abrupt-Shifts Set"
    )

    # Gradual-Shifts configuration
    plt.scatter(
        calibration_stressed["A"], calibration_stressed["B"],
        s=220, marker="^",
        color="orange",
        label="Gradual-Shifts Set"
    )
    
    x_min, x_max = int(df["A"].min()), int(df["A"].max())
    y_min, y_max = int(df["B"].min()), int(df["B"].max())
    
    plt.xticks(np.arange(x_min, x_max + 1, 1), fontsize=12)
    plt.yticks(np.arange(y_min, y_max + 1, 1), fontsize=12)

    plt.xlabel("Minimum abrupt shifts per subject", fontsize=14)
    plt.ylabel("Minimum consecutive blocks per subject", fontsize=14)
    plt.title("Pareto frontier under AAMI/BHS constraint", fontsize=16)
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()
    

def plot_ecg_ppg_fiducials(ecg, ppg, rpeaks, ppg_peaks, ppg_onsets, fs, filename, save_path):
    
    time = np.arange(len(ecg)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ------------------------------------------------
    # ECG
    # ------------------------------------------------

    axes[0].plot(time, ecg, color='lightcoral')

    if len(rpeaks) > 0:
        axes[0].scatter(
            time[rpeaks],
            ecg[rpeaks],
            marker="o",
            label="R-peaks"
        )

    axes[0].set_title("ECG with R-peaks")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    # ------------------------------------------------
    # PPG
    # ------------------------------------------------

    axes[1].plot(time, ppg, color='skyblue')

    if len(ppg_peaks) > 0:
        axes[1].scatter(
            time[ppg_peaks],
            ppg[ppg_peaks],
            marker="o",
            label="PPG Peaks"
        )

    if ppg_onsets is not None and len(ppg_onsets) > 0:
        axes[1].scatter(
            time[ppg_onsets],
            ppg[ppg_onsets],
            marker="x",
            label="PPG Onsets"
        )

    axes[1].set_title("PPG with Fiducials")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{filename}.png"), dpi=300)
    plt.close()
