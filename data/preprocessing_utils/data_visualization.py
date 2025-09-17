import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
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
        
        
def calculate_dataloaders_mean_std(dataloaders, dataloaders_names, sig2sig, savepath=f'./figs/dataset'):
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
    sig2sig : bool
        Flag indicating whether the dataloader provides raw signals (True) or precomputed annotations (False).
    savepath : str, default './figs/dataset'
        Path to save the generated plots.
        
    Returns
    ------------
    None (saves the plots to the specified savepath)   
    """

    for dataloader, dataloader_name in zip(dataloaders, dataloaders_names):
        sbp_values = []
        dbp_values = []
        map_values = []

        for batch in dataloader:
            _, annotation = batch
            if not sig2sig:
                sbp_values.extend(annotation[:, 0].flatten().tolist())
                dbp_values.extend(annotation[:, 1].flatten().tolist())
                map_values.extend(annotation[:, 2].flatten().tolist())
            else:
                window_abp = annotation.numpy()               
                for el in range(window_abp.shape[0]):
                    sbp, dbp, _, _ = compute_sp_dp(window_abp[el]) # Should not raise an error if the signal is valid
                    map = (2 * dbp + sbp) / 3
                    sbp_values.extend([sbp])
                    dbp_values.extend([dbp])
                    map_values.extend([map])
                    
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
    plt.close() # Close the figure to free up memory