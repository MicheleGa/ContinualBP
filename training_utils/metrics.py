import os
import sys
folders_to_add = ['data']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import numpy as np
import matplotlib.pyplot as plt
import pyCompare
from sklearn.metrics import r2_score
import torch
from pyampd.ampd import find_peaks
from helpers import numpy_mse_loss, numpy_smooth_l1_loss
from data.preprocessing_utils.signal_processing import compute_sp_dp


def plot_r_squared(gt, pd, name='', save_path='./figs/', figsize=(10, 8)):
    r"""
    Plots the R-squared coefficient for the given ground truth and predicted values.
    
    Parameters
    ------------
        gt:
            Ground truth values (list or numpy array).
        pd:
            Predicted values (list or numpy array).
        name:
            Name of the model or dataset for the plot title.
        save_path:
            Path to save the plot.
        figsize:
            Size of the figure (tuple, default: (10, 8)).
    
    Returns
    ------------
        None: The function saves the plot to the specified path.    
    """
    
    # Sample data (replace with your actual data)
    y_true = np.array(gt)
    y_pred = np.array(pd)

    # Calculate R-squared
    r2 = r2_score(y_true, y_pred)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, label="Data Points", alpha=0.6)
    
    # Add diagonal line (perfect agreement line) using axis limits
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x, high_y)
    ax.plot([low, high], [low, high], ls="--", c="red", alpha=0.8, 
            label="Perfect Agreement (R² = 1.00)", linewidth=2)
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"R-squared ({name}): {r2:.3f}")

    # Add R-squared value to the plot
    ax.text(0.1, 0.9, f"R-squared: {r2:.3f}", transform=ax.transAxes)
    
    # Ensure equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def bhs_grade(differences, thresholds=[5, 10, 15], title=''):
    r"""
    Calculates the BHS grade for a list of blood pressure differences.

    Parameters
    ------------
        differences: 
            A list of differences between blood pressure readings (device under evaluation - reference standard).
        thresholds: 
            A list of thresholds for grading (default: [5, 10, 15] mmHg)
        title: 
            The name of the model being tested.

    Returns
    ------------
        A character representing the BHS grade (A, B, C, or D).
    """

    within_5 = sum(abs(diff) <= thresholds[0] for diff in differences) / len(differences) * 100
    within_10 = sum(abs(diff) <= thresholds[1] for diff in differences) / len(differences) * 100
    within_15 = sum(abs(diff) <= thresholds[2] for diff in differences) / len(differences) * 100
    
    print('BHS grading standard:\n\t\t \u2264 5 \t \u2264 10 \t \u2264 15\n\t Grade A 60% \t 85% \t 95%\n\t Grade B 50% \t 75% \t 90%\n\t Grade C 40% \t 65% \t 85%')
    print(f'{title}:\n\t\t {within_5}% \t {within_10}% \t {within_15}%')
    
    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        return "A"
    elif within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        return "B"
    elif within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        return "C"
    else:
        return "D"
    

def aami_grade(differences, mean_threshold=5, std_dev_threshold=8):
    r"""
    Calculates the AAMI grade for a list of blood pressure differences.

    Parameters
    ------------
        differences: 
            A list of differences between blood pressure readings (device under evaluation - reference standard).
        mean_threshold: 
            Threshold for the mean difference (default: 5 mmHg).
        std_dev_threshold: 
            Threshold for the standard deviation (default: 8 mmHg).

    Returns
    ------------
        A string representing the AAMI grade ("Acceptable", "Potentially Acceptable", or "Unacceptable").
    """

    mean_diff = np.mean(differences)
    std_dev = np.std(differences)

    if abs(mean_diff) <= mean_threshold and std_dev <= std_dev_threshold:
        return "Acceptable"
    elif (abs(mean_diff) <= mean_threshold and std_dev > std_dev_threshold) or \
        (abs(mean_diff) > mean_threshold and std_dev <= std_dev_threshold):
        return "Potentially Acceptable"
    else:
        return "Unacceptable"
    

def call_metric(targets, outputs, config, figure_savepath, plot=False):
    r"""
    Metrics for the Blood Pressure Estimation: plot Bland Altman and r² (optional), SBP/DBP AAAMI and BHS standards scores, and returns the loss (MSE, SmoothL1, etc.).
    
    Parameters
    ------------
    targets (np.ndarray):
        Ground truth values.
    outputs (np.ndarray):
        Predicted values.
    config (dict):
        Model development configuration.
    figure_savepath (str):
        Path to save the figures.
    plot (bool):
        whether to plot metrics or not.
    
    Returns
    ------------
    metrics (dict):
        Dictionary containing the calculated metrics.        
    """
    
    if not os.path.exists(figure_savepath) and plot:
        os.makedirs(figure_savepath)

    if config['sig2sig']:
        # TODO: add MAP, bad reconstructions may not have peaks and valleys, se we replace SBP/DBP with -1 in this case
        outputs_sbp_values = []
        outputs_dbp_values = []
        for el in range(outputs.shape[0]):
            try:
                sbp, dbp, _, _, = compute_sp_dp(outputs[el], fs=config['fs'])
            except:
                sbp, dbp = -1, -1
            outputs_sbp_values.extend([sbp])
            outputs_dbp_values.extend([dbp])

        targets_sbp_values = []
        targets_dbp_values = []
        for el in range(targets.shape[0]):
            try:
                sbp, dbp, _, _, = compute_sp_dp(targets[el], fs=config['fs'])
            except:
                sbp,dbp = -1, -1
            targets_sbp_values.extend([sbp])
            targets_dbp_values.extend([dbp])
        
        sbp_errors = np.array(targets_sbp_values) - np.array(outputs_sbp_values)
        dbp_errors = np.array(targets_dbp_values) - np.array(outputs_dbp_values)
    else:
        sbp_errors = targets[:, 0] - outputs[:, 0]
        dbp_errors = targets[:, 1] - outputs[:, 1]

    if config['criterion'] == 'MSELoss':
        loss = numpy_mse_loss(outputs, targets)
    elif config['criterion'] == 'SmoothL1Loss':
        loss = numpy_smooth_l1_loss(outputs, targets, beta=config['smoothl1loss_beta'])
    else:
        raise ValueError("Invalid criterion ...")

    metrics = {
        'loss': loss,
        'sbp_mae': np.mean(np.abs(sbp_errors)),
        'sbp_mae_std': np.std(np.abs(sbp_errors)),
        'sbp_me': np.mean(sbp_errors),
        'sbp_me_std': np.std(sbp_errors),
        'dbp_mae': np.mean(np.abs(dbp_errors)),
        'dbp_mae_std': np.std(np.abs(dbp_errors)),
        'dbp_me': np.mean(dbp_errors),
        'dbp_me_std': np.std(dbp_errors),
    }
    
    print(f'SBP MAE \u03BC {metrics["sbp_mae"]} \u00B1  {metrics["sbp_mae_std"]} / ME \u03BC {metrics["sbp_me"]} \u00B1 {metrics["sbp_me_std"]}')
    print(f'DBP MAE \u03BC {metrics["dbp_mae"]} \u00B1  {metrics["dbp_mae_std"]} / ME \u03BC {metrics["dbp_me"]} \u00B1 {metrics["dbp_me_std"]}')
    
    if plot:
        # BlandAltman Plot
        pyCompare.blandAltman(targets[:, 0], outputs[:, 0], title=f'Bland Altman Plot (SBP)', savePath=os.path.join(figure_savepath, 'sbp_bland_altman.jpg'))
        pyCompare.blandAltman(targets[:, 1], outputs[:, 1], title=f'Bland Altman Plot (DBP)', savePath=os.path.join(figure_savepath, 'dbp_bland_altman.jpg'))

    if plot:
        # R squared coefficient
        plot_r_squared(targets[:, 0], outputs[:, 0], name='SBP', save_path=os.path.join(figure_savepath, 'sbp_r_squared_coefficient.jpg'))
        plot_r_squared(targets[:, 1], outputs[:, 1], name='DBP', save_path=os.path.join(figure_savepath, 'dbp_r_squared_coefficient.jpg'))

    # BHS standard
    print(f'BHS standard grade for SBP: {bhs_grade(sbp_errors, title=f"SBP")}')
    print(f'BHS standard grade for DBP: {bhs_grade(dbp_errors, title=f"DBP")}')
    
    # AAMI standard
    print(f'AAMI grade for SBP: {aami_grade(sbp_errors)}')
    print(f'AAMI grade for DBP: {aami_grade(dbp_errors)}')
    
    return metrics


class AverageMeter(object):
    r"""
    Computes and stores the average and current value.    
    """
    def __init__(self, name, fmt=':f'):
        r"""
        Parameters
        ------------
            name (str): Name of the metric.
            fmt (str): Format for displaying the value (default: ':f').
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def setup_meter(name, metrics):
    r"""
    Function to set up a meter with specified metrics.

    Parameters
    ------------
        name (str): 
            Prefix for the metric names (e.g., 'train', 'val').
        metrics (list): 
            List of metric names to initialize.

    Returns
    ------------
        dict: 
            A dictionary of AverageMeter objects for the specified metrics.
    """
    meter = {}
    for metric in metrics:
        meter[metric] = AverageMeter(name=f'{name}/{metric}')
    return meter


def get_metric_values(loss, outputs, targets, config):
    r"""
    Function to compute metric values for regression tasks.
    
    Parameters
    ------------
        loss (torch.Tensor): 
            Loss value computed by the criterion.
        outputs (torch.Tensor): 
            Model predictions.
        targets (torch.Tensor): 
            Ground truth values.
        config (dict): 
            Configuration dictionary containing model parameters.
    Returns
    ------------
        dict: 
            A dictionary containing computed metric values.
    """
    if config['sig2sig']:
        # Compute SBP/DBP on the reconstructed ABP waveform as in https://github.com/inventec-ai-center/bp-benchmark
        # TODO: add MAP, bad reconstructions may not have peaks and valleys, se we replace SBP/DBP with -1 in this case        
        targets_sbp_values = []
        targets_dbp_values = []
        for el in range(targets.shape[0]):
            try:
                sbp, dbp, _, _ = compute_sp_dp(targets[el].cpu().numpy(), fs=config['fs'])
            except:
                sbp, dbp = -1, -1
            targets_sbp_values.extend([sbp])
            targets_dbp_values.extend([dbp])
        
        outputs_sbp_values = []
        outputs_dbp_values = []
        for el in range(outputs.shape[0]):
            # May raise error for bad shaped signals
            try:
                sbp, dbp, _, _ = compute_sp_dp(outputs[el].cpu().numpy(), fs=config['fs'])
            except ValueError as e:
                sbp, dbp = -1, -1
            outputs_sbp_values.extend([sbp])
            outputs_dbp_values.extend([dbp])

        targets_sbp_values = torch.tensor(targets_sbp_values, device=targets.device)
        targets_dbp_values = torch.tensor(targets_dbp_values, device=targets.device)        
        outputs_sbp_values = torch.tensor(outputs_sbp_values, device=outputs.device)
        outputs_dbp_values = torch.tensor(outputs_dbp_values, device=outputs.device)
        
        metric_values = {
            'loss': loss.item(),
            'sbp_mae': torch.mean(torch.abs(outputs_sbp_values - targets_sbp_values)).item(),
            'dbp_mae': torch.mean(torch.abs(outputs_dbp_values - targets_dbp_values)).item(),
            'sbp_me': torch.mean(outputs_sbp_values - targets_sbp_values).item(),
            'dbp_me': torch.mean(outputs_dbp_values - targets_dbp_values).item(),
            'sbp_mae_std': torch.std(torch.abs(outputs_sbp_values - targets_sbp_values)).item(),
            'dbp_mae_std': torch.std(torch.abs(outputs_dbp_values - targets_dbp_values)).item(),
            'sbp_me_std': torch.std(outputs_sbp_values - targets_sbp_values).item(),
            'dbp_me_std': torch.std(outputs_dbp_values - targets_dbp_values).item(),
        }
    else:
        metric_values = {
            'loss': loss.item(),
            'sbp_mae': torch.mean(torch.abs(outputs[:, 0] - targets[:, 0])).item(),
            'dbp_mae': torch.mean(torch.abs(outputs[:, 1] - targets[:, 1])).item(),
            'map_mae': torch.mean(torch.abs(outputs[:, 2] - targets[:, 2])).item(),
            'sbp_me': torch.mean(outputs[:, 0] - targets[:, 0]).item(),
            'dbp_me': torch.mean(outputs[:, 1] - targets[:, 1]).item(),
            'map_me': torch.mean(outputs[:, 2] - targets[:, 2]).item(),
            'sbp_mae_std': torch.std(torch.abs(outputs[:, 0] - targets[:, 0])).item(),
            'dbp_mae_std': torch.std(torch.abs(outputs[:, 1] - targets[:, 1])).item(),
            'map_mae_std': torch.std(torch.abs(outputs[:, 2] - targets[:, 2])).item(),
            'sbp_me_std': torch.std(outputs[:, 0] - targets[:, 0]).item(),
            'dbp_me_std': torch.std(outputs[:, 1] - targets[:, 1]).item(),
            'map_me_std': torch.std(outputs[:, 2] - targets[:, 2]).item(),
        }
    return metric_values


def update_meter(meter, metric_values, batch_size):
    f"""
    Function to update a meter with specified metric values.

    Parameters
    ------------
        meter (dict): 
            Dictionary of AverageMeter objects.
        metric_values (dict): 
            Dictionary of metric names and their corresponding values.
        batch_size (int): 
            Batch size for weighting the updates.
    """
    for metric, value in metric_values.items():
        if metric in meter:
            meter[metric].update(value, batch_size)
            

def log_meter_to_tensorboard(writer, meter, epoch, name='val'):
    """
    Function to log metrics to TensorBoard.

    Parameters
    ------------
        writer (SummaryWriter): 
            TensorBoard SummaryWriter object.
        meter (dict): 
            Dictionary of AverageMeter objects.
        epoch (int): 
            Current epoch number.
    """
    for metric_name, avg_meter in meter.items():
        writer.add_scalar(f'{name}/{metric_name}', avg_meter.avg, epoch)
    