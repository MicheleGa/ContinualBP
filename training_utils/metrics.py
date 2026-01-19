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


def numpy_mse_loss(outputs, targets):
    r"""
    Calculates the Mean Squared Error (MSE) between predictions and ground truth labels using NumPy.
    
    Parameters
    ------------
    outputs (numpy.ndarray): 
        The predicted values from the model.
        
    targets (numpy.ndarray): 
        The actual ground truth values.
        
    Returns
    ------------
    output param 1:
        The computed MSE loss as a floating-point scalar.   
    """
    return np.mean((outputs - targets)**2)


def numpy_smooth_l1_loss(outputs, targets, beta=1.0):
    r"""
    Calculates the Smooth L1 loss (Huber loss variation) between predictions and targets using NumPy.
    
    Parameters
    ------------
    outputs (numpy.ndarray): 
        The predicted values from the model.
        
    targets (numpy.ndarray): 
        The actual ground truth values.
        
    beta (float): 
        The threshold at which the loss transitions from quadratic to linear.
        
    Returns
    ------------
    output param 1:
        The computed Smooth L1 loss as a floating-point scalar.   
    """
    absolute_error = np.abs(outputs - targets)
    quadratic_error = 0.5 * absolute_error**2 / beta
    linear_error = absolute_error - 0.5 * beta
    return np.mean(np.where(absolute_error < beta, quadratic_error, linear_error))


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
    else:
        return "Unacceptable"
    

def call_metric(targets, outputs, config, figure_savepath, plot=False):
    r"""
    Evaluates blood pressure estimation performance by calculating regression losses, 
    clinical standards (BHS/AAMI), and generating statistical validation plots.
    
    Parameters
    ------------
    targets (numpy.ndarray): 
        Ground truth blood pressure values with shape (N, 2) for SBP and DBP.
        
    outputs (numpy.ndarray): 
        Predicted blood pressure values from the model with shape (N, 2).
        
    config (dict): 
        Configuration dictionary containing the 'criterion' type and loss-specific hyperparameters.
        
    figure_savepath (str): 
        The directory path where generated Bland-Altman and R² plots will be saved.
        
    plot (bool): 
        Flag to determine whether to generate and save visualization plots.
        
    Returns
    ------------
    output param 1:
        A dictionary containing calculated loss, Mean Absolute Error (MAE), and Mean Error (ME) for both SBP and DBP.   
    """
    if not os.path.exists(figure_savepath) and plot:
        os.makedirs(figure_savepath)

    outputs_sbp_values, outputs_dbp_values = outputs[:, 0], outputs[:, 1]
    targets_sbp_values, targets_dbp_values = targets[:, 0], targets[:, 1]

    sbp_errors = targets_sbp_values - outputs_sbp_values
    dbp_errors = targets_dbp_values - outputs_dbp_values

    # ---- LOSS ----
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

    print(f'SBP MAE μ {metrics["sbp_mae"]} ± {metrics["sbp_mae_std"]} / ME μ {metrics["sbp_me"]} ± {metrics["sbp_me_std"]}')
    print(f'DBP MAE μ {metrics["dbp_mae"]} ± {metrics["dbp_mae_std"]} / ME μ {metrics["dbp_me"]} ± {metrics["dbp_me_std"]}')

    # ---- PLOTS ----
    if plot:
        # Bland-Altman
        pyCompare.blandAltman(
            targets_sbp_values, outputs_sbp_values,
            title='Bland Altman Plot (SBP)',
            savePath=os.path.join(figure_savepath, 'sbp_bland_altman.jpg')
        )
        pyCompare.blandAltman(
            targets_dbp_values, outputs_dbp_values,
            title='Bland Altman Plot (DBP)',
            savePath=os.path.join(figure_savepath, 'dbp_bland_altman.jpg')
        )

        # R²
        plot_r_squared(
            targets_sbp_values, outputs_sbp_values,
            name='SBP',
            save_path=os.path.join(figure_savepath, 'sbp_r_squared_coefficient.jpg')
        )
        plot_r_squared(
            targets_dbp_values, outputs_dbp_values,
            name='DBP',
            save_path=os.path.join(figure_savepath, 'dbp_r_squared_coefficient.jpg')
        )

    # ---- STANDARDS ----
    print(f'BHS standard grade for SBP: {bhs_grade(sbp_errors, title="SBP")}')
    print(f'BHS standard grade for DBP: {bhs_grade(dbp_errors, title="DBP")}')
    print(f'AAMI grade for SBP: {aami_grade(sbp_errors)}')
    print(f'AAMI grade for DBP: {aami_grade(dbp_errors)}')

    return metrics


def compute_transfer_metrics_from_matrix(errors_matrix):
    r"""
    Compute continual learning metrics (AA, BWT) for error-based metrics (lower = better).

    Parameters
    ----------
    errors_matrix : np.ndarray, shape (T, T)
        errors_matrix[t, i] = MAE on test set of block i after finishing adaptation on block t.

    Returns
    -------
    dict : {'AA': float, 'BWT': float}
    """
    T = errors_matrix.shape[0]
    # Average final MAE (lower = better)
    final_row = errors_matrix[T - 1, :]
    AA = float(np.nanmean(final_row))

    # Backward Transfer (ΔMAE): positive = forgetting, negative = improvement
    diag = np.array([errors_matrix[i, i] for i in range(T)])
    bwt_vals = []
    for i in range(T - 1):
        final_err = errors_matrix[T - 1, i]
        init_err = diag[i]
        if not np.isnan(final_err) and not np.isnan(init_err):
            bwt_vals.append(final_err - init_err)
    BWT = float(np.mean(bwt_vals)) if len(bwt_vals) > 0 else None

    return {"AA": AA, "BWT": BWT}