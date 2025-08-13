import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, correlate, welch, resample
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import pywt
from PyEMD import EEMD
from pyampd.ampd import find_peaks


def emd_decompose(signal, num_imfs=4, trials=50):
    r"""
    Decomposes a signal into IMFs using EMD (pyeemd).
    
    Parameters
    ------------
    signal (np.ndarray): 
        The input signal to be decomposed.
    num_imfs (int):
        The number of Intrinsic Mode Functions (IMFs) to return.
    trials (int):
        The number of trials for the EEMD algorithm to improve robustness against noise.    
        
    Returns
    ------------
    imfs (np.ndarray):
        A 2D NumPy array containing the first num_imfs IMFs of the signal.   
    """
    
    emd = EEMD(max_imfs=num_imfs, spline_kind='akima', trials=trials, DTYPE=np.float32) # Initialize the EEMD object
    imfs = emd(signal)  # Compute the EMD
    return imfs[:num_imfs, :]  # Return only the first num_imfs IMFs
    

def get_emd_imfs(signal, num_imfs=4, plot=False, title="EMD", savepath='./figs'):
    r"""
    Decomposes a signal using EMD, plots IMFs, and returns them.
    
    Parameters
    ------------
    signal (np.ndarray): 
        The input signal to be decomposed.
    num_imfs (int):
        The number of Intrinsic Mode Functions (IMFs) to return.
    plot (bool, optional):
        If True, plots the IMFs and the original signal. Defaults to False.
    title (str, optional):
        Title of the plot. Defaults to "EMD".
    savepath (str, optional):
        Path to save the plot. Defaults to './figs'.
        
    Returns
    ------------
    imfs (np.ndarray):
        A 2D NumPy array containing the first num_imfs IMFs of the signal.
    """

    imfs = emd_decompose(signal, num_imfs)
    
    for i in range(num_imfs):
        imfs[i, :] = standardize(imfs[i, :])

    if imfs is not None:
        if plot:
            time = np.arange(len(signal))  # Time vector for plotting

            plt.figure(figsize=(12, 2 * num_imfs))  # Adjust figure size as needed
            plt.suptitle(title, fontsize=14)  # Overall plot title

            for i in range(num_imfs):
                plt.subplot(num_imfs + 1, 1, i + 1)  # Create subplots
                plt.plot(time, imfs[i, :])  # Plot IMF
                plt.title(f"IMF {i+1}")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")

            plt.subplot(num_imfs + 1, 1, num_imfs + 1)  # Create subplots
            plt.plot(time, signal, label="Original Signal")  # Plot original signal
            plt.title("Original Signal")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot params for title
            plt.savefig(os.path.join(savepath, f'{title}.jpg'))
            plt.close()
    else:
        raise ValueError("IMFs are None")
    
    return imfs


def interpolate_nan_pchip(data, plot=False, title='PCHIP Interpolation', savepath='./figs'):
    r"""
    Interpolates NaN values in a 1D NumPy array using PCHIP.

    Args:
        data: The input 1D NumPy array.
        plot (bool, optional): If True, plots the original and interpolated signals. Defaults to False.
        title (str, optional): Title of the plot. Defaults to 'PCHIP Interpolation'.
        savepath (str, optional): Path to save the plot. Defaults to './figs'.

    Returns:
        The interpolated array.
    """

    original_data = data.copy() # Make a copy for plotting

    valid_indices = np.where(~np.isnan(data))[0]
    invalid_indices = np.where(np.isnan(data))[0]

    interpolator = PchipInterpolator(valid_indices, data[valid_indices])
    interpolated_values = interpolator(invalid_indices)

    data[invalid_indices] = interpolated_values

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(original_data, 'o-', label='Original Data (with NaNs)')
        plt.plot(data, 'x-', label='Interpolated Data')
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return data


def create_windows(win_len, fs, n_samp, overlap):
    r"""
    Function from https://github.com/Fabian-Sc85/non-invasive-bp-estimation-using-deep-learning/blob/main/prepare_MIMIC_dataset.py
    Intended to generate the start and end indexes of each window sliding over the signal for preprocessing.

    Parameters
    ------------
    
    win_len: int, 
        length of a window in seconds
    fs: int, 
        sampling frequency of the signal
    n_samp: int,
        length of the signal
    overlap: float,
        length of windows overlapping in seconds

    Returns
    ------------
    
    idx_start: np.array,
        array of window starting indexes in the signal
    idx_stop: np.array,
        array of window ending indexes in the signal.  
    """
    
    win_len = int(win_len * fs)
    overlap = int(overlap * fs)
    n_samp = n_samp - win_len + 1

    idx_start = np.round(np.arange(0, n_samp, win_len - overlap)).astype(int)
    idx_stop = np.round(idx_start + win_len - 1)

    return idx_start, idx_stop

    
def compute_sp_dp(sig, fs=125, verbose=False):
    r"""
    Computes the systolic peak (SP) and diastolic peak (DP) of a signal (e.g. ABP).
    Source: https://github.com/inventec-ai-center/bp-benchmark/blob/main/code/train/core/utils.py#L174
    
    Parameters
    ----------
    sig : np.ndarray
        The input signal as a NumPy array.
    fs : int, optional
        The sampling frequency of the signal in Hz. Defaults to 125.
    pk_th : float, optional 
        The threshold for peak detection relative to the mean amplitude of valleys. Defaults to 0.6.
    
    Returns
    -------
    tuple:
        A tuple containing the median values of the systolic peak and diastolic peak, 
        two flags indicating if the peaks were adjusted, and the indices of the peaks and valleys.    
    """
    sig = sig.astype(np.float64) # Extremely important for Pyampd !!!
    
    try:
        # Can raise error for very malformed waveforms
        peaks = find_peaks(sig, fs)
        valleys = find_peaks(-sig, fs)
    except:
        return -1, -1, [], []
    
    if len(peaks) == 0 or len(valleys) == 0:
        if verbose:
            print('Error during peaks/valleys processing: no peaks or valleys found in the signal.')
        return -1, -1, [], []
    
    ### Remove first or last if equal to 0 or len(sig)-1
    if peaks[0] == 0:
        peaks = peaks[1:]    
        if len(peaks) == 0:
            if verbose:
                print('Error during peaks/valleys processing: no peaks found in the signal.')
            return -1, -1, [], []
            
    if valleys[0] == 0:
        valleys = valleys[1:]
        if len(valleys) == 0:
            if verbose:
                print('Error during peaks/valleys processing: no valleys found in the signal.')
            return -1, -1, [], []
    
    if peaks[-1] == len(sig)-1:
        peaks = peaks[:-1]
        if len(peaks) == 0:
            if verbose:
                print('Error during peaks/valleys processing: no peaks found in the signal.')
            return -1, -1, [], []
    
    if valleys[-1] == len(sig)-1:
        valleys = valleys[:-1]
        if len(valleys) == 0:
            if verbose:
                print('Error during peaks/valleys processing: no valleys found in the signal.')
            return -1, -1, [], []
        
    if len(peaks) == 0 or len(valleys) == 0:
        if verbose:
            print('Error during peaks/valleys processing: no peaks or valleys found in the signal.')
        return -1, -1, [], []
    else:
        sig = sig.astype(np.float32)  # Extremely important for Pyampd !!! This is why we are casting here
        return np.median(sig[peaks]), np.median(sig[valleys]), peaks, valleys


def butter_lowpass_filter(data, lowcut, fs, order):
    r""" 
    Butterworth band-pass filter.
    Source: https://github.com/inventec-ai-center/bp-benchmark/blob/main/code/train/core/utils.py#L174
    
    Parameters
    ----------
    data : array
        Signal to be filtered.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    fs : float
        Sampling rate.
    order: int
        Filter's order.

    Returns
    -------
    array
        Signal filtered with butterworth algorithm.
    """  
    nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
    lowcut = lowcut / nyq  # Normalize
    #highcut = highcut / nyq
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    b, a = butter(order, lowcut, btype='low', analog=False)
    return filtfilt(b, a, data)
    

def align_pair(abp, raw_ppg, windowing_time, fs):
    """
    Align ABP and PPG signal passed as parameters using the maximum cross-correlation.
    Only PPG is shifted to align with ABP. The shift is limited to a second as maximum.
    Source: https://github.com/inventec-ai-center/bp-benchmark/blob/main/code/process/core/lib/preprocessing.py

    Parameters
    ----------
    abp : array
        ABP signal waveform
    raw_ppg : array
        PPG signal waveform
    windowing_time: int
        Length of the signals in seconds
    fs: int
        Frequency sampling rate (Hz)
    
    Returns
    -------
    array
        Aligned ABP signal waveform
    array
        Aligned PPG signal waveform
    Int
        Number of samples shifted

    """

    window_size = fs * windowing_time # original segment length
    extract_size = fs * (windowing_time-1)

    cross_correlation = correlate(abp, raw_ppg)
    shift = np.argmax(cross_correlation[extract_size:window_size]) #shift must happened within 1s
    shift += extract_size
    start = np.abs(shift-window_size)

    a_abp = abp[:extract_size]
    a_rppg = raw_ppg[start:start+extract_size]

    return a_abp, a_rppg, shift-window_size


def autocorrelation_filter(ppg_signal, threshold=0.7, verbose=False, plot=False, title='Autocorrelation Filter', savepath='./figs'):
    """
    Applies an autocorrelation filter to discard invalid PPG signals.

    Parameters
    ----------
    ppg_signal (np.ndarray): 
        The PPG signal as a NumPy array.
    threshold (float): 
        The threshold for maximum autocorrelation.
    plot (bool, optional): 
        If True, plots the autocorrelation. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Autocorrelation Filter'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    -------
    bool: True if the signal is valid, False otherwise.
    """

    # Calculate autocorrelation
    try:
        autocorr = np.correlate(ppg_signal, ppg_signal, mode='full')
        autocorr = autocorr[len(ppg_signal) - 1:] / np.max(autocorr) # Normalize and keep only the positive lags.
    
        # Find the maximum autocorrelation (excluding the first value)
        peaks = find_peaks(autocorr)[1:-1]
        max_autocorr = np.max(autocorr[peaks])
        is_valid = max_autocorr >= threshold
    except:
        if verbose:
            print('Error during autocorrelation filtering: empty autocorrelation or no peaks detected in the autocorrelation')
        return False

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(autocorr, label='Autocorrelation')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        plt.axhline(y=max_autocorr, color='g', linestyle='--', label=f'Max Autocorr: {max_autocorr:.3f}')
        plt.plot(peaks, autocorr[peaks], "x", color='purple', label='Peaks') #added peak plotting.

        plt.title(title)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()

        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return is_valid
    
        
def wavelet_denoise(signal, type='db8', fs=125, low_cut=0.5, high_cut=50, ecg=False, plot=False, title='DWT', savepath='./figs'):
    r"""
    Denoises a PPG signal using DWT.

    Parameters
    ------------
    
    signal (torch.Tensor): 
        Input PPG signal of shape (sequence_length,).

    Returns
    ------------
    
    torch.Tensor: Denoised PPG signal of the same shape as the input.
    """

    sequence_length = len(signal)
    denoised_signal = np.zeros_like(signal)
    
    max_level = pywt.dwt_max_level(sequence_length, type)

    # Multilevel DWT
    coeffs = pywt.wavedec(signal, type, level=max_level)
        
    if plot:
        for i, ci in enumerate(coeffs):
            plt.imshow(ci.reshape(1, -1), extent=[0, len(ci), i + 0.5, i + 1.5], cmap='inferno', aspect='auto', interpolation='nearest')
        plt.title(f'{title} - Noisy Signal Wavelet Coeffs.')
        plt.ylim(0.5, len(coeffs) + 0.5) 
        plt.yticks(range(1, len(coeffs) + 1), [f'cD{len(coeffs) - i - 1}' for i in range(len(coeffs))])
        plt.savefig(os.path.join(savepath, f'{title}_wav_coeffs.jpg'))
        plt.close()
        
    # Zeroing coefficients as in https://www.sciencedirect.com/science/article/pii/S0933365719309674?via%3Dihub
    # and as in https://www.sciencedirect.com/science/article/pii/S0957417424006754
    # Look at Pywavelets documentation for the order among coefficients: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html#multilevel-decomposition-using-wavedec
    
    # Zeroing coefficients outside the low_cut to high_cut Hz range
    filtered_coeffs = []
    for i, coeff in enumerate(coeffs):
        # Calculate the frequency band for the current level
        freq_band = (fs / (2 ** (i + 1)), fs / (2 ** i))
        if freq_band[1] < low_cut or freq_band[0] > high_cut:  # Zero out coefficients outside the 0.5 to 50 Hz range
            filtered_coeffs.append(np.zeros_like(coeff))
        else:
            filtered_coeffs.append(coeff)
            
    if plot:
        for i, ci in enumerate(filtered_coeffs):
            plt.imshow(ci.reshape(1, -1), extent=[0, len(ci), i + 0.5, i + 1.5], cmap='inferno', aspect='auto', interpolation='nearest')
        plt.title(f'{title} - Filtered Wavelet Coeffs.')
        plt.ylim(0.5, len(filtered_coeffs) + 0.5) 
        plt.yticks(range(1, len(filtered_coeffs) + 1), [f'cD{len(filtered_coeffs) - i - 1}' for i in range(len(filtered_coeffs))])
        plt.savefig(os.path.join(savepath, f'{title}_filtered_wav_coeffs.jpg'))
        plt.close()

    # Reconstruction
    denoised_signal = pywt.waverec(filtered_coeffs, type)

    # Ensure the reconstructed signal has the same length as the original
    denoised_signal = denoised_signal[:sequence_length]
    
    if plot:
        # Plotting for debugging
        fs = 125
        time = np.linspace(0, sequence_length / fs, sequence_length) 

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, signal, label='Noisy')
        plt.title('Noisy Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, denoised_signal, label='Denoised', color='orange')
        plt.title('Denoised Signal')
        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{title}_noisy_vs_denoised.jpg'))
        plt.close()
        
    return denoised_signal


def resample_signal(signal, original_fs, target_fs, plot=False, title='Resample', savepath='./figs'):
    r"""
    Resamples a signal from original_fs to target_fs.

    Parameters
    ------------
    signal : numpy.ndarray
        Input signal.
    original_fs : float
        Original sampling frequency (Hz).
    target_fs : float
        Target sampling frequency (Hz).
    plot : bool, optional
        If True, plots the original and resampled signals. Default is False.
    title : str, optional
        Title for the plot. Default is 'Resample'.
    savepath : str, optional
        Path to save the plot. Default is './figs'.

    Returns
    ------------
    resampled_signal : numpy.ndarray
        Resampled signal.
    """
    
    # Calculate the number of samples for the resampled signal
    num_samples = int(len(signal) * (target_fs / original_fs))

    # Resample the signal using the Fourier method
    resampled_signal = resample(signal, num=num_samples)

    if plot:
        # Plotting for debugging
        time = np.linspace(0, len(signal) / original_fs, len(signal))
        resampled_time = np.linspace(0, len(resampled_signal) / target_fs, len(resampled_signal))

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(resampled_time, resampled_signal, label='Resampled Signal', color='orange')
        plt.title('Resampled Signal')
        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        
        # Ensure the savepath directory exists
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f'{title}_original_vs_resampled.jpg'))
        plt.close()

    return resampled_signal


def scalogram(signal, fs, high_freq, low_freq, num_scales, wavelet='cmor1.0-1.0', plot=False, title='Scalogram', savepath='./figs'):
    r"""
    Calculates and optionally plots the scalogram of a signal.

    Parameters
    ------------
    signal : array_like
        The input signal.
    fs : float
        The sampling frequency of the signal in Hz.
    high_freq : float
        The highest frequency of interest in Hz.
    low_freq : float
        The lowest frequency of interest in Hz.
    wavelet : str, optional
        The wavelet function to use for the Continuous Wavelet Transform (CWT). Default is 'cmor1.0-1.0'.
    plot : bool, optional
        If True, plots the scalogram. Default is False.
    title : str, optional
        The title of the plot. Default is 'Scalogram'.
    savepath : str, optional
        The path to save the plot. Default is './figs'.

    Returns
    ------------
    scalogram : numpy.ndarray
        The scalogram of the input signal, a 2D array representing the magnitude of the CWT coefficients.
    """

    sampling_period = 1 / fs
    dt = sampling_period #sampling period

    # Calculate scales directly based on frequency limits.
    max_scale = 1 / (low_freq * dt)
    min_scale = 1 / (high_freq * dt)
    scales = np.geomspace(min_scale, max_scale, num_scales)

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
    scalogram = np.abs(coeffs)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.imshow(scalogram, extent=[0, len(signal) / fs, freqs[-1], freqs[0]], aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return scalogram


def butterworth_filtering(signal, fs=125, level=4, low_freq=0.5, high_freq=8, plot=False, title='Butterworth', savepath='./figs'):
    r"""
    Denoises a signal using a 4-th order bandpass butterworth filter.
    Sources for the frequency lower/upper cutoff points:
        1 - https://pmc.ncbi.nlm.nih.gov/articles/PMC5844675/
        2 - https://pmc.ncbi.nlm.nih.gov/articles/PMC5651164/
        3 - https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/1475-925X-8-13
        4 - https://www.nature.com/articles/sdata201876
        5 - https://dl.acm.org/doi/10.5555/3578948.3578955

    Parameters
    ------------
    signal (numpy.array): 
        Input signal of shape (sequence_length,).
    fs (int, optional): 
        Sampling frequency in Hz. Default is 125.
    level (int, optional): 
        Order of the Butterworth filter. Default is 4.
    low_freq (float, optional): 
        Lower cutoff frequency in Hz. Default is 0.5.
    high_freq (float, optional): 
        Upper cutoff frequency in Hz. Default is 8.
    plot (bool, optional): 
        If True, plots the original and denoised signals. Default is False.
    title (str, optional): 
        Title prefix for saved plots. Default is 'Butterworth'.
    savepath (str, optional):       
        Directory path to save plots. Default is './figs'.
        
    Returns
    ------------
    numpy.array: Denoised PPG signal of the same shape as the input.
    """

    sequence_length = len(signal)
    
    # Bandpass filter
    sos = butter(level, [low_freq, high_freq], btype='bp', analog=False, output='sos', fs=fs)
    
    # Apply the filter using filtfilt for zero-phase filtering
    denoised_signal = sosfiltfilt(sos, signal)

    if plot:
        # Plotting for debugging
        time = np.linspace(0, sequence_length / fs, sequence_length) 

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, signal, label='Noisy Signal')
        plt.title('Noisy Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, denoised_signal, label='Denoised Signal', color='orange')
        plt.title('Denoised Signal')
        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{title}_noisy_vs_denoised.jpg'))
        plt.close()
        
    return denoised_signal


def high_freq_butterworth(signal, fs=125, order=4, cutoff_freq=35.0, plot=False, title='Signal_LP', savepath='./figs'):
    r"""
    Removes high-frequency noise components above 35 Hz from a signal using a 4th-order
    Butterworth low-pass filter.

    This function is designed to specifically target and attenuate noise (e.g., industrial
    frequency interference, muscle artifacts) that typically occurs at frequencies
    significantly higher than the physiologically relevant components of signals
    like blood pressure or PPG.

    Parameters
    ----------
    signal (np.ndarray): 
        Input signal of shape (sequence_length,).
    fs (int, optional): 
        Sampling frequency in Hz. Default is 125.
    order (int, optional): 
        Order of the Butterworth filter. Default is 4.
    cutoff_freq (float, optional): 
        The cutoff frequency for the low-pass filter in Hz. Frequencies above this
        will be attenuated. Default is 35.0 Hz. This parameter is fixed here to 35.0 Hz
        to meet the specific requirement.
    plot (bool, optional): 
        If True, plots the original and denoised signals. Default is False.
    title (str, optional): 
        Title prefix for saved plots. Default is 'Signal_LP_35Hz'.
    savepath (str, optional):       
        Directory path to save plots. Default is './figs'.
        
    Returns
    ------------
    np.ndarray: Denoised signal with high-frequency noise above 35 Hz removed.
    """
    signal = np.array(signal) # Ensure signal is a NumPy array
    
    sequence_length = len(signal)
    
    # Normalize cutoff frequency to Nyquist frequency (fs/2)
    nyquist = fs / 2
    
    # Validate cutoff frequency
    if not (0 < cutoff_freq <= nyquist):
        raise ValueError(f"Cutoff frequency ({cutoff_freq} Hz) must be greater than 0 and less than or equal to Nyquist frequency ({nyquist} Hz).")
    
    # If you normnalize here you do not pass the argument fs in the butter function
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design the Butterworth low-pass filter
    # btype='low' specifies a low-pass filter
    # output='sos' (second-order sections) is used for numerical stability
    sos = butter(order, normalized_cutoff, btype='low', analog=False, output='sos')
    
    # Apply the filter using filtfilt for zero-phase filtering
    denoised_signal = sosfiltfilt(sos, signal)

    if plot:
        # Ensure savepath exists
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        # Plotting for visualization
        time = np.linspace(0, sequence_length / fs, sequence_length) 

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, signal, label='Original Signal (with high-freq noise)')
        plt.title('Original Signal')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, denoised_signal, label=f'Low-Pass Filtered Signal (<{cutoff_freq} Hz)', color='green')
        plt.title(f'Denoised Signal (Noise Above {cutoff_freq} Hz Removed)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{title}_original_vs_lowpass.jpg'))
        plt.close()
        
    return denoised_signal


def hampel_filtering(signal, fs=125, window_size=100, n_sigmas=3.0, plot=False, title='Hampel', savepath='./figs'):
    r"""
    Denoises an arterial BP signal using a hampel filter (sigma 3) as in https://bmcmedinformdecismak.biomedcentral.com/counter/pdf/10.1186/s12911-023-02215-2.pdf.

    Parameters
    ------------
    
    signal (torch.Tensor): 
        Input arterial BP signal of shape (sequence_length,).

    Returns
    ------------
    
    torch.Tensor: Denoised arterial BP signal of the same shape as the input.
    """

    signal = np.array(signal) # Ensure it's a numpy array.
    
    n = len(signal)
    denoised_signal = np.copy(signal)
    k = 1.4826  # Scale factor for robust standard deviation estimation

    for i in range(window_size, n - window_size):
        window = signal[i - window_size:i + window_size + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))  # Median Absolute Deviation
        threshold = n_sigmas * k * mad

        if np.abs(signal[i] - median) > threshold:
            denoised_signal[i] = median
            
    if plot:
        # Plotting for debugging
        time = np.linspace(0, n / fs, n) 

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, signal, label='Noisy ABP')
        plt.title('Noisy Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, denoised_signal, label='Denoised ABP', color='orange')
        plt.title('Denoised Signal')
        plt.legend()
        plt.xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(savepath, f'{title}_noisy_vs_denoised.jpg'))
        plt.close()
        
    return denoised_signal


def skewness_check(window_sig, fs=125, window_s=8, subsegment_s=1):
    r"""
    Calculates the skewness of subsegments within sliding windows and check if they are > 0.
    
    Parameters
    ------------
    window_sig (np.ndarray): 
        The 1D input signal as a NumPy array.
    fs (int): 
        the signal frequency
    window_s (int): 
        the window length in seconds
    subsegment_s (int): 
        the subegment in the window length in seconds

    Returns
    ------------
    np.ndarray: 
        returns False if the input winodw of the signal values have skewness < 0, otherwise True
    """
    
    window_size = window_s * fs
    subsegment_size = subsegment_s * fs
    subsegments = np.split(window_sig, window_size // subsegment_size)  # Split into subsegments

    skewness_subsegments = []
    for subsegment in subsegments:
        mean = np.mean(subsegment)
        std = np.std(subsegment) + np.finfo(np.float32).smallest_normal
        n = len(subsegment)
        sum_cubed_diff = np.sum((subsegment - mean)**3)
        skewness = (1/n) * (sum_cubed_diff / std)
        skewness_subsegments.append(skewness)
    
    return True if all(s >= 0.0 for s in skewness_subsegments) else False


def rescale_to_unit(signal, plot=False, title='Rescaled [0,1]', savepath='./figs'):
    r"""
    Rescales a signal to the range [0, 1].

    Parameters
    ------------
    signal (np.ndarray): 
        The input signal as a NumPy array.
    plot (bool, optional): 
        If True, plots the original and rescaled signals. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Rescaled [0,1]'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    np.ndarray: 
        The rescaled signal. Returns the original signal if the 
        range is zero (to avoid division by zero).
    """

    min_val = np.min(signal)
    max_val = np.max(signal)
    range_val = max_val - min_val + np.finfo(np.float32).eps  # Prevent division by zero

    rescaled_signal = (signal - min_val) / range_val

    if plot:
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(rescaled_signal, label='Rescaled Signal [0,1]', color='green')
        plt.title('Rescaled Signal [0,1]')
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return rescaled_signal


def standardize(signal, plot=False, title='Z-Score', savepath='./figs'):
    r"""
    Standardizes a signal to have zero mean and unit variance (Z-score normalization).

    Parameters
    ------------
    signal (np.ndarray): 
        The input signal as a NumPy array.
    plot (bool, optional): 
        If True, plots the original and standardized signals. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Standardized Signal'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    np.ndarray: 
        The standardized signal.  Returns the original signal if the 
        standard deviation is zero (to avoid division by zero).
    """

    mean = np.mean(signal)
    std = np.std(signal) + np.finfo(np.float32).smallest_normal

    standardized_signal = (signal - mean) / std

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(standardized_signal, label='Standardized Signal', color='orange')
        plt.title('Standardized Signal')
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return standardized_signal


def ema_normalization(signal, alpha=0.00796, plot=False, title='EMA Normalization', savepath='./figs'):
    r"""
    Applies Exponential Moving Average (EMA) normalization to a signal.

    Parameters
    ------------
    signal (np.ndarray): 
        The input signal as a NumPy array.
    alpha (float, optional): 
        The EMA smoothing factor. Default is 0.00796 (corresponding to ~2s of effective history for 125 Hz signals).
    plot (bool, optional): 
        If True, plots the original and EMA-normalized signals. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'EMA Normalization'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    np.ndarray: 
        The EMA-normalized signal. Returns zeros if the signal has zero variance (handled safely).
    """

    eps = np.finfo(np.float32).eps  # To avoid division by zero

    mu = 0.0
    sigma = 0.0
    norm_signal = np.zeros_like(signal, dtype=np.float32)

    for t in range(len(signal)):
        x = signal[t]
        mu = alpha * x + (1 - alpha) * mu
        sigma = alpha * abs(x - mu) + (1 - alpha) * sigma
        norm_signal[t] = (x - mu) / (sigma + eps)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(norm_signal, label='EMA Normalized Signal', color='orange')
        plt.title('EMA Normalized Signal')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return norm_signal


def percentile_normalize(signal, percentile=95, plot=False, title='Percentile Normalized', savepath='./figs'):
    r"""
    Normalizes a signal by dividing it by the specified percentile of its 
    absolute amplitude. This scaling is robust to outliers and maintains
    relative waveform shapes.

    Parameters
    ------------
    signal (np.ndarray): 
        The input signal as a NumPy array.
    percentile (float, optional): 
        The percentile of the absolute amplitude used for normalization. 
        Defaults to 95.
    plot (bool, optional): 
        If True, plots the original and normalized signals. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Percentile Normalized'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    np.ndarray: 
        The normalized signal. If the computed percentile is zero, returns the 
        original signal (to avoid division by zero).
    """

    scale_val = np.percentile(np.abs(signal), percentile)
    scale_val = scale_val + np.finfo(np.float32).eps  # Prevent division by zero

    normalized_signal = signal / scale_val

    if plot:
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(normalized_signal, label=f'Normalized by {percentile}th Percentile', color='green')
        plt.title(f'Percentile Normalized Signal ({percentile}th)')
        plt.legend()
        
        plt.tight_layout()
        
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return normalized_signal


def average_smoothing(signal, window_size=3, plot=False, title='AVGSmoothing', savepath='./figs'):
    r"""
    Average smoothing of a signal.

    Parameters
    ------------
    signal (np.ndarray): 
        The input signal as a NumPy array.
    window_size (int): 
        The size of the window to consider to average.
    plot (bool, optional): 
        If True, plots the original and smoothed signals. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Smoothed Signal'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    np.ndarray: 
        The smoothed signal. 
    """
    smoothed_signal = np.convolve(signal, np.ones(window_size), 'same') / window_size

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(smoothed_signal, label='Smoothed Signal', color='orange')
        plt.title('Smoothed Signal')
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return smoothed_signal


def calculate_differences(signal, plot=False, title='VPG-APG', savepath='./figs'):
    r"""
    Calculates the first-order and second-order differences of a preprocessed PPG signal.

    Parameters
    ------------
    signal (np.ndarray): 
        The preprocessed PPG signal as a NumPy array.
    plot (bool, optional): 
        If True, plots the original signal, first-order difference, and second-order difference. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Difference Signals'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    tuple: 
        A tuple containing the first-order and second-order differences of the signal,
        both as NumPy arrays.
    """
     
    # First-Order Difference
    first_diff = np.diff(signal)
    first_diff_padded = np.pad(first_diff, (0, len(signal) - len(first_diff)), 'constant') # Pad with zeros

    # Second-Order Difference
    second_diff = np.diff(first_diff)
    second_diff_padded = np.pad(second_diff, (0, len(signal) - len(second_diff)), 'constant') # Pad with zeros

    if plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(first_diff_padded, label='First-Order Difference', color='orange')
        plt.title('First-Order Difference')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(second_diff_padded, label='Second-Order Difference', color='green')
        plt.title('Second-Order Difference')
        plt.legend()

        plt.tight_layout()
        
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return first_diff_padded, second_diff_padded


def harmonic_filtering(signal, fs=125, order=2, plot=False, title='Harmonic', savepath='./figs'):
    r"""
    Apply an adaptive Butterworth filter where low/up cut-off frequencies are decided after the patient first harmonic.
    Source: https://dl.acm.org/doi/10.5555/3578948.3578955

    Parameters
    ------------
    signal (np.ndarray): 
        The PPG signal as a NumPy array.
    fs (int, optional):
        Sampling frequency. Defaults to 125.
    order (int, optional):
        Order of the Butterworth filter. Defaults to 2.
    plot (bool, optional): 
        If True, plots the original and filtered signals. Defaults to False.
    title (str, optional): 
        Title of the plot. Defaults to 'Harmonic Filtered Signal'.
    savepath (str, optional): 
        Path to save the plot. Defaults to './figs'.

    Returns
    ------------
    np.ndarray: 
        The filtered signal. 
    """
    
    # Harmonic filtering from CardioID
    f, Pxx_den = welch(signal, fs=fs, window='hann')

    # Find the peak frequency
    first_harmonic = f[np.argmax(Pxx_den)]

    # Adaptive Butterworth filtering
    lowcut = 2 * first_harmonic
    highcut = 5.5 * first_harmonic

    sos_ppg = butter(order,
                    [lowcut, highcut],
                    btype='bp',
                    analog=False,
                    output='sos',
                    fs=fs)
    filtered_signal = sosfiltfilt(sos_ppg, signal)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Original Signal')
        plt.title('Original Signal')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(filtered_signal, label='Harmonic Filtered Signal', color='orange')
        plt.title('Harmonic Filtered Signal')
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(savepath, f'{title}.jpg'))
        plt.close()

    return filtered_signal
