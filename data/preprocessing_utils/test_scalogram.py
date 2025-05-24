import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, welch

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

    up = target_fs
    down = original_fs

    gcd = np.gcd(int(up), int(down))
    up = int(up / gcd)
    down = int(down / gcd)

    resampled_signal = resample_poly(signal, up=up, down=down)

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
        plt.savefig(os.path.join(savepath, f'{title}_original_vs_resampled.jpg'))
        plt.close()

    return resampled_signal


# Example usage
original_fs = 125.0
target_fs = 25
num_samples = 625
signal_duration = 5  # seconds


# Take a PPG sample
time = np.linspace(0, signal_duration, num_samples)
original_signal = np.load('../raw_mimic_iii/ppg/p000735_ppg.npy')[0, :int(original_fs * signal_duration)]

#Plotting original signal.
plt.figure(figsize=(12,4))
plt.plot(time, original_signal, label = 'Original signal')
plt.title('Original signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# Resample the signal
resampled_signal = resample_signal(original_signal, original_fs, target_fs, plot=True, savepath='./')

def scalogram(signal, fs, num_scales=100, wavelet='cmor1.0-1.0', plot=False, title='Scalogram', savepath='./figs'):
    r"""
    Calculates and optionally plots the scalogram of a signal with a constant number of scales.

    Parameters
    ------------
    signal : numpy.ndarray
        Input signal.
    fs : float
        Sampling frequency (Hz).
    num_scales : int, optional
        Number of scales to use for CWT. Default is 100.
    wavelet : str, optional
        Wavelet to use for CWT. Default is 'cmor1.0-1.0'.
    plot : bool, optional
        If True, plots the scalogram. Default is False.
    title : str, optional
        Title for the plot. Default is 'Scalogram'.
    savepath : str, optional
        Path to save the plot. Default is './figs'.

    Returns
    ------------
    scalogram : numpy.ndarray
        Magnitude of the CWT coefficients.
    freqs : numpy.ndarray
        Frequencies corresponding to the scales.
    """

    # Define a fixed frequency range based on the resampled frequency
    low_freq = 0.5  # Hz, minimum frequency to analyze
    high_freq = 12.5 # Hz, maximum frequency to analyze (Nyquist limit)

    dt = 1 / fs
    scales = np.linspace(1 / (high_freq * dt), 1 / (low_freq * dt), num_scales)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
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



scalogram = scalogram(resampled_signal, target_fs, plot=True, savepath='./')

# Print signal and scalogram shapes
print(f"Resampled signal shape: {resampled_signal.shape}")
print(f"Scalogram shape: {scalogram.shape}")