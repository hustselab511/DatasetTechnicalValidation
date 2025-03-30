import pywt
import numpy as np
from scipy.signal import butter, hilbert, savgol_filter, filtfilt


def low_bandpass_filter(data, cutoff=5.0, fs=125):
    """
    Apply zero-phase low-pass filter to process the signal, suitable for noise suppression and baseline correction.

    Parameters.
        data (np.ndarray): input one-dimensional time series signal (shape: [N,]).
        cutoff (float): cutoff frequency (in Hz) above which the components will be attenuated.
        fs (int): sampling frequency (in Hz)

    Returns.
        np.ndarray: the filtered signal, with the same shape as the input.

    Processing Steps:
        1. calculate the effective Nyquist frequency (0.8 times the sampling frequency).
        2. Normalize the cutoff frequency to the range [0, 1].
        3. generate the second-order Butterworth low-pass filter coefficients.
        4. use forward-reverse filtering (filtfilt) to achieve zero-phase delay processing.
    """
    # Effective Nyquist frequency
    nyquist_freq = 0.8 * fs
    # Normalized cutoff frequency
    low = cutoff / nyquist_freq
    # Generate filter coefficients
    [b, a] = butter(2, low, btype='lowpass')
    smooth_data = filtfilt(b, a, data)
    return smooth_data


def extract_temporal_domain_envelope(signal, fs=125, cutoff=5):
    """
    Extract normalized temporal envelope using Hilbert transform and lowpass filtering.

    Parameters:
        signal (np.ndarray): Input 1D time-domain signal
        fs (int): Sampling frequency in Hz (default: 125)
        cutoff (float): Cutoff frequency for envelope smoothing in Hz (default: 5)

    Returns:
        np.ndarray: Normalized envelope signal in range [0,1]

    Processing Steps:
        1. Analytic Signal Generation: Hilbert transform implementation
        2. Envelope Smoothing: Zero-phase lowpass filtering
        3. Dynamic Range Normalization: Adaptive signal scaling
    """
    # Hilbert transform to obtain analytic signal
    analytic_signal = hilbert(signal)  # Complex signal representation
    amplitude_envelope = np.abs(analytic_signal)  # Instantaneous amplitude extraction

    # Post-processing: Smoothing with lowpass filter
    smooth_envelope = low_bandpass_filter(data=amplitude_envelope, cutoff=cutoff, fs=fs)

    # Dynamic range normalization
    return (smooth_envelope - np.min(smooth_envelope)) / (smooth_envelope.max() - np.min(smooth_envelope))


def extract_spectral_domain_envelope(signal, fs=125, wavelet='morl', total_scal=256, moving_window_sec=0.15,
                                     sg_window_sec=0.25, freq_band=None):
    """
    Extraction of frequency domain energy envelope based on continuous wavelet transform (CWT) and Savitzky-Golay filter.

    Parameters.
        signal (np.ndarray): input one-dimensional time-domain signal (shape: [N,])
        fs (int, optional): sampling frequency (in Hz).
        wavelet (str, optional): wavelet base type, supports pywt's built-in wavelets.
        total_scal (int, optional): total number of scalar parameters, determines frequency resolution.
        moving_window_sec (float, optional): moving average window length (in seconds) for initial smoothing.
        sg_window_sec (float, optional): SG filter window length (in seconds), to be converted to odd length.
        freq_band (list, optional): Target band range [low, high] (in Hz).

    Returns:
        np.ndarray: normalized frequency domain energy envelope in the range [0,1].

    Processing Steps:
        1. Initialize wavelet scales based on central frequency.
        2. Perform multiscale time-frequency decomposition via CWT.
        3. Calculate energy distribution across predefined frequency bands.
        4. Apply dual-stage smoothing (moving average + Savitzky-Golay).
        5. Generate normalized output through dynamic range compression.
    """
    # Parameter preprocessing and validation
    if freq_band is None:
        freq_band = [3, 12]
    # Wavelet parameterization
    fc = pywt.central_frequency(wavelet=wavelet)
    c_param = 2 * fc * total_scal
    scales = c_param / np.arange(total_scal, 1, -1)
    # Continuous wavelet transform
    coefficient, freq = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
    # Energy calculation and band selection
    power_spec = np.abs(coefficient) ** 2
    freq_low, freq_high = freq_band
    freq_mask = (freq >= freq_low) & (freq <= freq_high)
    energy = np.sum(power_spec[freq_mask], axis=0)
    # smoothing
    moving_avg_window = int(fs * moving_window_sec)
    smoothed = np.convolve(energy, np.ones(moving_avg_window) / moving_avg_window, mode='same')
    # Savitzky-Golay filter optimization
    sg_window = int(fs * sg_window_sec)
    # Force singular value window
    sg_window = sg_window + 1 if sg_window % 2 == 0 else sg_window
    smooth_envelope = savgol_filter(smoothed, sg_window, polyorder=2)

    # Dynamic range normalization
    return (smooth_envelope - np.min(smooth_envelope)) / (np.max(smooth_envelope) - np.min(smooth_envelope))


def calculate_fusion_envelope(temporal_domain_envelope, spectral_domain_envelope):
    """
    Fuses the time and frequency domain envelope signals to generate a normalized composite feature metric.

    Parameters.
        temporal_domain_envelope (np.ndarray): normalized temporal envelope (shape: [N,]), range [0,1].
        spectral_domain_envelope (np.ndarray): normalized frequency envelope (shape: [N,]), range [0,1].

    Returns.
        np.ndarray: normalized integrated envelope signal (shape: [N,], range [0,1]).

    Processing Steps:
        1. Signal Fusion: Linear combination of temporal and spectral features.
        2. Dynamic Range Normalization: Adaptive scaling to [0,1] range.
    """
    # Core fusion algorithms
    fusion_envelope = temporal_domain_envelope + spectral_domain_envelope
    # Dynamic range normalization
    fusion_envelope = (fusion_envelope - np.min(fusion_envelope)) / (np.max(fusion_envelope) - np.min(fusion_envelope))
    return fusion_envelope


def automatic_multiscale_based_peak_detection(fusion_envelope):
    """
    Detect peaks in noisy periodic/quasi-periodic signals using Automatic Multiscale-based Peak Detection (AMPD) algorithm.

    Parameters:
        fusion_envelope (np.ndarray): Input 1D signal array for peak detection

    Returns:
        np.ndarray: Indices of detected peaks in the input signal

    Processing Steps:
        1. Initialize peak candidate matrix and row_sum array
        2. Perform multiscale local maxima analysis across window sizes
        3. Determine optimal window size through row_sum minimization
        4. Accumulate peak occurrences across valid scales
        5. Identify final peaks with maximum consistency across scales
    """
    # Initialize peak counter matrix with same shape as input
    p_data = np.zeros_like(fusion_envelope, dtype=np.int32)

    # Calculate signal length and initialize row_sum storage
    signal_length = fusion_envelope.shape[0]
    arr_row_sum = []

    # First pass: Multiscale local maxima analysis
    for window_size in range(1, signal_length // 2 + 1):
        current_row_sum = 0
        # Scan through valid signal positions for current window size
        for position in range(window_size, signal_length - window_size):
            # Check local maximum condition (current > both neighbors at this scale)
            if (fusion_envelope[position] > fusion_envelope[position - window_size] and
                    fusion_envelope[position] > fusion_envelope[position + window_size]):
                current_row_sum -= 1  # Negative counting for minimization
        arr_row_sum.append(current_row_sum)

    # Determine optimal scale through row_sum minimization
    optimal_scale_index = np.argmin(arr_row_sum)
    max_window_length = optimal_scale_index

    # Second pass: Accumulate peak occurrences across valid scales
    for scale in range(1, max_window_length + 1):
        for position in range(scale, signal_length - scale):
            # Validate peak presence at current scale
            if (fusion_envelope[position] > fusion_envelope[position - scale] and
                    fusion_envelope[position] > fusion_envelope[position + scale]):
                p_data[position] += 1  # Increment peak confidence counter

    # Return positions with maximum consensus across scales
    return np.where(p_data == max_window_length)[0]
