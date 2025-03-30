import numpy as np
from scipy.signal import butter, filtfilt


def zero_phase_butter_worth_filter(data, low_cut=3, high_cut=12, fs=125, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to input data using forward and reverse directions.

    Parameters:
        data (np.ndarray): Input signal data to be filtered (1D array)
        low_cut (float): Lower cutoff frequency in Hz (default: 3Hz)
        high_cut (float): Upper cutoff frequency in Hz (default: 12Hz)
        fs (float): Sampling frequency in Hz (default: 125Hz)
        order (int): Order of the Butterworth filter (default: 4)

    Returns:
        np.ndarray: Filtered data with zero phase distortion

    Processing Steps:
        1. Calculate Nyquist frequency
        2. Normalize cutoff frequencies to Nyquist range [0, 1]
        3. Design Butterworth bandpass filter coefficients
        4. Apply forward-reverse filtering using filtfilt()
    """
    # Calculate Nyquist frequency (half of sampling frequency)
    nyquist_freq = 0.5 * fs

    # Normalize cutoff frequencies to Nyquist range [0, 1]
    low = low_cut / nyquist_freq
    high = high_cut / nyquist_freq

    # Design Butterworth bandpass filter coefficients
    [b, a] = butter(order, [low, high], btype='bandpass', analog=False)

    # Apply zero-phase filtering using forward and reverse directions
    filtered_data = filtfilt(b, a, data, axis=0)

    return filtered_data


def rms_filter(signal, window_size=200):
    """
    Calculate Root Mean Square (RMS) sequence using sliding symmetric window (loop implementation).

    Parameters:
        signal (np.ndarray): Input 1D signal array
        window_size (int): Number of samples in the analysis window (must be odd for symmetric window)

    Returns:
        np.ndarray: Normalized RMS sequence in range [0, 1]

    Processing Steps:
        1. Window configuration: Initialize symmetric analysis window
        2. Sliding computation: Iterate through signal with partial window handling
        3. Edge case management: Adaptive window adjustment at signal boundaries
        4. Energy normalization: Global max scaling to [0,1] range
    """
    rms_signal = np.zeros(signal.shape[0])
    for i in range(signal.shape[0]):
        # Dynamic window boundary calculations
        start = max(0, i - window_size // 2)
        end = min(signal.shape[0], i + window_size // 2)
        signal_mean = signal[start:end]
        # RMS energy calculations
        rms_signal[i] = np.sqrt(np.mean(signal_mean ** 2))
    # Global energy normalization
    return rms_signal / np.max(rms_signal)


def discard_near_threshold(rms, thresholds, discard_seconds=1, fs=125):
    """
    Discard RMS points near threshold-exceeding regions with specified duration.

    Parameters:
        rms (np.ndarray): Normalized RMS sequence in range [0, 1]
        thresholds (tuple): Lower and upper bounds (low, high) for valid RMS range
        discard_seconds (int): Time duration around threshold crossings to discard (default: 1)
        fs (int): Sampling frequency in Hz (default: 125)

    Returns:
        np.ndarray: Boolean mask indicating discarded RMS points (True = discarded)

    Processing Steps:
        1. Identifies areas where energy is exceeded (below low or above high).
        2. expand the exclusion window in both directions centered on the point of exceeding the limit.
        3. Adjust the window boundaries to prevent the index from crossing the boundaries
        4. merge all consecutive intervals to be excluded
    """
    low, high = thresholds
    exceed_low = rms < low
    exceed_high = rms > high
    exceed_mask = exceed_low | exceed_high
    discard_points = discard_seconds * fs

    # Initialize discard mask
    discard_mask = np.zeros_like(rms, dtype=bool)

    # Mark discard regions around threshold crossings
    for i in np.where(exceed_mask)[0]:
        start = max(0, i - discard_points)
        end = min(len(rms), i + discard_points + 1)
        discard_mask[start:end] = True

    return discard_mask


def cal_valid_segment_coverage(rms, thresholds):
    """
    Calculate valid signal coverage percentage after applying discard rules.

    Parameters:
        rms (np.ndarray): Normalized RMS sequence in range [0, 1]
        thresholds (tuple): Lower and upper bounds (low, high) for valid RMS range

    Returns:
        float: Proportion of valid signal in [0, 1] after discarding

    Processing Steps:
        1. application of the set-value(5-second) transition segment exclusion strategy
        2. Valid region determination: joint energy threshold and exclusion region constraints
        3. Coverage calculation: counting the proportion of valid sample points
    """
    # Generate transition segment exclusion masks (default 5-second exclusion window)
    rms_discard_mask = discard_near_threshold(rms, thresholds, discard_seconds=5)
    # Calculate the effective area of the composite condition
    valid_after_discard = (rms >= thresholds[0]) & (rms <= thresholds[1]) & np.logical_not(rms_discard_mask)
    # Calculate effective signal coverage
    coverage = np.mean(valid_after_discard)

    return coverage
