import pandas as pd
import numpy as np


def read_raw_data_from_csv(file_name, begin_sec, end_sec, fs):
    """
    Read and extract raw data (BCG or ECG) from CSV file within specified time range.

    Parameters:
        file_name (str): Path to CSV file containing raw data
        begin_sec (float): Start time in seconds for data extraction
        end_sec (float): End time in seconds for data extraction
        fs (int): Sampling frequency in Hz

    Returns:
        np.ndarray: 2D array of raw data samples from specified time range [begin_sample:end_sample, :]

    Note:
        - Since BCG data is single-channel and ECG data is three-channel, a two-dimensional array is returned
        - Uses integer conversion for time-to-sample calculation (may truncate fractional samples)
        - Follows Python slicing convention: includes begin_sample, excludes end_sample
        - No automatic range clipping - ensure input times are within file duration
        - Returns empty array if begin_sample >= end_sample
    """
    raw_data = np.array(pd.read_csv(file_name))
    begin_sample_point = int(begin_sec * fs)
    end_sample_point = int(end_sec * fs)
    return raw_data[begin_sample_point:end_sample_point, :]
