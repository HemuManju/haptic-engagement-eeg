import numpy as np
from scipy.signal import welch


def zero_crosses_counter(emg_data, config):
    """Number of zero crossing form the EMG data

    Parameters
    ----------
    emg_data : array
        An array of EMG data.

    Returns
    -------
    int
        Number of zeros crosses of all 8 channels.

    """
    zero_crosses = []
    for channel in range(config['n_emg_electrodes']):
        change_id = np.where(np.diff(np.sign(emg_data[:,channel])))[0]
        zero_crosses.append(len(change_id))

    return zero_crosses


def slope_zero_crosses_counter(emg_data, config):
    """Number of zero crossing form the EMG data

    Parameters
    ----------
    emg_data : array
        An array of EMG data.

    Returns
    -------
    int
        Number of zeros crosses of all 8 channels.

    """
    diff =  np.diff(emg_data, axis =0)
    slope_zero_crosses = zero_crosses_counter(diff, config)

    return slope_zero_crosses


def rms(emg_data):
    """RMS value of each channles EMG data.

    Parameters
    ----------
    emg_data : array
        An array of EMG data.

    Returns
    -------
    int
        Number of zeros crosses of all 8 channels.

    """
    rms = np.sqrt(np.sum(emg_data**2, axis=0)/x.shape[1])

    return rms


def fft(emg_data, config):
    """Get the fft value of emg signal from 8 electrodes.

    Parameters
    ----------
    emg_data : array
        An array of EMG data.

    Returns
    -------
    int
        Number of zeros crosses of all 8 channels.

    """
    fft = np.zeros((int (x.shape[0]/2) + 1, x.shape[1]))
    for channel in range(config['n_emg_electrodes']):
        f, fft[:,channel] = welch(x[:,channel], fs=config['emg_s_freq'])

    return fft.mean(axis = 0)
