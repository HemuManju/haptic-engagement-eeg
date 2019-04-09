import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import welch
from .utils import read_with_pickle, read_with_deepdish


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
        change_id = np.where(np.diff(np.sign(emg_data[channel,:])))[0]
        zero_crosses.append(len(change_id))

    return np.asarray(zero_crosses)


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
    diff =  np.diff(emg_data, axis=1)
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
    rms = np.sqrt(np.sum(emg_data**2, axis=1)/emg_data.shape[1])

    return rms


def welch_power(emg_data, config):
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
    psd = np.zeros((config['n_emg_electrodes'], config['emg_s_freq']//2 + 1))
    for channel in range(config['n_emg_electrodes']):
        f, psd[channel, :] = welch(emg_data[channel, :], fs=config['emg_s_freq'], nperseg=config['emg_s_freq']//2, nfft=config['emg_s_freq'], detrend=False)

    return psd.mean(axis=1)


def get_emg_feature(emg_data, config):
    """Create emg feature set.

    Parameters
    ----------
    emg_data : array
        A 8 channel array of emg.
    config : yaml
        The configuration file.

    Returns
    -------
    array
        An array of calculated emg features.

    """

    zero_crosses = zero_crosses_counter(emg_data, config)
    slope_zero_crosses = slope_zero_crosses_counter(emg_data, config)
    rms_values = rms(emg_data)
    psd = welch_power(emg_data, config)
    data = np.vstack((zero_crosses, slope_zero_crosses, rms_values, psd)).T
    df = pd.DataFrame(data, columns=config['emg_features'])

    return df


def create_emg_features(config):
    """Create EMG feature dataset

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    dataframe
        Pandas dataframe.

    """
    read_path = Path(__file__).parents[2] / config['raw_haptic_dataset']
    data = read_with_deepdish(read_path)
    emg_feature = pd.DataFrame(np.empty((0, len(config['emg_features']))), columns=config['emg_features'])
    channels = ['emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7']
    for subject in config['subjects']:
        for hand in config['hand_type']:
            for control in config['control_type']:
                emg_data = data[subject]['haptic'][hand][control]
                id = mne.pick_channels(emg_data.ch_names, channels)
                df = get_emg_feature(emg_data.get_data()[id, :], config)
                df['subject'] = subject
                df['hand_type'] = hand
                df['control_type'] = control

                emg_feature =  pd.concat([emg_feature, df], ignore_index=True, sort=False)

    return emg_feature
