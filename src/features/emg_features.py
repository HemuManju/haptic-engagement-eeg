import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import welch
from .utils import read_with_pickle, read_with_deepdish, compute_zero_crossings, compute_svd_entropy


def svd_entropy(emg_data):
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
    svd = compute_svd_entropy(emg_data)

    return svd


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
    df = pd.DataFrame(np.empty((0, len(config['emg_features']))), columns=config['emg_features'])
    for i in range(emg_data.shape[0]):
        zero_crosses = np.mean(compute_zero_crossings(emg_data[i,:,:]))
        diff =  np.diff(emg_data[i,:,:], axis=1)
        slope_zero_crosses = np.mean(compute_zero_crossings(diff))
        svd = np.mean(svd_entropy(emg_data[i,:,:]))
        rms = np.sqrt(np.sum(emg_data[i,:,:]**2, axis=1)/emg_data[i,:,:].shape[1])
        rms = np.mean(rms)
        data = np.vstack((zero_crosses, slope_zero_crosses, svd, rms)).T
        temp = pd.DataFrame(data, columns=config['emg_features'])
        df =  pd.concat([df, temp], ignore_index=True, sort=False)

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
                df = get_emg_feature(emg_data.get_data()[:, id, :], config)
                df['subject'] = subject
                df['hand_type'] = hand
                df['control_type'] = control

                emg_feature =  pd.concat([emg_feature, df], ignore_index=True, sort=False)

    return emg_feature
