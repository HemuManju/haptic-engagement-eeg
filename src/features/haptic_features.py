import numpy as np
import pandas as pd
import mne
from pathlib import Path
from .utils import read_with_deepdish


def get_haptic_feature(haptic_data, features):
    """Create haptic feature set.

    Parameters
    ----------
    haptic_data : array
        A n channel array of haptic.
    features : list
        A list of string specifying the featuers

    Returns
    -------
    array
        A dataframe of calculated haptic features.
    """
    mean_data = np.mean(haptic_data, axis=2)
    df = pd.DataFrame(mean_data, columns=features)
    return df


def create_haptic_features(config):
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
    channels = [
        'speed', 'time', 'total_force', 'total_error', 'total_time',
        'avg_error', 'avg_speed'
    ]
    haptic_features = pd.DataFrame(np.empty((0, len(channels))),
                                   columns=channels)
    for subject in config['subjects']:
        for hand in config['hand_type']:
            for control in config['control_type']:
                haptic_data = data[subject]['haptic'][hand][control]
                idx = mne.pick_channels(haptic_data.ch_names, channels)
                df = get_haptic_feature(haptic_data.get_data()[:, idx, :],
                                        channels)
                df['subject'] = subject
                df['hand_type'] = hand
                df['control_type'] = control
                haptic_features = pd.concat([haptic_features, df],
                                            ignore_index=True,
                                            sort=False)

    return haptic_features
