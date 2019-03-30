import numpy as np
from mne.time_frequency import psd_multitaper
import pandas as pd
import mne
from .utils import read_eeg_epochs


def get_band_power(subject, hand_type, config):
    """Calculate the band power of EEG signals.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    config : yaml file
        Configuration file.

    Returns
    -------
    dataframe
        6 band powers of given subject and hand type at different sensor locations.

    """
    epochs = read_eeg_epochs(subject, hand_type, config)
    picks = mne.pick_types(epochs.info, eeg=True)
    ch_names = epochs.ch_names[picks[0]:picks[-1]+1]
    psds, freqs = psd_multitaper(epochs, fmin=1.0, fmax=45.0, picks=picks)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    psd_band = []
    for freq_band in config['freq_bands']:
        temp = psds[:, :, (freqs >= freq_band[0]) & (freqs < freq_band[1])]
        psd_band.append(psds[:, :, (freqs >= freq_band[0]) & (freqs < freq_band[1])].mean(axis=-1))
    # Form pandas dataframe
    data = np.concatenate(psd_band, axis=1)
    columns = [x + '_' + y for x in ch_names for y in config['band_names']]
    df = pd.DataFrame(data, columns=columns)
    df['subject'] = subject
    df['hand_type'] = hand_type

    return df



def all_subjects_band_power(subjects, hand_type, config):
    """Band power of all subjects.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    config : yaml file
        Configuration file.

    Returns
    -------
    dataframe
        6 band powers of given subject and hand type at different sensor locations.

    """

    band_power_dataset = {}
    for subject in subjects:
        df = []
        for hand in hand_type:
            df.append(get_band_power(subject, hand, config))
        band_power_dataset[subject] = pd.concat([df[0], df[1]], ignore_index=True)

    return band_power_dataset
