import deepdish as dd
import numpy as np
import mne
import yaml
import collections
from pathlib import Path
import ast
from .eeg_utils import *


def get_haptic_path(subject, hand_type, control_type, config):
    """Get the trial file path  a subject.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    trial_path   : path to a trial (Force) data to the subject

    """
    # Trial time
    path = Path(__file__).parents[2] / \
        config['raw_haptic_path'] / subject / hand_type
    for file in path.iterdir():
        file_name = file.name.split('.')
        if file_name[0] == control_type:
            break
    haptic_path = file

    return haptic_path


def convert_to_array(data):
    """Convert the given list of data with strings to numpy array.

    Parameters
    ----------
    data : list of string tuple
        A list contaning tuples in string format.

    Returns
    -------
    array
        Converted numpy array.

    """

    converted = [list(ast.literal_eval(x)) for x in data]

    return np.asarray(converted).T


def get_haptic_emg_data(subject, hand_type, control_type, config):
    """Get the haptic data.

    Parameter
    ----------
    subject  : path to eeg file

    Returns
    ----------
    trial_start : start time of the trial with eeg as reference
    trial_end   : end time of the trial with eeg as reference

    """
    # EEG time
    haptic_path = get_haptic_path(subject, hand_type, control_type, config)

    # Trial time
    column_name = np.genfromtxt(
        haptic_path, dtype=str, delimiter=';', max_rows=1).tolist()
    features = ['CursorPosition', 'desiredPosition',
                ' desiredPointOnSpline', 'proportionalGain', 'keyPressed']
    dummy = np.genfromtxt(haptic_path, dtype=str,
                          delimiter=';', usecols=0, skip_header=1).tolist()
    ids = [i for i, x in enumerate(column_name) if x in features]
    haptic_data = np.empty((0, len(dummy)))
    columns = []
    for i, id in enumerate(ids):
        data = np.genfromtxt(haptic_path, dtype=str,
                             delimiter=';', usecols=id, skip_header=1).tolist()
        columns.append(features[i].lower())
        array = convert_to_array(data)
        haptic_data = np.append(haptic_data, array, axis=0)

    return haptic_data, columns


def create_haptic_emg_epoch(subject, hand_type, control_type, config):
    """Creates haptic and emg epochs.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    control_type : str
        Control type no_force, convergent or divergent.
    config : yaml
        Configuration file.

    Returns
    -------
    epochs
        mne epoch object.

    """

    haptic_data, columns = get_haptic_emg_data(
        subject, hand_type, control_type, config)
    id_cursor = columns.index('cursorposition')
    id_desired = columns.index('desiredposition')
    id_gain = columns.index('proportionalgain')

    # Calculate the error
    error = haptic_data[id_cursor * 3:id_cursor * 3 + 3] - \
        haptic_data[id_desired * 3:id_desired * 3 + 3]
    k = haptic_data[id_gain * 3:id_gain * 3 + 3]  # gain
    force = np.multiply(error, k)

    # Concatenate with the haptic data
    data = np.concatenate((haptic_data, force, error), axis=0)

    # The data was stored in such a way that keyPressed is actual emg, so replace the name
    columns[columns.index('keypressed')] = 'emg'
    haptic_info = [x + y for x in ['cursor', 'desired',
                                   'spline', 'gain'] for y in ['_x', '_y', '_z']]
    emg_info = ['emg_' + str(i) for i in range(8)]
    force_info = ['force' + x for x in ['_x', '_y', '_z']]
    error_info = ['error' + x for x in ['_x', '_y', '_z']]
    names_info = haptic_info + emg_info + force_info + error_info

    info = mne.create_info(ch_names=names_info,
                           ch_types=['misc'] * len(names_info),
                           sfreq=50.0)
    epochs = mne.io.RawArray(data, info, verbose=False)

    return epochs
