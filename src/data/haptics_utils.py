import numpy as np
import mne

import matplotlib.pyplot as plt

from pathlib import Path
import ast
from .eeg_utils import read_eeg_epochs


def get_haptic_path(subject, hand_type, control_type, config):
    """Get the trial file path  a subject.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    control_type : str
        Control type (error augmentation or error reduction)
    config : yaml
        The configuration file.

    Returns
    ----------
    trial_path  : str
        A path to a trial (Force) data to the subject

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
    array : numpy array
        Converted numpy array.

    """
    try:
        converted = [list(ast.literal_eval(x)) for x in data]
        length = len(converted[0])
    except TypeError:
        converted = [ast.literal_eval(x) for x in data]
        length = 1
    data_array = np.asarray(converted).reshape((length, -1))
    return data_array


def get_haptic_emg_data(subject, hand_type, control_type, config):
    """Get the haptic data.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    control_type : str
        Control type (error augmentation or error reduction)
    config : yaml
        The configuration file.

    Returns
    ----------
    trial_start : start time of the trial with eeg as reference
    trial_end   : end time of the trial with eeg as reference

    """
    # EEG time
    haptic_path = get_haptic_path(subject, hand_type, control_type, config)

    # Trial time
    column_name = np.genfromtxt(haptic_path,
                                dtype=str,
                                delimiter=';',
                                max_rows=1).tolist()
    # features = [
    #     'totalTime', 'CursorPosition', 'CursorPositionVirtual', 'TrialNumber',
    #     'cursorVelocity', 'desiredPosition', ' desiredPointOnSpline',
    #     'proportionalGain', 'keyPressed'
    # ]
    features = [
        'totalTime', 'CursorPositionVirtual', 'cursorVelocity',
        'desiredPosition', ' desiredPointOnSpline', 'proportionalGain',
        'keyPressed'
    ]
    n_columns = np.genfromtxt(haptic_path,
                              dtype=str,
                              delimiter=';',
                              usecols=0,
                              skip_header=1).tolist()
    time = np.genfromtxt(haptic_path,
                         dtype=float,
                         delimiter=';',
                         usecols=1,
                         skip_header=1)
    sampling_freq = 1 / np.mean(np.diff(time))
    ids = [i for i, x in enumerate(column_name) if x in features]
    haptic_data = np.empty((0, len(n_columns)))

    columns = []
    for i, id in enumerate(ids):
        data = np.genfromtxt(haptic_path,
                             dtype=str,
                             delimiter=';',
                             usecols=id,
                             skip_header=1).tolist()
        columns.append(features[i].lower())
        array = convert_to_array(data)
        haptic_data = np.append(haptic_data, array, axis=0)

    # t = haptic_data.T

    # x = [1, 10, 13]
    # y = [2, 11, 14]
    # z = [3, 12, 15]
    # fig, ax = plt.subplots(1, 4)
    # for i in range(4, 8):

    #     ind = t[:, 7] == i
    #     ax[i % 4].plot(t[ind, 4], t[ind, 6], label='CursorVirtual')
    #     ax[i % 4].plot(t[ind, 11], t[ind, 13], label='DesiredPosition')
    #     ax[i % 4].plot(t[ind, 14], t[ind, 16], label='DesiredPositionOnSpline')

    # plt.legend()
    # plt.show()
    # print(a)
    return haptic_data, columns, sampling_freq


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
    epochs  : mne epoch object.
        A mne epoch containing all the information from the haptic device.


    """

    haptic_data, columns, sampling_freq = get_haptic_emg_data(
        subject, hand_type, control_type, config)
    id_cursor = columns.index('cursorpositionvirtual') - 1
    id_desired = columns.index('desiredposition') - 1
    id_gain = columns.index('proportionalgain') - 1
    id_velocity = columns.index('cursorvelocity') - 1
    id_time = columns.index('totaltime')

    # Calculate total time
    time_mask = haptic_data[id_time, :]
    time = np.expand_dims(time_mask - time_mask[0], axis=1).T
    total_time = np.max(time) + time * 0

    # Remove time column
    haptic_data = np.delete(haptic_data, id_time, 0)

    # Calculate the error
    error = haptic_data[id_cursor * 3:id_cursor * 3 + 3, :] - \
        haptic_data[id_desired * 3:id_desired * 3 + 3, :]
    k = haptic_data[id_gain * 3:id_gain * 3 + 3, :]  # gain
    k = k * 0 + 1.5
    total_error = np.linalg.norm(error[0:2, :], axis=0, keepdims=True)
    avg_error = np.mean(total_error) + total_error * 0

    # Calculate the force
    force = np.multiply(error, k)
    total_force = np.linalg.norm(force[0:2, :], axis=0, keepdims=True)
    avg_force = np.mean(total_force) + total_force * 0

    # Calculate the speed
    speed = np.linalg.norm(haptic_data[id_velocity * 3:id_velocity * 3 + 2, :],
                           axis=0,
                           keepdims=True)
    avg_speed = np.mean(speed) + speed * 0

    # Concatenate with the haptic data
    data = np.concatenate(
        (haptic_data, force, error, speed, time, total_force, total_error,
         total_time, avg_error, avg_speed, avg_force),
        axis=0)

    # The data was stored in such a way that
    # keyPressed is actual emg, so replace the name
    columns[columns.index('keypressed')] = 'emg'
    writing_info = [
        x + y for x in ['cursor', 'velocity', 'desired', 'spline', 'gain']
        for y in ['_x', '_y', '_z']
    ]
    emg_info = ['emg_' + str(i) for i in range(8)]
    force_info = ['force' + x for x in ['_x', '_y', '_z']]
    error_info = ['error' + x for x in ['_x', '_y', '_z']]
    haptic_info = [
        'speed', 'time', 'total_force', 'total_error', 'total_time',
        'avg_error', 'avg_speed', 'avg_force'
    ]

    names_info = writing_info + emg_info + force_info + error_info + haptic_info  # noqa
    info = mne.create_info(ch_names=names_info,
                           ch_types=['misc'] * len(names_info),
                           sfreq=sampling_freq)
    raw = mne.io.RawArray(data, info, verbose=False)
    events = mne.make_fixed_length_events(raw, duration=config['epoch_length'])
    epochs = mne.Epochs(raw,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        verbose=False,
                        baseline=(0, 0))

    # Sync with eeg time
    eeg_epochs = read_eeg_epochs(subject, hand_type, control_type,
                                 config)  # eeg file
    drop_id = [id for id, val in enumerate(eeg_epochs.drop_log) if val]
    if len(eeg_epochs.drop_log) != len(epochs.drop_log):
        print(len(eeg_epochs.drop_log), len(epochs.drop_log))
        raise Exception('Two epochs are not of same length!')
    else:
        epochs.drop(drop_id)
    return epochs


def create_sync_haptic_data(subject, hand_type, control_type, config):
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
    epochs  : mne epoch object.
        A mne epoch containing all the information from the haptic device.


    """

    haptic_data, columns, sampling_freq = get_haptic_emg_data(
        subject, hand_type, control_type, config)
    id_cursor = columns.index('cursorpositionvirtual') - 1
    id_desired = columns.index('desiredposition') - 1
    id_gain = columns.index('proportionalgain') - 1
    id_time = columns.index('totaltime')

    # Calculate total time
    time_mask = haptic_data[id_time, :]
    time = np.expand_dims(time_mask - time_mask[0], axis=1).T
    total_time = np.max(time) + time * 0

    # Remove time column
    haptic_data = np.delete(haptic_data, id_time, 0)

    # Calculate the error
    error = haptic_data[id_cursor * 3:id_cursor * 3 + 3, :] - \
        haptic_data[id_desired * 3:id_desired * 3 + 3, :]
    k = haptic_data[id_gain * 3:id_gain * 3 + 3, :]  # gain
    if control_type == 'error_reduction':
        k = k * 0 - 1.5
    elif control_type == 'no_force':
        k = k * 0
    else:
        k = k * 0 + 1.5

    total_error = np.linalg.norm(error[0:2, :], axis=0, keepdims=True)
    avg_error = np.mean(total_error) + total_error * 0

    # Calculate the force
    force = np.multiply(error, k)
    total_force = (k[0, 0] / 1.5) * np.linalg.norm(
        force[0:2, :], axis=0, keepdims=True)
    avg_force = np.mean(total_force) + total_force * 0

    # Concatenate with the haptic data
    data = np.concatenate((haptic_data, force, error, time, total_force,
                           total_error, total_time, avg_error, avg_force),
                          axis=0)

    # The data was stored in such a way that
    # keyPressed is actual emg, so replace the name
    columns[columns.index('keypressed')] = 'emg'
    writing_info = [
        x + y for x in ['cursor', 'velocity', 'desired', 'spline', 'gain']
        for y in ['_x', '_y', '_z']
    ]
    emg_info = ['emg_' + str(i) for i in range(8)]
    force_info = ['force' + x for x in ['_x', '_y', '_z']]
    error_info = ['error' + x for x in ['_x', '_y', '_z']]
    haptic_info = [
        'time', 'total_force', 'total_error', 'total_time', 'avg_error',
        'avg_force'
    ]

    names_info = writing_info + emg_info + force_info + error_info + haptic_info  # noqa
    info = mne.create_info(ch_names=names_info,
                           ch_types=['misc'] * len(names_info),
                           sfreq=sampling_freq)
    raw = mne.io.RawArray(data, info, verbose=False)
    events = mne.make_fixed_length_events(raw, duration=config['epoch_length'])
    epochs = mne.Epochs(raw,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        verbose=False,
                        baseline=(0, 0))

    total_force = epochs.get_data(['total_force']).mean(axis=-1)

    # Sync with eeg time
    eeg_epochs = read_eeg_epochs(subject, hand_type, control_type,
                                 config)  # eeg file
    drop_id = [id for id, val in enumerate(eeg_epochs.drop_log) if val]
    if len(eeg_epochs.drop_log) != len(epochs.drop_log):
        print(len(eeg_epochs.drop_log), len(epochs.drop_log))
        raise Exception('Two epochs are not of same length!')
    else:
        epochs.drop(drop_id)
    return epochs
