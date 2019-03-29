import mne
import numpy as np
from pathlib import Path
import seaborn as sb
import pandas as pd
from mne.parallel import parallel_func
from mne.decoding import CSP
from scipy import signal
from scipy.signal import resample
from datetime import datetime
from autoreject import Ransac, AutoReject
from mne.time_frequency import psd_multitaper
from autoreject import get_rejection_threshold
import yaml
import deepdish as dd


# Import configuration
config = yaml.load(open(str(Path(__file__).parents[1]) + '/config.yml'))


def get_eeg_path(subject, hand_type, raw=True):
    """Path to EEG data

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    raw : bool
        Raw file or decontaminated file.

    Returns
    -------
    str
        path to a EEG data to the subject.

    """

    # EEG file
    path = Path(__file__).parents[2] / config['raw_eeg_path'] / subject
    fname = [str(f) for f in path.iterdir() if f.suffix == '.edf']
    fname.sort() # sorted according to time
    id = 1 if hand_type == 'dominant' else 3
    if raw:
        eeg_path = fname[id]  # raw file
    else:
        eeg_path = fname[id-1]  # decontaminated file

    return eeg_path


def get_emg_path(subject, hand_type):
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
    path = Path(__file__).parents[2] / config['raw_emg_path'] / subject
    for file in path.iterdir():
        file_name = file.name.split('_')
        if file_name[1] == trial:
            break
    trial_path = file

    return trial_path


def get_eeg_time(subject, hand_type):
    """Start time of eeg recording.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.

    Returns
    -------
    time
        EEG recorded time.

    """
    # EEG time
    eeg_path = get_eeg_path(subject, hand_type)
    eeg_time = eeg_path.split('.')
    eeg_time = datetime.strptime(
        ''.join(eeg_time[1:3]) + '0000', '%d%m%y%H%M%S%f')

    return eeg_time


def get_trial_time(subject, trial):
    """Get the start and end time of a trial to align with eeg data.

    Parameter
    ----------
    subject  : path to eeg file

    Returns
    ----------
    trial_start : start time of the trial with eeg as reference
    trial_end   : end time of the trial with eeg as reference

    """
    # EEG time
    eeg_time = get_eeg_time(subject)
    trial_path = get_trial_path(subject, trial)

    # Trial time
    trial_time = np.genfromtxt(trial_path, dtype=str, delimiter=',',
                               usecols=0, skip_footer=150, skip_header=100).tolist()

    # Update year, month, and day
    start_t = datetime.strptime(trial_time[0], '%H:%M:%S:%f')
    start_t = start_t.replace(year=eeg_time.year,
                              month=eeg_time.month, day=eeg_time.day)
    end_t = datetime.strptime(trial_time[-1], '%H:%M:%S:%f')
    end_t = end_t.replace(year=eeg_time.year,
                          month=eeg_time.month, day=eeg_time.day)

    trial_start = (start_t - eeg_time).total_seconds()  # convert to seconds
    trial_end = (end_t - eeg_time).total_seconds()

    return trial_start, trial_end


def get_eeg_data(subject, hand_type):
    """Get the eeg data excluding unnecessary channels from edf file.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    hand_type : str
        hand_type of the subject dominant or non-dominant.
    raw : bool
        Raw file or decontaminated file.

    Returns
    -------
    mne data file
        selected raw eeg out of whole experiment.

    """


    eeg_path = get_eeg_path(subject, hand_type)
    eeg_time = get_eeg_time(subject, hand_type)
    # EEG info
    info = mne.create_info(ch_names=['POz','Fz','Cz','C3','C4','F3','F4','P3','P4','STI 014'],
                           ch_types=['eeg'] * 9 + ['stim'],
                           sfreq=256.0,
                           montage="standard_1020")
    # Read the raw data
    exclude = ['ECG', 'AUX1', 'AUX2', 'AUX3', 'ESUTimestamp',
               'SystemTimestamp', 'Tilt X', 'Tilt Y', 'Tilt Z']
    raw = mne.io.read_raw_edf(eeg_path, preload=True,
                              exclude=exclude, verbose=False)
    data = raw.get_data()
    raw_selected = mne.io.RawArray(data, info, verbose=False)

    # Additional information
    meas_date = 'measure_time:' + eeg_time.strftime('%m-%d-%Y,%H:%M:%S')
    raw_selected.info['description'] = meas_date
    raw_selected.info['subject_info'] = subject
    raw_selected.info['experimenter'] = 'hemanth'

    return raw_selected


def create_eeg_epochs(subject, hand_type, preload=True):
    """Get the epcohed eeg data excluding unnessary channels from fif file and also filter the signal.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    epochs  : epoched data

    """
    raw = get_eeg_data(subject, hand_type)
    raw.notch_filter(60, filter_length='auto',
                             phase='zero', verbose=False)  # Line noise
    raw.filter(l_freq=1, h_freq=50, fir_design='firwin', verbose=False)  # Band pass filter
    raw.set_eeg_reference('average')
    events = mne.make_fixed_length_events(raw, duration=config['epoch_length'])
    epochs = mne.Epochs(raw, events, tmin=0,
                        tmax=config['epoch_length'], verbose=False, preload=preload)

    return epochs


def read_eeg_epochs(subject, hand_type):
    """Reads the eeg epoch file of given subject and trial

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    epoch
        EEG epoch.

    """
    eeg_path = str(Path(__file__).parents[2] / config['clean_eeg_dataset'])
    data = dd.io.load(eeg_path, group='/' + subject)
    eeg_epochs = data['eeg'][hand_type]

    return eeg_epochs
