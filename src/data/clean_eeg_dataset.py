import deepdish as dd
from autoreject import get_rejection_threshold
import yaml
import collections
from .eeg_utils import *


def autoreject_repair_epochs(epochs, reject_plot=False):
    """Rejects the bad epochs with AutoReject algorithm

    Parameter
    ----------
    epochs : Epoched, filtered eeg data
    Returns
    ----------
    epochs : Epoched data after rejection of bad epochs

    """
    # Cleaning with autoreject
    picks = mne.pick_types(epochs.info, eeg=True)  # Pick EEG channels
    ar = AutoReject(n_interpolate=[1, 2, 3], n_jobs=6, picks=picks,
                    thresh_func='bayesian_optimization',
                    cv=3,
                    random_state=42, verbose=False)

    cleaned_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    if reject_plot:
        reject_log.plot_epochs(epochs, scalings=dict(eeg=40e-6))

    return cleaned_epochs


def append_eog_index(epochs, ica):
    """Detects the eye blink aritifact indices and adds that information to ICA

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne
    Returns
    ----------
    ICA : ICA object with eog indices appended

    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='F3', verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='F4', verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    return ica


def clean_with_ica(epochs, subject, hand, control, config, show_ica=False):
    """Clean epochs with ICA.

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs

    """

    picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                           eog=False, stim=False, exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="picard", verbose=False)
    # Get the rejection threshold using autoreject
    if config['use_previous_ica']:
        read_path = Path(__file__).parents[2] / config['previous_ica']
        data = data = dd.io.load(str(read_path))
        ica_previous = data[subject]['ica'][hand][control]
        ica_previous.apply(epochs)
    else:
        reject_threshold = get_rejection_threshold(epochs)
        ica.fit(epochs, picks=picks, reject=reject_threshold)
        # mne pipeline to detect artifacts
        ica.detect_artifacts(epochs, eog_criterion=range(2))
        ica.apply(epochs)  # Apply the ICA

    if show_ica:
        ica.plot_components(inst=epochs)


    return epochs, ica


def clean_dataset(config):
    """Create cleaned dataset (by running autoreject and ICA) with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    clean_eeg_dataset : dataset of all the subjects with different conditions

    """

    clean_eeg_dataset = {}
    read_path = Path(__file__).parents[2] / config['raw_eeg_dataset']
    raw_eeg = dd.io.load(str(read_path))  # load the raw eeg
    def nested_dict(): return collections.defaultdict(nested_dict)
    for subject in config['subjects']:
        data = nested_dict()
        for hand in config['hand_type']:
            for control in config['control_type']:
                epochs = raw_eeg[subject]['eeg'][hand][control]
                ica_epochs, ica = clean_with_ica(epochs, subject, hand, control, config)
                repaired_eeg = autoreject_repair_epochs(ica_epochs)
                data['eeg'][hand][control] = repaired_eeg
                data['ica'][hand][control] = ica
        clean_eeg_dataset[subject] = data

    return clean_eeg_dataset
