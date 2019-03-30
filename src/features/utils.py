import deepdish as dd
from pathlib import Path


def read_eeg_epochs(subject, hand_type, config):
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


def read_dataframe_dict(path):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save


    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
