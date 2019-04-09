import deepdish as dd
from pathlib import Path
import pickle
import feather

def read_eeg_epochs(subject, hand_type, control_type, config):
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
    eeg_epochs = data['eeg'][hand_type][control_type]

    return eeg_epochs


def read_with_deepdish(path):
    """Read the hdf5 dataset

    Parameters
    ----------
    path : string
        Path to the dataset

    """
    data = dd.io.load(path)

    return data


def save_to_r_dataset(df, path):
    """Convert pandas dataframe to r dataframe.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe.
    path : str
        Path to save.

    Returns
    -------
    None
        Description of returned object.

    """

    feather.write_dataframe(df, path)

    return None


def read_with_pickle(path):
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
