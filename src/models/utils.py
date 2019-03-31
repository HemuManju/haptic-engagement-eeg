import deepdish as dd
from pathlib import Path
import pickle


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
