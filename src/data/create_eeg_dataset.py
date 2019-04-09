import deepdish as dd
import yaml
import collections
from .eeg_utils import *


def eeg_dataset(config):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 8801
    hand   : Dominant or Non dominant hand

    Returns
    ----------
    eeg_dataset : dataset of all the subjects with different conditions

    """
    eeg_dataset = {}
    nested_dict = lambda: collections.defaultdict(nested_dict)
    data = nested_dict()
    for subject in config['subjects']:
        for hand in config['hand_type']:
            for control in config['control_type']:
                data['eeg'][hand][control] = create_eeg_epochs(subject, hand, control, config)
        eeg_dataset[subject] = data

    return eeg_dataset
