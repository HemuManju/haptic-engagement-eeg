import deepdish as dd
import yaml
import collections
from .eeg_utils import *


def eeg_dataset(subjects, hand_type, control_type, config):
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
    for subject in subjects:
        for hand in hand_type:
            for control in control_type:
                data['eeg'][hand][control] = create_eeg_epochs(subject, hand, control, config)
        eeg_dataset[subject] = data

    return eeg_dataset
