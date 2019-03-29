import deepdish as dd
import yaml
import collections
from .eeg_utils import *


def eeg_dataset(subjects, hand_type):
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
    for subject in subjects:
        data = collections.defaultdict(dict)
        for hand in hand_type:
            data['eeg'][hand] = create_eeg_epochs(subject, hand)
        eeg_dataset[subject] = data

    return eeg_dataset
