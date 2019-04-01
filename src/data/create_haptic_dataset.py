import deepdish as dd
import yaml
import collections
from .haptics_utils import *


def haptic_dataset(subjects, hand_type, control_type, config):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 8801
    hand   : Dominant or Non dominant hand

    Returns
    ----------
    eeg_dataset : dataset of all the subjects with different conditions

    """
    haptic_dataset = {}
    def nested_dict(): return collections.defaultdict(nested_dict)
    data = nested_dict()
    for subject in subjects:
        for hand in hand_type:
            for control in control_type:
                data['haptic'][hand][control] = create_haptic_emg_epoch(
                    subject, hand, control, config)
        haptic_dataset[subject] = data

    return haptic_dataset
