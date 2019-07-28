import collections
from .haptics_utils import (create_haptic_emg_epoch)


def haptic_dataset(config):
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

    def nested_dict():
        return collections.defaultdict(nested_dict)

    for subject in config['subjects']:
        data = nested_dict()
        for hand in config['hand_type']:
            for control in config['control_type']:
                data['haptic'][hand][control] = create_haptic_emg_epoch(
                    subject, hand, control, config)
        haptic_dataset[subject] = data

    return haptic_dataset
