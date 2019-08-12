import collections
from .eeg_utils import (create_eeg_epochs)


def eeg_dataset(config):
    """Create the data with each subject data in a dictionary.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    ----------
    eeg_dataset : dict
        A dictionary dataset of all the subjects with different conditions.

    """
    eeg_dataset = {}
    for subject in config['subjects']:
        data = collections.defaultdict(dict)
        for hand in config['hand_type']:
            for control in config['control_type']:
                data['eeg'][hand][control] = create_eeg_epochs(
                    subject, hand, control, config)
        eeg_dataset[subject] = data

    return eeg_dataset
