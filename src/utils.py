import feather
import deepdish as dd
from contextlib import contextmanager
import pickle

import sys


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """
    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>12}'.format(
                'Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>12}'.format('Running the block',
                                                       '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


def save_with_deepdish(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        hdf5 dataset.
    save : Bool

    """
    if save:
        dd.io.save(path, dataset)

    return None


def read_with_deepdish(path):
    """Read the dataset.

    Parameters
    ----------
    path : str
        path to read from.

    """
    dataset = dd.io.load(path)

    return dataset


def save_with_pickle(path, dataframe, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save

    save : Bool

    """
    if save:
        with open(path, 'wb') as f:
            pickle.dump(dataframe, f, pickle.HIGHEST_PROTOCOL)

    return None


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
