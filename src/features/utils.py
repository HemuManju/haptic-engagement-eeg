import deepdish as dd
from pathlib import Path
import pickle
import feather
import numpy as np
from math import floor
from warnings import warn


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


def compute_zero_crossings(data, threshold=np.finfo(np.float64).eps):
    """Number of zero-crossings (per channel).
        The "threshold" parameter is used to clip 'small' values
        to zero.Changing its default value is likely
        to affect the number ofzero-crossings returned by the function.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    threshold : float (default: np.finfo(np.float64).eps)
        Threshold used to determine when a float should de treated as zero.

    Returns
    -------
    output : ndarray, shape (n_channels,)

    """
    _data = data.copy()
    # clip 'small' values to 0
    _data[np.abs(_data) < threshold] = 0
    sgn = np.sign(_data)
    # sgn may already contain 0 values (either 'true' zeros or clipped values)
    aux = np.diff((sgn == 0).astype(np.int64), axis=-1)
    count = np.sum(aux == 1, axis=-1) + (_data[:, 0] == 0)
    # zero between two consecutive time points (data[i] * data[i + 1] < 0)
    mask_implicit_zeros = sgn[:, 1:] * sgn[:, :-1] < 0
    count += np.sum(mask_implicit_zeros, axis=-1)
    return count


def compute_svd_entropy(data, tau=2, emb=10):
    """SVD entropy (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    tau : int (default: 2)
        Delay (number of samples).
    emb : int (default: 10)
        Embedding dimension.
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **svd_entropy**. See [1]_.
    References
    ----------
    .. [1] Roberts, S. J. et al. (1999). Temporal and spatial complexity
           measures for electroencephalogram based brain-computer interfacing.
           Medical & biological engineering & computing, 37(1), 93-98.
    """
    _, sv, _ = np.linalg.svd(_embed(data, d=emb, tau=tau))
    m = np.sum(sv, axis=-1)
    sv_norm = np.divide(sv, m[:, None])
    return -np.sum(np.multiply(sv_norm, np.log2(sv_norm)), axis=-1)


def _embed(x, d, tau):
    """Time-delay embedding.
    Parameters
    ----------
    x : ndarray, shape (n_channels, n_times)
    d : int
        Embedding dimension.
        The embedding dimension ``d`` should be greater than 2.
    tau : int
        Delay.
        The delay parameter ``tau`` should be less or equal than
        ``floor((n_times - 1) / (d - 1))``.
    Returns
    -------
    output : ndarray, shape (n_channels, n_times - (d - 1) * tau, d)
    """
    tau_max = floor((x.shape[1] - 1) / (d - 1))
    if tau > tau_max:
        warn('The given value (%s) for the parameter `tau` exceeds '
             '`tau_max = floor((n_times - 1) / (d - 1))`. Using `tau_max` '
             'instead.' % tau)
        _tau = tau_max
    else:
        _tau = int(tau)
    x = x.copy()
    X = np.lib.stride_tricks.as_strided(
        x, (x.shape[0], x.shape[1] - d * _tau + _tau, d),
        (x.strides[-2], x.strides[-1], x.strides[-1] * _tau))
    return X
