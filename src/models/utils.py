import pickle


def read_with_pickle(path):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.

    Returns
    __________
    data : dict
        dictionary of pandas dataframe to save


    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data
