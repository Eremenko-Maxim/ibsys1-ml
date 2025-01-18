from itertools import chain
from typing import Tuple
from numpy import ndarray
import pandas as pd
import ansi_escape_codes as c


def get_data_set_from_url(self) -> Tuple[ndarray, ndarray]:
    """Retrieve a dataset from a given URL and split it into features and targets.

    If no URL is provided, the default dataset 'Test00.txt' is used.

    Returns
    -------
    features : ndarray
        The feature dataset
    targets : ndarray
        The target dataset
    """
    file_path = input(f"{c.YELLOW}Type in the path to the dataset: {c.RESET}")
    if not file_path:
        file_path = "Test00.txt"

    data = pd.read_csv(file_path, delimiter=' ', header=None)

    features = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].str.strip(';').values

    return features, targets

def get_values_for_feature(self, index: int, features: ndarray) -> list:
    """
    Return a list of all values for a given feature index in the dataset.

    Parameters
    ----------
    index : int
        The index of the feature in the dataset
    features : ndarray
        The feature dataset

    Returns
    -------
    list
        A list containing all values for the feature at the given index
    """
    return list(features[:, index])

