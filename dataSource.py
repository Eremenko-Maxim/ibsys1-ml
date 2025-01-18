from itertools import chain
from multiprocessing import Value
from typing import Tuple
from numpy import ndarray
import pandas as pd
import ansi_escape_codes as c


def get_data_set_from_url(self) -> Tuple[ndarray, ndarray]:
    """
    Retrieve a dataset from a given URL and split it into features and targets.

    If no path is provided, the default dataset 'Test00.txt' is used.

    Returns
    -------
    features : ndarray
        The feature dataset
    targets : ndarray
        The target dataset
    """
    # Prompt the user to enter the dataset path
    dataset_path = input(f"{c.YELLOW}Type in the path to the dataset: {c.RESET}")

    # If no path is provided, use the default dataset
    if not dataset_path:
        dataset_path = "Test00.txt"

    try:
        # Read the dataset from the given path
        dataset = pd.read_csv(dataset_path, delimiter=' ', header=None)
    except FileNotFoundError:
        # Raise an error if the file path is invalid
        print(f"{c.RED}Invalid file path. Please try again.{c.RESET}")
        exit(1)

    # Extract the feature data from the dataset
    feature_data = dataset.iloc[:, :-1].values

    # Extract the target data from the dataset and strip the semicolon from the end
    target_data = dataset.iloc[:, -1].str.strip(';').values

    # Return the feature and target data
    return feature_data, target_data

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

