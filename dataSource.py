from typing import Tuple
from numpy import ndarray
from pandas import DataFrame, Series, read_csv
from logger_config import logger
import ansi_escape_codes as c

def get_data_set_from_url() -> Tuple[DataFrame, DataFrame]:
    """
    Retrieve a dataset from a given URL and split it into features and targets.

    If no path is provided, the default dataset 'Test00.txt' is used.

    Returns
    -------
    features : DataFrame
        The feature dataset
    targets : DataFrame
        The target dataset
    """
    # Define the column names
    column_names = [f"Feature_{i}" for i in range(1, 17)] + ["Target"]

    # Prompt the user to enter the dataset path
    dataset_path = input(f"{c.YELLOW}Type in the path to the dataset: {c.RESET}")

    # If no path is provided, use the default dataset
    if not dataset_path:
        dataset_path = "Test00.txt"

    try:
        # Read the dataset from the given path
        dataset = read_csv(dataset_path, sep=' ', header=None, names=column_names)

        # Strip the semicolon from the end of each value
        dataset = dataset.map(lambda x: str(x).rstrip(';'))

    except FileNotFoundError:
        # Raise an error if the file path is invalid
        logger.fatal(f"{c.RED}Invalid file path. Please try again.{c.RESET}")
        exit(1)

    # Extract the feature data from the dataset
    feature_data = dataset[[col for col in dataset.columns if col != 'Target']].astype('category')

    # Extract the target data from the dataset 
    target_data = dataset["Target"].astype('category')

    # Return the features and target
    return feature_data, target_data
