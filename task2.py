from re import X
from pandas import DataFrame, Series

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import ansi_escape_codes as c
from logger_config import logger

def splitDataSet(features: DataFrame, targets: Series, splitRatio: list[float]) -> tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """
    Split the dataset into training, evaluation, and test sets.

    Parameters
    ----------
    features : ndarray
        The feature dataset.
    targets : ndarray
        The target dataset.
    splitRatio : list[float]
        The ratio of the dataset to be used for training, evaluation, and testing.
   
    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]
        A tuple containing the training, evaluation, and test feature and target datasets.
    """
    # Check if the split ratios are valid
    if len(splitRatio) != 3 or sum(splitRatio) != 1.0:
        logger.fatal("{c.RED}You must provide 3 split ratios and the sum of split ratios must be 1.0{c.RESET}")
        exit(1)

    # Split the dataset into training set and temporary set (evaluation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, targets, test_size=splitRatio[1] + splitRatio[2], random_state=42
    )

    # Split the temporary set into evaluation and test sets
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_temp, y_temp, test_size=splitRatio[2] / (splitRatio[1] + splitRatio[2]), random_state=42
    )

    # Print the split ratios and shapes of the datasets
    logger.info(
        f"Split ratios: {c.CYAN}{splitRatio[0]*100}%{c.RESET} for {c.RED}training{c.RESET}, {c.CYAN}{splitRatio[1]*100}%{c.RESET} for {c.RED}evaluation{c.RESET}, and {c.CYAN}{splitRatio[2]*100}%{c.RESET} for {c.RED}testing{c.RESET}"
    )

    # Set data as categorical
    for col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_eval[col] = X_eval[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # Return the split datasets
    return X_train, X_eval, X_test, y_train, y_eval, y_test
