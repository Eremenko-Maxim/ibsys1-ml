from pandas import DataFrame, Series

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import ansi_escape_codes as c
import logging

logging.basicConfig(
    filename="log.txt",
    filemode="w",
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S")

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
        logging.fatal("{c.RED}You must provide 3 split ratios and the sum of split ratios must be 1.0{c.RESET}")
        exit(1)

    # Split the dataset into training set and temporary set (evaluation + test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, targets, test_size=splitRatio[1] + splitRatio[2], random_state=42
    )

    # Split the temporary set into evaluation and test sets
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_temp, y_temp, test_size=splitRatio[2] / (splitRatio[1] + splitRatio[2]), random_state=42
    )

    # Print the split ratios and shapes of the datasets
    logging.debug(
        f"Split ratios: {c.CYAN}{splitRatio[0]*100}%{c.RESET} for {c.RED}training{c.RESET}, {c.CYAN}{splitRatio[1]*100}%{c.RESET} for {c.RED}evaluation{c.RESET}, and {c.CYAN}{splitRatio[2]*100}%{c.RESET} for {c.RED}testing{c.RESET}"
    )
    logging.info(
        f"{c.BLUE}x_train{c.RESET}: {x_train.shape} \n{c.BLUE}x_eval{c.RESET}: {x_eval.shape} \n{c.BLUE}x_test{c.RESET}: {x_test.shape} \n{c.BLUE}y_train{c.RESET}: {y_train.shape} \n{c.BLUE}y_eval{c.RESET}: {y_eval.shape} \n{c.BLUE}y_test{c.RESET}: {y_test.shape}"
    )

    # Return the split datasets
    return x_train, x_eval, x_test, y_train, y_eval, y_test
