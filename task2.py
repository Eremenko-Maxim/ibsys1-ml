from numpy import ndarray

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def splitDataSet(self, features: ndarray, targets: ndarray, splitRatio: list[float], verbose:bool = False) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Split the dataset into training, evaluation and test set.

    Parameters
    ----------
    features : ndarray
        The feature dataset
    targets : ndarray
        The target dataset
    splitRatio : float[3]
        The ratio of the dataset to be used for training, evaluation and testing
    verbose : bool, optional
        If True, print the split ratios to the console (default is False)

    Returns
    -------
    x_train : ndarray
        The training feature dataset
    x_eval : ndarray
        The evaluation feature dataset
    x_test : ndarray
        The test feature dataset
    y_train : ndarray
        The training target dataset
    y_eval : ndarray
        The evaluation target dataset
    y_test : ndarray
        The test target dataset
    """
    if(len(splitRatio) != 3 or float(splitRatio[0]) + float(splitRatio[1]) + float(splitRatio[2]) != 1.0):
        raise ValueError("You must provide 3 split ratios and the sum of split ratios must be 1.0")

    x_train, x_temp, y_train, y_temp = train_test_split(features, targets, test_size=splitRatio[1]+splitRatio[2], random_state=42)
    x_eval, x_test, y_eval, y_test = train_test_split(x_temp, y_temp, test_size=splitRatio[2] / (splitRatio[1] + splitRatio[2]), random_state=42)

    verbose and print(f"Split ratios: {splitRatio[0]*100}% for training, {splitRatio[1]*100}% for evaluation and {splitRatio[2]*100}% for testing")
    verbose and print(f"x_train: {x_train.shape} \n x_eval: {x_eval.shape} \n x_test: {x_test.shape} \n y_train: {y_train.shape} \n y_eval: {y_eval.shape} \n y_test: {y_test.shape}")
    return x_train, x_eval, x_test, y_train, y_eval, y_test
