from lightgbm import Booster, plot_tree
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import ansi_escape_codes as c
from logger_config import logger

import os

def visualizeTree(model: Booster) -> None:
    """Visualize the decision tree

    This function renders a visual representation of the decision tree
    using Graphviz and saves it as a PNG image.

    Parameters
    ----------
    model : Booster
        The LightGBM model to be visualized.

    Returns
    -------
    None
    """
    # Check if a previous visualization exists and remove it
    if os.path.exists(f"./images/tree.png"):
        os.remove(f"./images/tree.png")
        logger.info(f"Old file {c.MAGENTA}tree.png{c.RESET} has been successfully deleted.")

    logger.info("Visualizing decision tree...")

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, tree_index=0)

    # Save the plot as a PNG image
    plt.savefig("./images/tree.png", dpi=300)

    # Print a message indicating that the visualization has been saved
    logger.info(f"Decision tree saved at {c.MAGENTA}./images/tree.png{c.RESET}")

def predict(model: Booster, X_train: DataFrame, X_eval: DataFrame, X_test: DataFrame) -> tuple[Series, Series, Series]:
    """
    Predict the target values for the given feature datasets.

    Parameters
    ----------
    model : Booster
        The LightGBM model to be used for prediction
    X_train : DataFrame
        The feature dataset for which to predict the target values for training
    X_eval : DataFrame
        The feature dataset for which to predict the target values for evaluation
    X_test : DataFrame
        The feature dataset for which to predict the target values for testing

    Returns
    -------
    tuple[Series, Series, Series]
        A tuple containing the predicted target values for training, evaluation and testing
    """
    # Predict the target values using the given model and feature datasets
    logger.info("Predicting target values...")

    # Use the model to predict the target values for each dataset
    y_train_pred = (model.predict(X_train) > 0.5).astype(int)
    y_val_pred = (model.predict(X_eval) > 0.5).astype(int)
    y_test_pred = (model.predict(X_test) > 0.5).astype(int)

    # Return the predicted target values as a tuple
    return y_train_pred, y_val_pred, y_test_pred

def generate_confusion_matrix(y_train: Series, y_train_pred: ndarray, y_eval: Series, y_eval_pred: ndarray, y_test: Series, y_test_pred: ndarray) -> None:
    """
    Generate and save confusion matrices for training, evaluation, and test datasets.

    Parameters
    ----------
    y_train : ndarray
        The true target values for the training dataset.
    y_train_pred : ndarray
        The predicted target values for the training dataset.
    y_eval : ndarray
        The true target values for the evaluation dataset.
    y_eval_pred : ndarray
        The predicted target values for the evaluation dataset.
    y_test : ndarray
        The true target values for the test dataset.
    y_test_pred : ndarray
        The predicted target values for the test dataset.

    Returns
    -------
    None
    """
    # Calculate confusion matrices
    # The confusion matrix is a 2D array of size (n_classes, n_classes)
    # It is a square matrix with the number of classes as the number of rows and columns
    # The element at the i-th row and j-th column is the number of samples with true label i
    # that were predicted to have label j
    confusion_matrix_train = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    confusion_matrix_eval = confusion_matrix(y_true=y_eval, y_pred=y_eval_pred)
    confusion_matrix_test = confusion_matrix(y_true=y_test, y_pred=y_test_pred)

    # Print message if verbose is enabled
    logger.info("Confusion matrices have been calculated.")

    # Save confusion matrices to files
    # The confusion matrix will be saved as a PNG image
    # The filename will be the name of the dataset (e.g. training_data, evaluation_data, test_data)
    save_confusion_matrix(confusion_matrix_train, "training_data")
    save_confusion_matrix(confusion_matrix_eval, "evaluation_data")
    save_confusion_matrix(confusion_matrix_test, "test_data")

def save_confusion_matrix(confusion_matrix: ndarray, title: str) -> None:
    """
    Save a confusion matrix to a file.

    Parameters
    ----------
    confusion_matrix : ndarray
        The confusion matrix to be saved
    title : str
        The title of the confusion matrix

    Returns
    -------
    None
    """
    # Create a ConfusionMatrixDisplay object
    # This object is used to plot the confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["k0", "k1"])

    # Plot the confusion matrix
    # The confusion matrix is a square matrix with the number of classes as the number of rows and columns
    # The element at the i-th row and j-th column is the number of samples with true label i
    # that were predicted to have label j
    display.plot()

    # Set the title of the plot
    plt.title(title)

    # Check if the file exists, and if it does, delete it
    # Check if the file exists, and if it does, delete it
    # This is done to avoid overwriting an existing file
    if os.path.exists(f"./images/confusion_matrix_{title}.png"):
        os.remove(f"./images/confusion_matrix_{title}.png")
        logger.info(f"Old file {c.MAGENTA}confusion_matrix_{title}.png{c.RESET} has been successfully deleted.")

    # Save the confusion matrix to a file
    # The confusion matrix is saved as a PNG image
    plt.savefig(f"./images/confusion_matrix_{title}.png")

    # Print a message indicating that the confusion matrix has been saved
    logger.info(f"Confusion matrix saved at {c.MAGENTA}./images/confusion_matrix_{title}.png{c.RESET}")
