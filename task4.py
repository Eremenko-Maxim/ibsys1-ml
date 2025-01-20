from base64 import decode
from lightgbm import Booster, create_tree_digraph, plot_tree
import matplotlib.pyplot as plt
from numpy import argmax, array, ndarray
from pandas import DataFrame, Series
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import ansi_escape_codes as c
import logging

import graphviz
import os
from sklearn import tree

logging.basicConfig(
    filename="log.txt",
    filemode="w",
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S")

def visualizeTree(depth: int, model: Booster) -> None:
    """Visualize the decision tree

    This function renders a visual representation of the decision tree
    using Graphviz and saves it as a PNG image.

    Parameters
    ----------
    depth : int
        The maximum depth of the decision tree.
    model : DecisionTreeClassifier
        The decision tree model to be visualized.
    verbose : bool, optional
        If True, print messages indicating the progress of the visualization
        process to the console (default is False).

    Returns
    -------
    None
    """
    # Check if a previous visualization exists and remove it
    if os.path.exists(f"./images/tree_with_depth_{depth}.png"):
        os.remove(f"./images/tree_with_depth_{depth}.png")
        logging.debug(f"Old file {c.MAGENTA}tree_with_depth_{depth}.png{c.RESET} has been successfully deleted.")

    logging.debug("Visualizing decision tree...")

    graph = create_tree_digraph(model)

    # Render the graph as a PNG image and save it
    graph.render(filename=f"tree_with_depth_{depth}", directory="./images", format="png")
    logging.info(f"Decision tree with depth {c.CYAN}{depth}{c.RESET} saved at {c.MAGENTA}./images/tree_with_depth_{depth}.png{c.RESET}")

    # Remove the intermediate dot file created by Graphviz
    os.remove(f"./images/tree_with_depth_{depth}")

def predict(model: Booster, X_train: DataFrame, X_eval: DataFrame, X_test: DataFrame) -> tuple[Series, Series, Series]:
    """
    Predict the target values for the given feature datasets.

    Parameters
    ----------
    model : DecisionTreeClassifier
        The decision tree model to be used for prediction
    x_train : ndarray
        The feature dataset for which to predict the target values for training
    x_eval : ndarray
        The feature dataset for which to predict the target values for evaluation
    x_test : ndarray
        The feature dataset for which to predict the target values for testing
    verbose : bool, optional
        If True, print the message "Predicting target values..." to the console (default is False)

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        A tuple containing the predicted target values for training, evaluation and testing
    """
    # Predict the target values using the given model and feature datasets
    logging.debug("Predicting target values...")

    # Use the model to predict the target values for each dataset
    y_train_pred = argmax(model.predict(X_train), axis=1)
    y_val_pred = argmax(model.predict(X_eval), axis=1)
    y_test_pred = argmax(model.predict(X_test), axis=1)

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
    verbose : bool, optional
        If True, print messages indicating the progress (default is False).

    Returns
    -------
    None
    """
    # Calculate confusion matrices
    confusion_matrix_train = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    confusion_matrix_eval = confusion_matrix(y_true=y_eval, y_pred=y_eval_pred)
    confusion_matrix_test = confusion_matrix(y_true=y_test, y_pred=y_test_pred)

    # Print message if verbose is enabled
    logging.debug("Confusion matrices have been calculated.")

    # Save confusion matrices to files
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
    verbose : bool, optional
        If True, print the message indicating that the confusion matrix has been saved (default is False)

    Returns
    -------
    None
    """
    # Create a ConfusionMatrixDisplay object
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["k0", "k1"])

    # Plot the confusion matrix
    display.plot()

    # Set the title of the plot
    plt.title(title)

    # Check if the file exists, and if it does, delete it
    if os.path.exists(f"./images/confusion_matrix{title}.png"):
        os.remove(f"./images/confusion_matrix{title}.png")
        logging.debug(f"Old file {c.MAGENTA}confusion_matrix{title}.png{c.RESET} has been successfully deleted.")

    # Save the confusion matrix to a file
    plt.savefig(f"./images/confusion_matrix_{title}.png")

    # Print a message indicating that the confusion matrix has been saved
    logging.info(f"Confusion matrix saved at {c.MAGENTA}./images/confusion_matrix_{title}.png{c.RESET}")
