from base64 import decode
import matplotlib.pyplot as plt
from numpy import array, ndarray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from task3 import encode_data

import ansi_escape_codes as c

import graphviz
import os
from sklearn import tree

def visualizeTree(depth: int, model: DecisionTreeClassifier, verbose: bool = False) -> None:
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
        verbose and print(f"Old file {c.MAGENTA}tree_with_depth_{depth}.png{c.RESET} has been successfully deleted.")

    verbose and print("Visualizing decision tree...")

    # Export the decision tree to DOT format for visualization
    dot_data = tree.export_graphviz(
        model, 
        out_file=None, 
        max_depth=depth, 
        filled=True, 
        rounded=True, 
        special_characters=True
    )
    # Create a Graphviz graph from the DOT data
    graph = graphviz.Source(dot_data)

    # Render the graph as a PNG image and save it
    graph.render(filename=f"tree_with_depth_{depth}", directory="./images", format="png")
    verbose and print(f"Decision tree with depth {c.CYAN}{depth}{c.RESET} saved at {c.MAGENTA}./images/tree_with_depth_{depth}.png{c.RESET}")

    # Remove the intermediate dot file created by Graphviz
    os.remove(f"./images/tree_with_depth_{depth}")

def predict(self, model: DecisionTreeClassifier, x_train: ndarray, x_eval: ndarray, x_test: ndarray, verbose: bool = False) -> tuple[ndarray, ndarray, ndarray]:
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
    verbose and print("Predicting target values...")
    x_train_encoded, _ = encode_data(self, x_train)
    x_eval_encoded, _ = encode_data(self, x_eval)
    x_test_encoded, _ = encode_data(self, x_test)

    # Use the model to predict the target values for each dataset
    y_train_pred = model.predict(x_train_encoded)
    y_val_pred = model.predict(x_eval_encoded)
    y_test_pred = model.predict(x_test_encoded)

    # Return the predicted target values as a tuple
    return y_train_pred, y_val_pred, y_test_pred

def generate_confusion_matrix(self, y_train: ndarray, y_train_pred: ndarray, y_eval: ndarray, y_eval_pred: ndarray, y_test: ndarray, y_test_pred: ndarray, verbose: bool = False) -> None:
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
    # Encode true target values
    _, y_train_encoded = encode_data(self, targets=y_train)
    _, y_eval_encoded = encode_data(self, targets=y_eval)
    _, y_test_encoded = encode_data(self, targets=y_test)

    # Calculate confusion matrices
    confusion_matrix_train = confusion_matrix(y_true=y_train_encoded, y_pred=y_train_pred)
    confusion_matrix_eval = confusion_matrix(y_true=y_eval_encoded, y_pred=y_eval_pred)
    confusion_matrix_test = confusion_matrix(y_true=y_test_encoded, y_pred=y_test_pred)

    # Print message if verbose is enabled
    verbose and print("Confusion matrices have been calculated.")

    # Save confusion matrices to files
    save_confusion_matrix(confusion_matrix_train, "training_data", verbose)
    save_confusion_matrix(confusion_matrix_eval, "evaluation_data", verbose)
    save_confusion_matrix(confusion_matrix_test, "test_data", verbose)

def save_confusion_matrix(confusion_matrix: ndarray, title: str, verbose: bool = False) -> None:
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
        verbose and print(f"Old file {c.MAGENTA}confusion_matrix{title}.png{c.RESET} has been successfully deleted.")

    # Save the confusion matrix to a file
    plt.savefig(f"./images/confusion_matrix_{title}.png")

    # Print a message indicating that the confusion matrix has been saved
    verbose and print(f"Confusion matrix saved at {c.MAGENTA}./images/confusion_matrix_{title}.png{c.RESET}")

def decode_targets(targetlist: list, verbose: bool = False) -> None:
    """
    Decode encoded target values in the provided list of arrays.

    Parameters
    ----------
    targetlist : list
        A list of arrays containing encoded target values.
    verbose : bool, optional
        If True, print a message indicating that the targets have been decoded (default is False).

    Returns
    -------
    None
    """
    # Iterate over each array in the target list and replace numerical target values with their
    # corresponding string labels ('k1' for 1 and 'k0' for 0)
    targetlist = [array(["k1" if target == 1 else "k0" for target in targets]) for targets in targetlist]
    # Print a message indicating that the targets have been decoded
    verbose and print("Targets have been decoded.")
