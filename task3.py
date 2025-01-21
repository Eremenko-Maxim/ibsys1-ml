from re import X
from typing import Tuple
from numpy import unique, argmax
from lightgbm import Booster, Dataset, train
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, classification_report

import ansi_escape_codes as c
from logger_config import logger

def code_targets(y_train: Series, y_eval: Series, y_test: Series) -> Tuple[Series, Series, Series]:
    """
    Encode the target values of the training, evaluation, and test datasets as codes.

    Parameters
    ----------
    y_train : Series
        The target dataset for training.
    y_eval : Series
        The target dataset for evaluation.
    y_test : Series
        The target dataset for testing.

    Returns
    -------
    Tuple[Series, Series, Series]
        A tuple containing the encoded target datasets for training, evaluation, and testing.
    """
    logger.info("Encoding target values...")
    y_train_encoded = y_train.cat.codes
    y_eval_encoded = y_eval.cat.codes
    y_test_encoded = y_test.cat.codes
    return y_train_encoded, y_eval_encoded, y_test_encoded

def get_trained_model(X_train: DataFrame, y_train: Series) -> Booster:
    """
    Train the decision tree model using the provided training dataset.

    This function trains a LightGBM model using the provided training dataset and
    returns the trained model.

    Parameters
    ----------
    X_train : DataFrame
        The training feature dataset.
    y_train : Series
        The training target dataset.

    Returns
    -------
    Booster
        The trained LightGBM model.
    """
    # Set up the parameters for the LightGBM model
    params = {
        # The type of the target variable
        'objective': 'binary',
        # The evaluation metric to be used
        'metric': 'binary_error',
        # Disable verbosity
        'verbosity': -1
    }

    # Print a message indicating the start of the training process
    logger.info(f"Training LightGBM model with {c.CYAN}{len(X_train)}{c.RESET} samples...")

    # Create a LightGBM dataset from the provided feature and target datasets
    dataset = Dataset(X_train, label=y_train)

    # Train the LightGBM model using the provided training dataset
    model = train(params, train_set=dataset)
    # Print a message indicating the finish of the training process if verbose is enabled
    logger.info(f"Finished training LightGBM model.")
    # Return the trained model
    return model

def evaluate_model(model: Booster, X_eval: DataFrame, y_eval: Series) -> None:
    """
    Evaluate the decision tree model using the provided evaluation dataset.

    This function evaluates the decision tree model using the provided evaluation dataset and calculates the accuracy.

    Parameters
    ----------
    model : Booster
        The decision tree model to be evaluated.
    X_eval : DataFrame
    X_eval : ndarray
        The evaluation feature dataset.
    y_eval : Series
    y_eval : ndarray
        The evaluation target dataset.

    Returns
    -------
    None
    """
    logger.info("Evaluating model...")

    # Get the predictions from the model
    y_pred_prob = model.predict(X_eval)

    # Convert the predictions to binary values
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Evaluate the model using the evaluation dataset
    accuracy = accuracy_score(y_eval, y_pred)

    # Print the accuracy to the console
    logger.info(f"Accuracy: {c.CYAN}{accuracy*100:.6f}%{c.RESET}")
