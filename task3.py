from numpy import ndarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import ansi_escape_codes as c


def intializeModel(verbose:bool = False) -> DecisionTreeClassifier:
    """
    Initialize a decision tree model with a fixed random state for reproducibility.

    The model is initialized with a random state of 42 for reproducibility.

    Parameters
    ----------
    verbose : bool, optional
        If True, print the message "Intializing decision tree model..." to the console (default is False)

    Returns
    -------
    DecisionTreeClassifier
        The initialized model
    """
    verbose and print("Intializing decision tree model...")
    return DecisionTreeClassifier(random_state=42)

def encode_data(self, features: ndarray = None, targets: ndarray = None) -> tuple[ndarray, ndarray]:
    """
    Encode the given features and/or targets using an ordinal encoder.

    Parameters
    ----------
    features : ndarray, optional
        The feature dataset to be encoded (default is None)
    targets : ndarray, optional
        The target dataset to be encoded (default is None)

    Returns
    -------
    tuple[ndarray, ndarray]
        A tuple containing the encoded features and targets
    """
    encoder = OrdinalEncoder()
    if features is not None: 
        encoded_features = encoder.fit_transform(features) 
    else: 
        encoded_features = None
    if targets is not None: 
        encoded_targets = targets.reshape(-1, 1)
    else:
        encoded_targets = None
    return encoded_features, encoded_targets


def train_model(self, model: DecisionTreeClassifier, X_train: ndarray, y_train: ndarray, verbose: bool = False) -> None:
    """
    Train the decision tree model using the provided training dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier
        The decision tree model to be trained.
    X_train : ndarray
        The training feature dataset.
    y_train : ndarray
        The training target dataset.
    verbose : bool, optional
        If True, print messages indicating the training process (default is False).
    """
    verbose and print(f"Training decision tree model with {c.CYAN}{len(X_train)}{c.RESET} samples...")
    # Encode the features and targets using an ordinal encoder
    X_encoded, y_encoded = encode_data(self, X_train, y_train)
    verbose and print(f"Encoded features and targets.")
    # Train the decision tree model
    model.fit(X_encoded, y_encoded)
    verbose and print(f"Finished training the decision tree model.")

def evaluate_model(self, model: DecisionTreeClassifier, X_eval: ndarray, y_eval: ndarray, verbose: bool = False) -> None:
    """
    Evaluate the decision tree model using the provided evaluation dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier
        The decision tree model to be evaluated.
    X_eval : ndarray
        The evaluation feature dataset.
    y_eval : ndarray
        The evaluation target dataset.
    verbose : bool, optional
        If True, print messages indicating the evaluation process (default is False).

    Returns
    -------
    None
    """
    X_encoded, y_encoded = encode_data(self, X_eval, y_eval)
    verbose and print("Evaluating model...")
    accuracy = model.score(X_encoded, y_encoded)
    verbose and print(f"Accuracy: {c.CYAN}{accuracy*100:.6f}%{c.RESET}")
