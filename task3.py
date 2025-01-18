from numpy import ndarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


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

def encode_data(self, features: ndarray, targets: ndarray) -> tuple[ndarray, ndarray]:
    """
    Encode features and targets using an ordinal encoder.

    Parameters
    ----------
    features : ndarray
        Feature dataset
    targets : ndarray
        Target dataset

    Returns
    -------
    tuple[ndarray, ndarray]
        Encoded features and targets
    """
    encoder = OrdinalEncoder()
    encoded_features = encoder.fit_transform(features)
    encoded_targets = encoder.fit_transform(targets.reshape(-1, 1))
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
    verbose and print(f"Training decision tree model with {len(X_train)} samples...")
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
    verbose and print(f"Accuracy: {accuracy:.3f}")
