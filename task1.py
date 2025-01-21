from numpy import unique 
from pandas import DataFrame, Series
import ansi_escape_codes as c
from logger_config import logger

def get_feature_size(features: DataFrame) -> int:
    """
    Calculate and return the number of features in the given dataset.

    This function determines the number of columns (features) present in the 
    provided DataFrame and logs the count for informational purposes.

    Parameters
    ----------
    features : DataFrame
        The feature dataset for which the size is to be calculated.
    
    Returns
    -------
    int
        The number of features (columns) in the dataset.
    """
    # Calculate the number of features by counting the columns in the DataFrame
    feature_size = len(features.columns)

    # Log the number of features using the logger
    logger.info(f"Number of features: {c.CYAN}{feature_size}{c.RESET}")

    # Return the calculated number of features
    return feature_size

def get_feature_values(features: DataFrame) -> list[set]:
    """
    Get unique values for each feature in the dataset.

    This method iterates over all features and collects unique values for each feature
    into a list of sets. Each set in the list corresponds to a particular feature and
    contains all unique values observed for that feature across the dataset.

    Parameters
    ----------
    features : DataFrame
        The feature dataset

    Returns
    -------
    List[Set]
        A list where each element is a set containing unique values for a specific feature
    """
    feature_values = []
    for feature_index, feature in enumerate(features.columns):
        # Get unique values for the current feature
        unique_values = set(features[feature].unique())
        # Append the set of unique values to the list
        feature_values.append(unique_values)
        # Log the set of unique values using the logger
        logger.info(
            f"Values for {c.CYAN}{feature.title()}{c.RESET}: {c.BLUE}{unique_values}{c.RESET}"
        )
    # Return the list of sets
    return feature_values

def get_target_values(targets: Series) -> set:
    """
    Retrieve unique values for the target variable.

    This function extracts unique values from the provided target dataset, logs them,
    and returns the unique values as a set.

    Parameters
    ----------
    targets : Series
        The target dataset from which to extract unique values.
    
    Returns
    -------
    set
        A set containing unique values from the target variable.
    """
    # Extract unique values from the target dataset
    unique_targets = set(targets.unique())

    # Log the unique values for informational purposes
    logger.info(f"Unique target values: {c.BLUE}{unique_targets}{c.RESET}")

    # Return the set of unique target values
    return unique_targets

def get_compliance_absolute_frequencies(feature_values: DataFrame, target_values: Series) -> dict[str, dict[str, int]]:
    """
    Calculate the absolute frequencies of target values for each unique feature value.

    This function creates a dictionary that maps each unique feature value to another 
    dictionary, which maps each unique target value to its count (frequency) of occurrence 
    for that feature value.

    Parameters
    ----------
    feature_values : DataFrame
        The feature dataset containing the values of the feature for which frequencies are calculated.
    target_values : Series
        The target dataset containing the target values corresponding to each feature entry.
    
    Returns
    -------
    dict[str, dict[str, int]]
        A nested dictionary where the keys are unique feature values and the values are 
        dictionaries mapping target values to their absolute frequencies.
    """
    # Initialize the dictionary to store frequencies
    # The outer dictionary maps each unique feature value to another dictionary
    # The inner dictionary maps each unique target value to its count (frequency) of occurrence
    frequencies = {f_val: {str(target): 0 for target in target_values} for f_val in {*feature_values}}
    
    # Iterate over the feature and target datasets
    # Count occurrences of each target value for each feature value
    for f_val, target in zip(feature_values, target_values):
        # Increment the count for the current target value and feature value
        frequencies[f_val][str(target)] += 1
    
    # Print the frequencies
    logger.info(f"Absolute frequencies for {c.CYAN}{feature_values.name}{c.RESET}: {c.BLUE}{frequencies}{c.RESET}")

    return frequencies

def get_compliance_frequencies(feature_values: Series, target_values: Series) -> None:
    """
    Calculate the relative frequencies of target values for each unique feature value.

    Parameters
    ----------
    feature_values : Series
        The feature values for which frequencies are calculated.
    target_values : Series
        The target values corresponding to each feature entry.

    Returns
    -------
    None
    """
    # Calculate the absolute frequencies first
    absolute_frequencies = get_compliance_absolute_frequencies(feature_values, target_values)

    # Calculate the relative frequency for each target value
    relative_frequencies = {}
    for feature_value, target_freqs in absolute_frequencies.items():
        # Initialize the dictionary to store relative frequencies for this feature value
        relative_frequencies[feature_value] = {}

        # Calculate the relative frequency for each target value
        for target_value, count in target_freqs.items():
            # Calculate the relative frequency as a percentage
            relative_frequency = f"{count / len(target_values) * 100:.3f}%"
            # Store the relative frequency in the dictionary
            relative_frequencies[feature_value][target_value] = relative_frequency
    
    # Print the frequencies
    logger.info(f"Relative frequencies for {c.CYAN}{feature_values.name}{c.RESET}: {c.BLUE}{relative_frequencies}{c.RESET}")


    