from numpy import unique 
from pandas import DataFrame, Series
import ansi_escape_codes as c
from logger_config import logger

def get_feature_size(features: DataFrame) -> int:
    """Return the number of features in the given dataset

    Parameters
    ----------
    features : DataFrame
        The feature dataset
    
    Returns
    -------
    int
        The number of features in the dataset
    """
    feature_size = len(features.columns)
    logger.info(f"Number of features: {c.CYAN}{feature_size}{c.RESET}")
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
        unique_values = set(features[feature].unique())
        feature_values.append(unique_values)
        logger.info(f"Values for {c.CYAN}{feature.title()}{c.RESET}: {c.BLUE}{unique_values}{c.RESET}")
    return feature_values

def get_target_values(targets: Series) -> set:
    """
    Get unique values for the target variable in the dataset.

    Parameters
    ----------
    targets : DataFrame
        The target dataset.
    
    Returns
    -------
    set
        A set containing unique values for the target variable.
    """
    unique_targets = set(targets.unique())
    logger.info(f"Unique target values: {c.BLUE}{unique_targets}{c.RESET}")
    return unique_targets

def get_compliance_absolute_frequencies(feature_values: DataFrame, target_values: Series) -> dict[str, dict[str, int]]:
    """
    Calculate the absolute frequencies of target values for each unique feature value.

    This function creates a dictionary that maps each unique feature value to another 
    dictionary, which maps each unique target value to its count (frequency) of occurrence 
    for that feature value.

    Parameters
    ----------
    feature : DataFrame
        The feature dataset containing the values of the feature for which frequencies are calculated.
    targets : DataFrame
        The target dataset containing the target values corresponding to each feature entry.
    
    Returns
    -------
    dict[str, dict[str, int]]
        A nested dictionary where the keys are unique feature values and the values are 
        dictionaries mapping target values to their absolute frequencies.
    """
    # Initialize the dictionary to store frequencies
    frequencies = {f_val: {str(target): 0 for target in target_values} for f_val in {*feature_values}}

    # Count occurrences of each target value for each feature value
    for f_val, target in zip(feature_values, target_values):
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
    dict[str, dict[str, str]]
        A nested dictionary where the keys are unique feature values and the values are
        dictionaries mapping target values to their relative frequencies as percentages.
    """
    # Calculate the absolute frequencies first
    absolute_frequencies = get_compliance_absolute_frequencies(feature_values, target_values)

    # Calculate the relative frequency for each target value
    relative_frequencies = {}
    for feature_value, target_freqs in absolute_frequencies.items():
        relative_frequencies[feature_value] = {}
        for target_value, count in target_freqs.items():
            relative_frequencies[feature_value][target_value] = f"{count / len(target_values) * 100:.3f}%"
    
     # Print the frequencies
    logger.info(f"Relative frequencies for {c.CYAN}{feature_values.name}{c.RESET}: {c.BLUE}{relative_frequencies}{c.RESET}")


    