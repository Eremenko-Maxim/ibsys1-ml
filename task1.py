from numpy import ndarray, unique 
import ansi_escape_codes as c

def getFeatureSize(self, features: ndarray, verbose:bool = False) -> int:
    """Return the number of features in the given dataset

    Parameters
    ----------
    features : ndarray
        The feature dataset
    verbose : bool, optional
        If True, print the result to the console (default is False)

    Returns
    -------
    int
        The number of features in the dataset
    """
    verbose and print(f"Number of features: {c.CYAN}{len(features[0])}{c.RESET}")
    return len(features[0])

def getFeatureValues(self, features: ndarray, verbose:bool = False) -> list[set]:
    """
    Get unique values for each feature in the dataset.

    This method iterates over all features and collects unique values for each feature
    into a list of sets. Each set in the list corresponds to a particular feature and
    contains all unique values observed for that feature across the dataset.

    Parameters
    ----------
    features : ndarray
        The feature dataset
    verbose : bool, optional
        If True, print the unique values for each feature to the console (default is False)

    Returns
    -------
    List[Set]
        A list where each element is a set containing unique values for a specific feature
    """
    featureValues = list()
    # Iterate over all features
    for featureIndex in range (getFeatureSize(self, features)):
        featureValues.append(set())
        # Iterate over all data points and add unique values to the set
        for feature in features:
            featureValues[featureIndex].add(feature[featureIndex])
        # Print the unique values for each feature if verbose is enabled
        verbose and print(f"Values for feature number {c.CYAN}{featureIndex+1}{c.RESET}: {c.BLUE}{featureValues[featureIndex]}{c.RESET}")
    return featureValues

def getTargetValues(self, targets: ndarray, verbose:bool = False) -> set:
    """
    Get unique values for the target variable in the dataset.

    This method collects unique values for the target variable into a set.

    Parameters
    ----------
    targets : ndarray
        The target dataset
    verbose : bool, optional
        If True, print the result to the console (default is False)

    Returns
    -------
    Set
        A set containing unique values for the target variable
    """
    targetValues = unique(targets)
    # Print the result if verbose is enabled
    verbose and print(f"Values for target: {c.BLUE}{targetValues}{c.RESET}")
    return targetValues

def getComplianceAbsoluteFrequencies(self, featureIndex: int, featureValues: ndarray, targets: ndarray, verbose: bool = False) -> dict:
    """
    Calculate the absolute frequencies of target values for each unique feature value.

    This method creates a dictionary that maps each unique feature value to another 
    dictionary, which maps each unique target value to its count (frequency) of occurrence 
    for that feature value.

    Parameters
    ----------
    featureIndex : int
        The index of the feature in the dataset for which absolute frequencies are calculated.
    featureValues : ndarray
        The array containing the values of the feature for which frequencies are calculated.
    targets : ndarray
        The array containing the target values corresponding to each feature entry.
    verbose : bool, optional
        If True, print the absolute frequencies to the console (default is False).

    Returns
    -------
    dict
        A nested dictionary where the keys are unique feature values and the values are 
        dictionaries mapping target values to their absolute frequencies.
    """
    length = len(targets)
    # Initialize the dictionary to store frequencies
    complianceNumbers = dict()
    # Set up the dictionary structure for features and targets
    for feature in {*featureValues}:
        complianceNumbers[feature] = dict()
        for target in targets:
            complianceNumbers[feature][str(target)] = 0

    # Count occurrences of each target value for each feature value
    for i in range(length):
        complianceNumbers[featureValues[i]][str(targets[i])] += 1

    # Print the frequencies if verbose is enabled
    verbose and print(f"Compliance absolute frequencies for feature number {c.CYAN}{featureIndex+1}{c.RESET}: {c.BLUE}{complianceNumbers}{c.RESET}")

    # Return the frequency dictionary
    return complianceNumbers

def getComplianceRelativeFrequencies(self, featureIndex: int, featureValues: ndarray, targets: ndarray, verbose: bool = False) -> dict:
    """
    Calculate the relative frequencies of target values for each unique feature value.

    This method computes the relative frequency (as a percentage) of each target value
    for every unique feature value. The relative frequencies are derived from the absolute
    frequencies obtained through `getComplianceAbsoluteFrequencies`.

    Parameters
    ----------
    featureIndex : int
        The index of the feature in the dataset for which relative frequencies are calculated.
    featureValues : ndarray
        The array containing the values of the feature for which frequencies are calculated.
    targets : ndarray
        The array containing the target values corresponding to each feature entry.
    verbose : bool, optional
        If True, print the relative frequencies to the console (default is False).

    Returns
    -------
    dict
        A nested dictionary where the keys are unique feature values and the values are 
        dictionaries mapping target values to their relative frequencies as percentages.
    """
    # Calculate the absolute frequencies first
    absoluteFrequencies = getComplianceAbsoluteFrequencies(self, featureIndex, featureValues, targets)

    # Calculate the relative frequency for each target value
    relativeFrequencies = {feature: {target: f"{round(absoluteFrequencies[feature][target] * 100 / len(targets), 3)}%" for target in absoluteFrequencies[feature].keys()} for feature in absoluteFrequencies.keys()}

    # Print the relative frequencies to the console if verbose is enabled
    verbose and print(f"Compliance relative frequencies for feature number {c.CYAN}{featureIndex + 1}{c.RESET}: {c.BLUE}{relativeFrequencies}{c.RESET}")

    # Return the relative frequencies
    return relativeFrequencies


    