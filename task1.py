from numpy import ndarray, vectorize, unique, where


def getFeatureSize(self, features: ndarray, logging:bool = False):
    """Return the number of features in the given dataset

    Parameters
    ----------
    features : ndarray
        The feature dataset
    logging : bool, optional
        If True, print the result to the console (default is False)

    Returns
    -------
    int
        The number of features in the dataset
    """

    logging and print(f"Number of features: {len(features[0])}")
    return len(features[0])

def getFeatureValues(self, features: ndarray, logging:bool = False):
    """
    Get unique values for each feature in the dataset.

    This method iterates over all features and collects unique values for each feature
    into a list of sets. Each set in the list corresponds to a particular feature and 
    contains all unique values observed for that feature across the dataset.

    Parameters
    ----------
    features : ndarray
        The feature dataset
    logging : bool, optional
        If True, print the unique values for each feature to the console (default is False)

    Returns
    -------
    List[Set]
        A list where each element is a set containing unique values for a specific feature
    """

    featureValues = list()
    for featureIndex in range (getFeatureSize(self, features)):
        featureValues.append(set())
        for feature in features:
            featureValues[featureIndex].add(feature[featureIndex])
        logging and print(f"Values for feature number {featureIndex+1}: {featureValues[featureIndex]}")
    return featureValues

def getTargetValues(self, targets: ndarray, logging:bool = False):
    """
    Get unique values for the target variable in the dataset.

    This method collects unique values for the target variable into a set.

    Parameters
    ----------
    targets : ndarray
        The target dataset
    logging : bool, optional
        If True, print the result to the console (default is False)

    Returns
    -------
    Set
        A set containing unique values for the target variable
    """
    targetValues = unique(targets)
    logging and print(f"Values for target: {targetValues}")
    return targetValues

def getComplianceAbsoluteFrequencies(self, featureIndex: int, featureValues: ndarray, targets: ndarray, logging:bool = False):
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
    logging : bool, optional
        If True, print the absolute frequencies to the console (default is False).

    Returns
    -------
    dict
        A nested dictionary where the keys are unique feature values and the values are 
        dictionaries mapping target values to their absolute frequencies.
    """

    length = len(targets)
    complianceNumbers = dict()
    for feature in {*featureValues}:
        complianceNumbers[feature] = dict()
        for target in targets:
            complianceNumbers[feature][str(target)] = 0

    for i in range(length):
        complianceNumbers[featureValues[i]][str(targets[i])] += 1
    logging and print(f"Compliance absolute frequencies for feature number {featureIndex+1}: {complianceNumbers}")
    return complianceNumbers

def getComplianceRelativeFrequencies(self, featureIndex: int, featureValues: ndarray, targets: ndarray, logging:bool = False):
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
    logging : bool, optional
        If True, print the relative frequencies to the console (default is False).

    Returns
    -------
    dict
        A nested dictionary where the keys are unique feature values and the values are 
        dictionaries mapping target values to their relative frequencies as percentages.
    """

    length = len(targets)
    complianceNumbers = getComplianceAbsoluteFrequencies(self,featureIndex, featureValues, targets)
    complianceNumbers.update({feature: {target: f"{round(complianceNumbers[feature][target]*100 / length, 3)}%" for target in complianceNumbers[feature].keys()} for feature in complianceNumbers.keys()})
    logging and print(f"Compliance relative frequencies for feature number {featureIndex+1}: {complianceNumbers}")
    return complianceNumbers


    