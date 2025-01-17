from numpy import ndarray, vectorize, unique, where


def getFeatureSize(self, features: ndarray, logging:bool = False):
    logging and print(f"Number of features: {len(features[0])}")
    return len(features[0])

def getFeatureValues(self, features: ndarray, logging:bool = False):
    featureValues = list()
    for featureIndex in range (getFeatureSize(self, features)):
        featureValues.append(set())
        for feature in features:
            featureValues[featureIndex].add(feature[featureIndex])
        logging and print(f"Values for feature number {featureIndex+1}: {featureValues[featureIndex]}")
    return featureValues

def getTargetValues(self, targets: ndarray, logging:bool = False):
    targetValues = unique(targets)
    logging and print(f"Values for target: {targetValues}")
    return targetValues

def getComplianceAbsoluteFrequencies(self, featureIndex: int, featureValues: ndarray, targets: ndarray, logging:bool = False):
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
    length = len(targets)
    complianceNumbers = getComplianceAbsoluteFrequencies(self,featureIndex, featureValues, targets)
    complianceNumbers.update({feature: {target: f"{round(complianceNumbers[feature][target]*100 / length, 3)}%" for target in complianceNumbers[feature].keys()} for feature in complianceNumbers.keys()})
    logging and print(f"Compliance relative frequencies for feature number {featureIndex+1}: {complianceNumbers}")
    return complianceNumbers


    