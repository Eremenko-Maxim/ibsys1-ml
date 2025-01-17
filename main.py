import numpy as np
import pandas as pd
from itertools import chain
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import task1

class Main:
    def getDataSetFromURL(self):
        path = input("Type in the path to the dataset: ")
        if path == "":
            path = "Test00.txt"
        data = pd.read_csv(path, delimiter=' ', header=None, )
    
        features = data.iloc[:, :-1].values
        targets = np.vectorize(lambda target: target.strip(';'))(data.iloc[:, -1].values)

        return features, targets
    
    def getValuesForfeature(self, index: int, features: np.ndarray):
        return list(chain.from_iterable(features[:, index:index+1]))

    
    def main(self):
        features, targets = Main.getDataSetFromURL(self)
        print (f"Features: {features[0]} \n Target: {targets[0]}")

        task1.getFeatureSize(self, features, True)
        task1.getFeatureValues(self, features, True)
        task1.getTargetValues(self, targets, True)

        firstFeatureIndex = 0
        firstfeatureValues = Main.getValuesForfeature(self, firstFeatureIndex, features)
        secondFeatureIndex = 1
        secondFeatureValues = Main.getValuesForfeature(self, secondFeatureIndex, features)

        task1.getComplianceAbsoluteFrequencies(self, firstFeatureIndex, firstfeatureValues, targets, True)
        task1.getComplianceAbsoluteFrequencies(self, secondFeatureIndex, secondFeatureValues, targets, True)
        task1.getComplianceRelativeFrequencies(self, firstFeatureIndex, firstfeatureValues, targets, True)
        task1.getComplianceRelativeFrequencies(self, secondFeatureIndex, secondFeatureValues, targets, True)


# Erstelle eine Instanz der Klasse
main_instance = Main()

# Ruf die main-Methode auf
main_instance.main()