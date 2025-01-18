import numpy as np
import pandas as pd
from itertools import chain

import task1
import task2

class Main:
    def getDataSetFromURL(self):
        """
        Retrieve a dataset from a given URL and split it into features and targets.

        If no URL is provided, the default dataset 'Test00.txt' is used.

        Parameters
        ----------
        None

        Returns
        -------
        features : ndarray
            The feature dataset
        targets : ndarray
            The target dataset
        """
        path = input("Type in the path to the dataset: ")
        if path == "":
            path = "Test00.txt"
        data = pd.read_csv(path, delimiter=' ', header=None, )
    
        features = data.iloc[:, :-1].values
        targets = np.vectorize(lambda target: target.strip(';'))(data.iloc[:, -1].values)

        return features, targets
    
    def getValuesForfeature(self, index: int, features: np.ndarray):
        """
        Return a list of all values for a given feature index in the dataset.

        Parameters
        ----------
        index : int
            The index of the feature in the dataset
        features : ndarray
            The feature dataset

        Returns
        -------
        list
            A list containing all values for the feature at the given index
        """
        
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

        training_ratio = float(input("Type in the training ratio: "))
        eval_ratio = float(input("Type in the evaluation ratio: "))
        x_train, x_eval, x_test, y_train, y_eval, y_test = task2.splitDataSet(self, features, targets, [training_ratio, eval_ratio, 1 - training_ratio - eval_ratio], True)


# Erstelle eine Instanz der Klasse
main_instance = Main()

# Ruf die main-Methode auf
main_instance.main()