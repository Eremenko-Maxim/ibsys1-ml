import numpy as np
import pandas as pd
from itertools import chain

import dataSource
import task1
import task2
import task3

class Main:
    def main(self, verbose:bool = True):
        features, targets = dataSource.get_data_set_from_url(self)
        verbose and print (f"First row: \n  Features: {features[0]} \n  Target: {targets[0]}")

        task1.getFeatureSize(self, features, verbose)
        task1.getFeatureValues(self, features, verbose)
        task1.getTargetValues(self, targets, verbose)

        firstFeatureIndex = 0
        firstfeatureValues = dataSource.get_values_for_feature(self, firstFeatureIndex, features)
        secondFeatureIndex = 1
        secondFeatureValues = dataSource.get_values_for_feature(self, secondFeatureIndex, features)

        task1.getComplianceAbsoluteFrequencies(self, firstFeatureIndex, firstfeatureValues, targets, verbose)
        task1.getComplianceAbsoluteFrequencies(self, secondFeatureIndex, secondFeatureValues, targets, verbose)
        task1.getComplianceRelativeFrequencies(self, firstFeatureIndex, firstfeatureValues, targets, verbose)
        task1.getComplianceRelativeFrequencies(self, secondFeatureIndex, secondFeatureValues, targets, verbose)

        training_ratio = float(input("Type in the training ratio: "))
        eval_ratio = float(input("Type in the evaluation ratio: "))
        x_train, x_eval, x_test, y_train, y_eval, y_test = task2.splitDataSet(self, features, targets, [training_ratio, eval_ratio, 1 - training_ratio - eval_ratio], verbose=verbose)

        model = task3.intializeModel(verbose)
        task3.train_model(self, model, x_train, y_train, verbose)
        task3.evaluate_model(self, model, x_eval, y_eval, verbose)


# Create an instance of the Main class
main_instance = Main()

# Call the main method
main_instance.main()