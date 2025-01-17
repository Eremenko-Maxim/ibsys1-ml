import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Main:
    def getDataSetFromURL(self):
        path = input("Type in the path to the dataset: ")
        if path == "":
            path = "Test00.txt"
        data = pd.read_csv(path, delimiter=' ', header=None)
        features = data.iloc[:, :-1].values
        target = data.iloc[:, -1].values
        return features, target
    
    def main(self):
        features, target = Main.getDataSetFromURL(self)
        print (f"Features: {features[0]} \n Target: {target[0]}")
        


# Erstelle eine Instanz der Klasse
main_instance = Main()

# Ruf die main-Methode auf
main_instance.main()