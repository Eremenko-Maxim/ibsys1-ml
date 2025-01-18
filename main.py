import ansi_escape_codes as c
import dataSource
import task1
import task2
import task3
import task4

class Main:
    def main(self, verbose: bool = True):
        """
        Main function to execute data processing, training, evaluation, and visualization.

        Parameters
        ----------
        verbose : bool, optional
            If True, print messages indicating the progress (default is True).
        """
        # Retrieve features and targets from the dataset
        features, targets = dataSource.get_data_set_from_url(self)

        # Analyze feature and target values
        task1.getFeatureSize(self, features, verbose)
        task1.getFeatureValues(self, features, verbose)
        task1.getTargetValues(self, targets, verbose)

        # Get values for the first and second features
        firstFeatureIndex = 0
        firstfeatureValues = dataSource.get_values_for_feature(self, firstFeatureIndex, features)
        secondFeatureIndex = 1
        secondFeatureValues = dataSource.get_values_for_feature(self, secondFeatureIndex, features)

        # Calculate compliance frequencies for the features
        task1.getComplianceAbsoluteFrequencies(self, firstFeatureIndex, firstfeatureValues, targets, verbose)
        task1.getComplianceAbsoluteFrequencies(self, secondFeatureIndex, secondFeatureValues, targets, verbose)
        task1.getComplianceRelativeFrequencies(self, firstFeatureIndex, firstfeatureValues, targets, verbose)
        task1.getComplianceRelativeFrequencies(self, secondFeatureIndex, secondFeatureValues, targets, verbose)

        # Input training and evaluation ratios
        training_ratio = input(f"{c.YELLOW}Type in the {c.RED}training{c.YELLOW} ratio: {c.RESET}")
        if not training_ratio:
            training_ratio = 0.6
        else:
            training_ratio = float(training_ratio)

        eval_ratio = input(f"{c.YELLOW}Type in the {c.RED}evaluation{c.YELLOW} ratio: {c.RESET}")
        if not eval_ratio:
            eval_ratio = 0.2
        else:
            eval_ratio = float(eval_ratio)

        # Split the dataset into training, evaluation, and test sets
        x_train, x_eval, x_test, y_train, y_eval, y_test = task2.splitDataSet(
            self, features, targets, [training_ratio, eval_ratio, 1 - training_ratio - eval_ratio], verbose=verbose
        )

        # Initialize, train and evaluate the decision tree model
        model = task3.intializeModel(verbose)
        task3.train_model(self, model, x_train, y_train, verbose)
        task3.evaluate_model(self, model, x_eval, y_eval, verbose)

        # Input desired depth for the decision tree visualization
        depth = input("Type in the desired depth: ")
        if not depth:
            depth = 3
        else:
            depth = int(depth)

        # Visualize the decision tree
        task4.visualizeTree(depth, model, verbose)
        
        # Predict and generate confusion matrices for the datasets
        y_train_pred, y_eval_pred, y_test_pred = task4.predict(self, model, x_train, x_eval, x_test, verbose)
        task4.generate_confusion_matrix(self, y_train, y_train_pred, y_eval, y_eval_pred, y_test, y_test_pred, verbose)


# Create an instance of the Main class
main_instance = Main()

# Call the main method
main_instance.main()