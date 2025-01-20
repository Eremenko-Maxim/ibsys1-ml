from re import split
import ansi_escape_codes as c
import dataSource
import task1
import task2
import task3
import task4

class Main:
    def main(self):
        """
        Main function to execute data processing, training, evaluation, and visualization.
        """
        # Retrieve features and targets from the dataset
        features, targets = dataSource.get_data_set_from_url()

        # Analyze feature and target values
        task1.get_feature_size(features)
        task1.get_feature_values(features)
        task1.get_target_values(targets)

        # Calculate compliance frequencies for the features
        task1.get_compliance_absolute_frequencies(features['Feature 1'], targets)
        task1.get_compliance_relative_frequencies(features['Feature 1'], targets)
        task1.get_compliance_absolute_frequencies(features['Feature 2'], targets)
        task1.get_compliance_relative_frequencies(features['Feature 2'], targets)

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
        x_train, x_eval, x_test, y_train, y_eval, y_test = task2.splitDataSet(features, targets, [training_ratio, eval_ratio, 1 - training_ratio - eval_ratio])

        # Encode the target values of the training, evaluation, and test datasets
        y_train_encoded, y_eval_encoded, y_test_encoded = task3.code_targets(y_train, y_eval, y_test)
        # Initialize, train and evaluate the LightGBM model
        model = task3.get_trained_model(x_train, y_train_encoded)
        task3.evaluate_model(model, x_eval, y_eval_encoded)

        # Input desired depth for the decision tree visualization
        depth = input("Type in the desired depth: ")
        if not depth:
            depth = 3
        else:
            depth = int(depth)

        # Visualize the decision tree
        task4.visualizeTree(depth, model)
        
        # Predict and generate confusion matrices for the datasets
        y_train_pred, y_eval_pred, y_test_pred = task4.predict(model, x_train, x_eval, x_test)
        task4.generate_confusion_matrix(y_train_encoded, y_train_pred, y_eval_encoded, y_eval_pred, y_test_encoded, y_test_pred)


# # Create an instance of the Main class
main_instance = Main()

# # Call the main method
main_instance.main()