# Submission for IBSYS 1 machine learning exercise

<!-- View with [ctrl]+[shift]+[v] -->

Link to [GitHub Repository](https://github.com/Eremenko-Maxim/ibsys1-ml)

## Task 1

### a) Features and their values

| x1             | x2                              | x3                                                                                       | x4        | x5        | x6        | x7               | x8        |
|----------------|--------------------------------|------------------------------------------------------------------------------------------|-----------|-----------|-----------|------------------|-----------|
| k1, k0         | kB, kS, kP, kH, kV            | kMB, kMA, kSI, kPS, kAG, kRU, kAV, kAS, kSH, kAD, kOO, kMD, kFF, kTP, kNL, kKA, kME, kEP, kHV, kHE | k1, k0    | k1, k0    | k1, k0, k2 | k1, k2, k0, k3   | k1, k0, k2 |

| x9             | x10      | x11                                      | x12                                            | x13                                            | x14                                            | x15                                            | x16                                      |
|----------------|----------|------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------|
| k1, k0         | k1, k0   | kGB, kL, kNL, kF, kB, kPL, kD, kO, kW, kA, kI | k0, k5, k3, k9, k1, k6, k8, k7, k4, k2         | k0, k5, k3, k9, k1, k6, k7, k8, k4, k2         | k0, k5, k3, k9, k1, k7, k8, k6, k4, k2         | k0, k5, k3, k9, k1, k6, k7, k8, k4, k2         | k0, k5, k3, k9, k1, k7, k8, k6, k., k4, k2 |

### b) Frequencies of values for x1 and x2

#### For x1

| Value     | k1 total   | k1 in %    | k0 total   | k0 in %    |
|-----------|------------|------------|------------|------------|
| k1        | 80595      | 72.063%    | 5970       | 5.338%     |
| k0        | 21960      | 19.635%    | 3315       | 2.964%     |

#### For x2

| Value     | k1 total   | k1  in %   | k0 total   | k0 in %    |
|-----------|------------|------------|------------|------------|
| kB        | 10905      | 9.751%     | 735        | 0.657%     |
| kS        | 81075      | 72.492%    | 6960       | 6.223%     |
| kP        | 10485      | 9.375%     | 1590       | 1.422%     |
| kH        | 45         | 0.040%     | 0          | 0.000%     |
| kV        | 45         | 0.040%     | 0          | 0.000%     |

## Task 2

See file ./task2.py in project root directory

## Task 3

See file ./task3.py in project root directory

## Task 4

### a) Decision tree

#### LightGBM model (tree index=0)

![Decision tree with LightGBM model](./images/tree.png)

#### RandomForestClassifier model (depth=3)

![Decision tree with RandomForestClassifier model](./images/rf_tree_with_depth_3.png)

### b) Confusion Matrix

#### LightGBM model

![LightGBM Confusion Matrix for training data](./images/confusion_matrix_training_data.png)

![LightGBM Confusion Matrix for evaluation data](./images/confusion_matrix_evaluation_data.png)

![LightGBM Confusion Matrix for test data](./images/confusion_matrix_test_data.png)

#### RandomForestClassifier model

![RandomForestClassifier Confusion Matrix for training data](./images/rf_confusion_matrix_training_data.png)

![RandomForestClassifier Confusion Matrix for evaluation data](./images/rf_confusion_matrix_evaluation_data.png)

![RandomForestClassifier Confusion Matrix for test data](./images/rf_confusion_matrix_test_data.png)

## Task 5

All the features near the root of the decision tree are of great importance; these include feature No. 4, 5, and 6, to name a few.
Generally, there is a difference in the results depending on which model has been used. The RandomForestClassifier has proven to be more accurate, despite being primarily intended for ordinal data.
Using the RandomForestClassifier model, there are only false positives in small numbers and no false negatives.
With the LightGBM model, there are false positives and false negatives in roughly equal proportions.
Since the data was received all at once, split randomly, and no timestamps for the rows were provided, there is no reason to expect concept drift or shift. This is reflected in the confusion matrices.
Regarding an early warning system, according to the decision tree from the RandomForestClassifier model, the value of feature No. 4 is of great importance. Therefore, it would be useful to monitor this process step.
