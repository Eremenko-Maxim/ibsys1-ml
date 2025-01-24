# Submission for IBSYS 1 machine learning exercise

<!-- View with [ctrl]+[shift]+[v] -->

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

See ./task2.py

## Task 3

See ./task3.py

## Task 4

### a) Decision tree

![Decision tree with LightGBM model](./images/tree.png)

### b) Confusion Matrix

![Confusion Matrix for training data](./images/confusion_matrix_training_data.png)

![Confusion Matrix for evaluation data](./images/confusion_matrix_evaluation_data.png)

![Confusion Matrix for test data](./images/confusion_matrix_test_data.png)
