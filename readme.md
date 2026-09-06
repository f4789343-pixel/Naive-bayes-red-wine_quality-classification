# Naive Bayes Classification from Scratch

This project implements Naive Bayes Classification from scratch without relying on machine learning libraries for the core algorithm. The implementation is validated by comparing its performance with scikit-learn's `GaussianNB` on the Red Wine Quality dataset.

## Features

* Naive Bayes Classification implemented from scratch
* Gaussian probability distribution
* Mean and variance calculation
* Class prior probability calculation
* Conditional probability calculation
* Bayes' theorem
* Feature independence assumption
* Posterior probability calculation
* Prediction using maximum posterior probability
* Model evaluation using Accuracy
* Comparison with scikit-learn
* Confusion Matrix visualization
* Actual vs Predicted visualization

## Dataset

**Wine Quality - Red Dataset**

Source:

UCI Machine Learning Repository

The dataset contains physicochemical measurements of red wine samples and a quality score assigned to each wine.

The dataset contains **1,599 samples and 11 numerical input features**.

### Features

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

### Target

**Quality**

The target represents the quality score of the wine.

For classification, the quality values are used as class labels.

## Algorithm

Naive Bayes is a supervised machine learning algorithm based on Bayes' theorem.

The algorithm calculates the probability of each class given the observed features and predicts the class with the highest posterior probability.

### Bayes' Theorem

The posterior probability is calculated using:

```text id="b6v7xk"
P(Class | Features) =
    P(Features | Class) × P(Class)
    --------------------------------
            P(Features)
```

The algorithm compares the posterior probabilities of the possible classes and selects the class with the highest probability.

### Naive Assumption

Naive Bayes assumes that the features are conditionally independent given the class.

Therefore:

```text id="e7s9k2"
P(X | Class) =
P(x₁ | Class) × P(x₂ | Class) × ... × P(xₙ | Class)
```

Although this independence assumption is often not completely true in real-world datasets, Naive Bayes can still perform well in many classification problems.

### Gaussian Naive Bayes

Because the Wine Quality dataset contains continuous numerical features, Gaussian probability distributions are used.

For each feature and class, the mean and variance are calculated from the training data.

The Gaussian probability density is:

```text id="r5j3xq"
P(x | Class) =
1
------------------------- ×
√(2πσ²)

e^(-(x-μ)² / (2σ²))
```

where:

* `μ` = mean of the feature for the class
* `σ²` = variance of the feature for the class

### Class Prior

The prior probability of each class is calculated from its frequency in the training data:

```text id="k4w8pz"
P(Class) =
Number of samples in Class
--------------------------
Total number of samples
```

### Prediction

For each test sample, the posterior probability is calculated for every possible class.

The class with the highest posterior probability is selected as the prediction.

```text id="m9q2wr"
Prediction = argmax P(Class | Features)
```

## Implementation

* Loaded the Red Wine Quality dataset.
* Separated the input features and target variable.
* Split the dataset into training and testing data.
* Calculated the mean of each feature for each class.
* Calculated the variance of each feature for each class.
* Calculated prior probabilities for each class.
* Implemented the Gaussian probability density function.
* Calculated conditional probabilities for each feature.
* Combined feature probabilities using the Naive independence assumption.
* Applied Bayes' theorem to calculate class probabilities.
* Selected the class with the highest posterior probability.
* Implemented prediction for unseen test samples.
* Evaluated the model using classification accuracy.
* Compared the from-scratch implementation with scikit-learn's `GaussianNB`.
* Visualized classification performance using a confusion matrix.

## Results

### From Scratch

```text id="t6p8ra"
Accuracy: [Accuracy: 0.5517241379310345]
confusion Matrix: 
[[ 0  0  1  0  0  0]
 [ 1  2  5  1  0  0]
 [ 0  1 86 40  3  0]
 [ 0  3 37 66 24  2]
 [ 0  0  2 18 22  0]
 [ 0  0  0  1  4  0]]
Average of f1: 0.31674271663297954
Weighted F1: 0.54877120183125
```

### Scikit-learn

```text id="y3n6kc"
Accuracy: [0.55]
Confusion Metrix: 
[[ 0  0  1  0  0  0]
 [ 1  1  7  1  0  0]
 [ 0  1 85 41  3  0]
 [ 0  4 36 69 21  2]
 [ 0  0  2 19 21  0]
 [ 0  0  0  1  4  0]]
F1 Average: 0.29376571443406374
F1 Weighted: 0.5455353196905618
F1 Score: [0 0.125     0.651341   0.52471483 0.46153846 0]

```

The from-scratch implementation was compared with scikit-learn's `GaussianNB` using the same dataset and train-test split.

## Visualizations

### Confusion Matrix

The confusion matrix shows the number of correctly and incorrectly classified wine-quality classes.

![Confusion Matrix](confusion_matrix.png)



## Folder Structure

```text id="p3k7vz"
Naive_Bayes_Classification/
│
├── plots/
│   ├── confusion_matrix.png
│
├── winequality-red.csv
├── from_scratch.py
├── sklearn_model.py
├── visualization.py
└── README.md
```

## What I Learned

* Implemented Naive Bayes Classification from scratch.
* Learned how Bayes' theorem is used for classification.
* Learned the difference between prior, likelihood, and posterior probability.
* Learned why the Naive Bayes independence assumption is used.
* Learned how Gaussian distributions can be used for continuous numerical features.
* Learned how to calculate feature means and variances for each class.
* Learned how class prior probabilities are calculated.
* Learned how conditional probabilities are calculated.
* Learned how posterior probabilities are used to make predictions.
* Learned why the class with the highest posterior probability is selected.
* Learned the difference between Naive Bayes Classification and other classification algorithms.
* Evaluated the model using classification accuracy.
* Compared the from-scratch implementation with scikit-learn's `GaussianNB`.
* Used a confusion matrix to analyze classification performance.
