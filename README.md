# K-Nearest-Neighbour-Project

This project demonstrates the application of the k-Nearest Neighbors (KNN) algorithm for classification. The project involves loading a dataset, preprocessing the features, exploring the data through visualizations, building a KNN model, and evaluating its performance.

## Table of Contents

- [Getting Started](#getting-started)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Getting Started

To get started, ensure you have the required dependencies installed. You can install them using the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```
## Data
The dataset for this project is stored in the file KNN_Project_Data.csv. It is loaded into a Pandas DataFrame, and the first five rows of the dataset are displayed using the data.head(5) function.

## Data Preprocessing
To prepare the features for the KNN algorithm, standardization is applied using the StandardScaler from scikit-learn. Standardization ensures that each feature has a mean of 0 and a standard deviation of 1, which is essential for distance-based algorithms like KNN.

## Exploratory Data Analysis (EDA)
An exploratory data analysis is conducted to visually explore relationships between different features. Seaborn's pairplot is utilized, with the hue parameter set to 'TARGET CLASS' to distinguish between classes.

## Model Building
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. A KNN classifier is instantiated with n_neighbors=1 and is then trained on the training set.

## Model Building
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. A KNN classifier is instantiated with n_neighbors=1 and is then trained on the training set.

## Model Evaluation
The performance of the KNN model is evaluated using common classification metrics. The confusion matrix and classification report are printed to the console for a detailed assessment.
```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

```
## Conclusion
This README provides a comprehensive overview of the KNN project, covering data loading, preprocessing, exploratory data analysis, model building, and evaluation. Feel free to customize this README based on your specific project details.

For more details on the project, code documentation can be found in the Python script.
