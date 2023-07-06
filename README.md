# ICR - Identifying Age-Related Conditions

The objective of this notebook is to create a model to predict whether a person has none (Class 0) or any of three medical conditions (Class 1). We will solve this problem by building a Random Forest model using TensorFlow Decision Forests on our dataset ICR - Identifying Age-Related Conditions.

## Dataset Description

The dataset consists of three files:

1. `train.csv`: The training set.
   - `Id`: Unique identifier for each observation.
   - `AB-GL`: Fifty-six anonymized health characteristics. All are numeric except for `EJ`, which is categorical.
   - `Class`: A binary target: 1 indicates the subject has been diagnosed with one of the three conditions, 0 indicates they have not.

2. `test.csv`: The test set. The goal is to predict the probability that a subject in this set belongs to each of the two classes.

3. `greeks.csv`: Supplemental metadata, only available for the training set.
   - `Alpha`: Identifies the type of age-related condition, if present.
   - `A`: No age-related condition. Corresponds to class 0.
   - `B, D, G`: The three age-related conditions. Correspond to class 1.
   - `Beta, Gamma, Delta`: Three experimental characteristics.
   - `Epsilon`: The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected.

## Libraries Used
The following libraries were imported in this notebook:
```
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
```

## Versions
The versions of TensorFlow and TensorFlow Decision Forests used in this notebook are:
- TensorFlow: v2.12.0
- TensorFlow Decision Forests: v1.3.0

## Loading the Dataset
The training dataset was loaded using the following code:
```
train_data = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
train_data.shape
```
The training dataset contains 617 rows and 58 columns.

## Dataset Exploration
The dataset exploration was performed using the following code:
```
train_data.describe()
```
The output provides descriptive statistics for each numerical column in the dataset.

The goal of the model is to predict the value of the `Class` column for each person, indicating whether they have any of the three medical conditions (Class 1) or none of them (Class 0).
