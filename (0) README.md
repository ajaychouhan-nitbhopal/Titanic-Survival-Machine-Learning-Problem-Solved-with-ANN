# Titanic-Survival-Machine-Learning-Problem-Solved-with-ANN-
![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fajaychouhan-nitbhopal%2FTitanic-Survival-Machine-Learning-Problem-Solved-with-ANN)

This is the code for "Titanic: Machine Learning from Disaster" on Titanic dataset (offered by Kaggle.com) by Ajay Chouhan.

## Overview

This is the code of Artificial Neural Networks (ANN) which is implemented on Titanic dataset.

Train set and Test set are given, but survival status of passengers is only given in Train set. We have to find the survival status of Test dataset.

NOTE- I splitted Train dataset into train and test set by name convention as Train_train and Train_test (with Test_size of 0.25). Further I got X_train_train, X_train_test, y_train_train, y_train_test arrays.

You can find abovementioned dataset [here](https://www.kaggle.com/c/titanic/data)

## Dataset Overview given by Kaggle.com
The data has been split into two groups:

1. training set (train.csv)
2. test set (test.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

### Data Dictionaries
Open Data_Dicionaries.jpg in Repository.

### Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

## Dependencies

[numpy](https://numpy.org/)

[pandas](https://pandas.pydata.org/)

[scikt-learn](https://scikit-learn.org/stable/)

[matplotlib](https://matplotlib.org/)

[tensorflow](https://www.tensorflow.org/)

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)

## Usage
1. Titanic-ANN.ipynb is Jupyter Notebook which contains ANN model.
2. Train.csv is CSV file which contains dataset of 892 Passengers with information of their survival.
3. Test.csv is CSV file which contains dataset of 418 Passengers without information of their survival. We have to find out their survival status.
4. Python_file_Titanic-ANN.py is Python file which contains python code of the classifier.
5. Confusion matrix of Test set.JPG is Jpeg file which contains Confusion Matrix of Test set (which is part of Train dataset).
6. Data_Dicionaries.jpg is Jpeg file which contains overview of Data Dictionaries.

Install jupyter [here](https://jupyter.org/install).

## Credits
This problem is taken from [Kaggle.com](https://www.kaggle.com/c/titanic/)
