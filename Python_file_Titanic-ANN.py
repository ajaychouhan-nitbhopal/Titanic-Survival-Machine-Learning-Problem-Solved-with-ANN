# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing raw train and test set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv') 
final = test['PassengerId']
final


#Preprocessing

# Checking missing values in train set
train.isnull().sum()
# Checking missing values in test set
test.isnull().sum()
# Replacing "" to NaN in train
train.replace('', np.nan, inplace = True)
train.head(5)
# Replacing "" to NaN in test
test.replace('', np.nan, inplace = True)
test.head(5)
# Checking missing values in train set
train.isnull().sum()
# To see how many values are present in each label of Embarked column
train['Embarked'].value_counts()
# It is shwon that S is mode so we will replace all NaN values of column Embarked with 'S'
# replacing the missing 'Embarked' values by the most frequent 'S'
train["Embarked"].replace(np.nan, "S", inplace=True)
test["Embarked"].replace(np.nan, "S", inplace=True)
# Finding average age of all passengers train set
avg_age_train = train['Age'].astype('float').mean(axis=0)
print("Average Age in train set:", avg_age_train)
# Finding average age of all passengers in test set
avg_age_test = test['Age'].astype('float').mean(axis=0)
print("Average Age in test set:", avg_age_test)
# Replacing NaN values of Age column by the average age
train['Age'].replace(np.nan, avg_age_train, inplace=True)
test['Age'].replace(np.nan, avg_age_test, inplace=True)
# Dropping cabin columns in train and test
train.drop(['Cabin','Name','Ticket','PassengerId'], axis=1, inplace = True)
test.drop(['Cabin','Name','Ticket','PassengerId'], axis=1, inplace = True)
train.head()
# Replacing rows of fare column in test set where values are NaN with average
# finding average age of all passengers in test set
avg_fare_test = test['Fare'].astype('float').mean(axis=0)
print("Average Fare in test set:", avg_fare_test)
# Replacing NaN values of Fare column by the average Fare
test['Fare'].replace(np.nan, avg_fare_test, inplace=True)
test.head()
# Checking datatypes of all values in columns of train set
train.dtypes
# Replacing categorical values of Sex column by dummy variables
dummy_variable_1_train = pd.get_dummies(train["Sex"])
dummy_variable_1_test = pd.get_dummies(test["Sex"])
# merge data frame "df" and "dummy_variable_1" 
train = pd.concat([train, dummy_variable_1_train], axis=1)
test = pd.concat([test, dummy_variable_1_test], axis=1)
# drop original column "Sex" from "train and test"
train.drop("Sex", axis = 1, inplace=True)
test.drop("Sex", axis = 1, inplace=True)
# Replacing categorical values of Embarked column by dummy variables
dummy_variable_2_train = pd.get_dummies(train["Embarked"])
dummy_variable_2_test = pd.get_dummies(test["Embarked"])
# merge data frame "df" and "dummy_variable_1" 
train = pd.concat([train, dummy_variable_2_train], axis=1)
test = pd.concat([test, dummy_variable_2_test], axis=1)
# drop original column "fuel-type" from "df"
train.drop("Embarked", axis = 1, inplace=True)
test.drop("Embarked", axis = 1, inplace=True)



#Splitting the dataset into the Training set and Test set
X_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values
X_test = test.iloc[:,:]
X_test.isnull().sum()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Further splitting X_train into X_train_train and X_train_test for gaining more insights
from sklearn.model_selection import train_test_split
X_train_train,X_train_test,y_train_train,y_train_test = train_test_split(X_train,y_train, test_size = 0.25, random_state = 42)


# Applying Artificial Neural Network
# intitializing ann
import tensorflow as tf
ann = tf.keras.models.Sequential()

# input layer and hidden layer 1
ann.add(tf.keras.layers.Dense(units = 14, activation = 'relu'))

#Hidden layer 2
ann.add(tf.keras.layers.Dense(units = 14, activation = 'relu'))

# Output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Optimizing by Adam Optimizer, 'binary_crossentropy' is selected as Loss function
# accuracy is selected as metrics
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting X_train_train and y_train_train in our model
ann.fit(X_train_train, y_train_train, batch_size = 8, epochs = 100)

#Predicting the Train set results
y_pred_train = ann.predict(X_train_train)
# Threshold is set to 0.5
y_pred_train = (y_pred_train > 0.5)
Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm1 = confusion_matrix(y_train_train, y_pred_train)
print("Confusion Matrix:", cm1)
print("Accuracy of the model is", accuracy_score(y_train_train, y_pred_train))

from sklearn.metrics import classification_report, confusion_matrix
import itertools

# Defining th function for plotting Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Train Set Accuracy and Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train_train, y_pred_train, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_train_train, y_pred_train))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not Survived(0)','Survived(1)'], normalize= False,  title='Confusion matrix')


# Predicting the Test set results
In [47]:
y_pred = ann.predict(X_train_test)
In [48]:
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_train_test.reshape(len(y_train_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_train_test, y_pred)
print(cm)
accuracy_score(y_train_test, y_pred)

# Test Set Accuracy and Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train_test, y_pred, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_train_test, y_pred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not Survived(0)','Survived(1)'],normalize= False,  title='Confusion matrix')

# Prediction of X_test
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.reshape(len(y_pred),1)
passangerid = np.asarray(final)
passangerid = passangerid.reshape(len(passangerid),1)
# Final result for the test set
final_array = np.concatenate((passangerid, y_pred),axis=1)
final_array

# NOTE - By submitting this result to Kaggle's Machine Learning Problem I got 0.78468 Accuracy