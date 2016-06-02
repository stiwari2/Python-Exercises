## Let us simplify the problem ##
# read the training data into pandas and print it out
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import csv as csv

data = pd.read_csv('C:/Projects/01 Data Science Training/04 Python Modeling/titanic_train.csv')
data_test = pd.read_csv('C:/Projects/01 Data Science Training/04 Python Modeling/titanic_test.csv')

### Data Exploration ##
print data.groupby('Survived').count()
print data.groupby('Survived').mean()

## Gender
print data.groupby(['Sex','Survived']).count().transpose()
print data[['PassengerId','Survived','Sex']].groupby(['Sex','Survived']).count().transpose()

## Pclass
print data[['PassengerId','Survived','Pclass']].groupby(['Pclass']).count().transpose()
print data[['PassengerId','Survived','Pclass']].groupby(['Pclass','Survived']).count().transpose()
print data[data['Survived'] ==1][['PassengerId','Pclass']].groupby(['Pclass']).count().transpose()

##SibSp
print data[['PassengerId','SibSp']].groupby(['SibSp']).count().transpose()
print data[['PassengerId','Survived','SibSp']].groupby(['SibSp','Survived']).count().transpose()

##Parch
print data[['PassengerId','Parch']].groupby(['Parch']).count().transpose()
print data[['PassengerId','Survived','Parch']].groupby(['Parch','Survived']).count().transpose()

##Embarked
print data[['PassengerId','Embarked']].groupby(['Embarked']).count().transpose()
print data[['PassengerId','Survived','Embarked']].groupby(['Embarked','Survived']).count().transpose()

''''
data[data['Age'].isnull()]
data[data['Age'] > 60]
import pylab as P
data['Age'].hist()
P.show()
data['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

#data_feature['Gender'] = data_feature['Sex'].map(lambda x: x[0].upper())
data_feature['Sex'].replace('female',0,inplace=True)
data_feature['Sex'].replace('male',1,inplace=True)
data_feature['Age'].replace('NaN',data_feature['Age'].median(),inplace=True)
data_feature['Fare']
'''

###########################################################################################################################################
################################################ Feature Engineering on Train data#########################################################
###########################################################################################################################################

############################## Train Data ##############################
data_feature = data

#****************** 1. GENDER ******************#
data_feature['Gender'] = data_feature['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#****************** 2. AGE ******************#
# Filling in the median age of people in every class for every gender into the arrary.
median_ages = np.zeros((2,3))
median_ages
data_feature['Pclass'].unique()
data_feature['Gender'].unique()

##Creating an array with the the median age for each gender class
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = data_feature[(data_feature['Gender'] == i) & (data_feature['Pclass'] == j+1)]['Age'].dropna().median()
median_ages

##Filling in the missing age using the array
data_feature['AgeFill'] = data_feature['Age']
data_feature.head()
for i in range(0, 2):
    for j in range(0, 3):
        data_feature.loc[(data_feature.Age.isnull()) & (data_feature.Gender == i) & (data_feature.Pclass == j+1),'AgeFill'] = median_ages[i,j]
data_feature.describe()

#****************** 3. FAMILY ******************#
data_feature['FamilySize'] = data_feature['SibSp'] + data_feature['Parch']

#****************** 4. CLASS AND AGE ******************#
data_feature['ClassAge'] = data_feature['AgeFill']* data_feature['Pclass']

#****************** 5. FARE  ********************#
data_feature["Fare"] = data_feature["Fare"].fillna(data_feature["Fare"].mean())

#****************** 6. EMBARKED ******************#
##Checking null values in embarked. Assigning binary form to data for calculation purpose
data_feature[['Embarked']].isnull().any()
data_feature[['Embarked','PassengerId']].groupby(['Embarked']).count().transpose()
data_feature["Embarked"] = data_feature["Embarked"].fillna("S")
data_feature['Embarked'] = data_feature['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#****************** 7. TITLE  ******************#
##Reducing the titles to just 4 types- Mr, Miss, Mrs, Master
new_titles = []
for i in range(len(data_feature)):
    data = data_feature.iloc[i]
    title = data['Name'][data['Name'].find(',') + 2:data['Name'].find('.')]
    new_titles.append(title)

pd.DataFrame(new_titles).groupby(0).size()
data_feature['Title_Value'] = np.array(new_titles)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "the Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2,"Dona":10}
for k,v in title_mapping.items():
    data_feature['Title_Value'][data_feature['Title_Value'] == k] = v
pd.DataFrame(data_feature['Title_Value']).groupby('Title_Value').size()

## Convert string to categorical variables ##
data_feature['Ticket'] = data_feature['Ticket'].astype('category')
data_feature['Cabin'] = data_feature['Cabin'].astype('category')

###########################################################################################################################################
################################################ Feature Engineering on Test data#########################################################
###########################################################################################################################################

############################## Test Data ##############################
data_feature_test = data_test

#****************** 1. GENDER ******************#
data_feature_test['Gender'] = data_feature_test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#****************** 2. AGE ******************#
# Filling in the median age of people in every class for every gender into the arrary.
median_ages = np.zeros((2,3))
median_ages
data_feature_test['Pclass'].unique()
data_feature_test['Gender'].unique()

##Creating an array with the the median age for each gender class
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = data_feature_test[(data_feature_test['Gender'] == i) & (data_feature_test['Pclass'] == j+1)]['Age'].dropna().median()
median_ages

##Filling in the missing age using the array
data_feature_test['AgeFill'] = data_feature_test['Age']
data_feature_test.head()
for i in range(0, 2):
    for j in range(0, 3):
        data_feature_test.loc[(data_feature_test.Age.isnull()) & (data_feature_test.Gender == i) & (data_feature_test.Pclass == j+1),'AgeFill'] = median_ages[i,j]
data_feature_test.describe()

#****************** 3. FAMILY ******************#
data_feature_test['FamilySize'] = data_feature_test['SibSp'] + data_feature_test['Parch']

#****************** 4. CLASS AND AGE ******************#
data_feature_test['ClassAge'] = data_feature_test['AgeFill']* data_feature_test['Pclass']

#****************** 5. FARE  ********************#
data_feature_test["Fare"] = data_feature_test["Fare"].fillna(data_feature_test["Fare"].mean())

#****************** 6. EMBARKED ******************#
##Checking null values in embarked. Assigning binary form to data for calculation purpose
data_feature_test[['Embarked']].isnull().any()
data_feature_test[['Embarked','PassengerId']].groupby(['Embarked']).count().transpose()
data_feature_test["Embarked"] = data_feature_test["Embarked"].fillna("S")
data_feature_test['Embarked'] = data_feature_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#****************** 7. TITLE  ******************#
new_titles = []
for i in range(len(data_feature_test)):
    data = data_feature_test.iloc[i]
    title = data['Name'][data['Name'].find(',') + 2:data['Name'].find('.')]
    new_titles.append(title)

pd.DataFrame(new_titles).groupby(0).size()
data_feature_test['Title_Value'] = np.array(new_titles)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "the Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2,"Dona":10}
for k,v in title_mapping.items():
    data_feature_test['Title_Value'][data_feature_test['Title_Value'] == k] = v
pd.DataFrame(data_feature_test['Title_Value']).groupby('Title_Value').size()


## Convert string to categorical variables ##
data_feature_test['Ticket'] = data_feature_test['Ticket'].astype('category')
data_feature_test['Cabin'] = data_feature_test['Cabin'].astype('category')

######################################################################################################################
################################################ Modelling ###########################################################
######################################################################################################################

####################### Linear Regression ######################
# The columns we'll use to predict the target
predictors = ["Pclass", "Gender", "AgeFill","SibSp", "Parch", "Fare", "Embarked",
              "FamilySize","ClassAge","Title_Value"]

# Initialize our algorithm class
alg = LinearRegression()

## Training using K fold Cross Validation- Step by Step
kf = KFold(data_feature.shape[0], n_folds=10, random_state=10)
predictions = []
for train, test in kf:
    train_predictors = (data_feature[predictors].iloc[train,:])# The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds..
    train_target = (data_feature["Survived"].iloc[train])   # The target we're using to train the algorithm
    alg.fit(train_predictors, train_target)     # Training the algorithm using the predictors and target.
    test_predictions = alg.predict(data_feature[predictors].iloc[test,:])   # We can now make predictions on the test fold
    predictions.append(test_predictions)

# We concatenate the three numpy arrays of into single predictions on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1 ## # Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == data_feature["Survived"]]) / len(predictions)
print "Accuracy of the model = ",accuracy
## 0.803591470258

####################### Logistic Regression ######################
# The columns we'll use to predict the target
predictors = ["Pclass", "Gender", "AgeFill","SibSp", "Parch", "Fare", "Embarked",
              "FamilySize","ClassAge","Title_Value"]

## Training using K fold Cross Validation- One Step
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, data_feature[predictors], data_feature["Survived"], cv=10)
print(scores.mean()) # Take the mean of the scores (because we have one for each fold)
##0.806961468619

## Prediction on Test Set
alg = LogisticRegression(random_state=1)
alg.fit(data_feature[predictors], data_feature["Survived"])# Train the algorithm using all the training data
predictions = alg.predict(data_feature_test[predictors]) # Make predictions using the test set.

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission_logistic = pd.DataFrame({"PassengerId": data_feature_test["PassengerId"],
                                     "Survived": predictions })
submission_logistic.to_csv("output_logistic.csv", index=False) ##0.77518


######################### Random Forest - Run ############################
# The columns we'll use to predict the target
predictors = ["Pclass", "Gender", "AgeFill","SibSp", "Parch", "Fare", "Embarked",
              "FamilySize","ClassAge","Title_Value"]

## Training using K fold Cross Validation
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores = cross_validation.cross_val_score(alg, data_feature[predictors], data_feature["Survived"], cv=10)
print(scores.mean()) ##0.811506355692
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, data_feature[predictors], data_feature["Survived"], cv=3)
print(scores.mean()) ##0.833894500561

## Prediction on Test Set
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
alg.fit(data_feature[predictors], data_feature["Survived"])
predictions = alg.predict(data_feature_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission_rf = pd.DataFrame({"PassengerId": data_feature_test["PassengerId"],
                                     "Survived": predictions })
submission_rf.to_csv("output_rf5.csv", index=False) ##0.75598
## Accuracy reduced !!


############################ Feature Selection ################################
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)
selector.fit(data_feature[predictors], data_feature["Survived"])
scores = -np.log10(selector.pvalues_) # Get the raw p-values for each feature, and transform from p-values into scores

import matplotlib
import matplotlib.pyplot as plt
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
## The plot shows that Pclass,Gender,Fare,ClassAge and Title_Value are important predictors

#### Random Forest on Selected Features - 1 ####
predictors = ["Pclass", "Gender", "ClassAge","Fare", "Title_Value"]
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
scores = cross_validation.cross_val_score(alg, data_feature[predictors], data_feature["Survived"], cv=3)
print(scores.mean()) ##0.836139169473

## Prediction on Test Set
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
alg.fit(data_feature[predictors], data_feature["Survived"])
predictions = alg.predict(data_feature_test[predictors])
submission_rf = pd.DataFrame({"PassengerId": data_feature_test["PassengerId"],
                                     "Survived": predictions })
submission_rf.to_csv("output_rf6.csv", index=False) ##0.76077
## Still lesser than logistic ##

################################### Ensemble ###################################
from sklearn.ensemble import GradientBoostingClassifier
predictors_1 = ["Pclass", "Gender", "AgeFill", "SibSp", "Parch", "Fare", "Embarked",
              "FamilySize", "ClassAge", "Title_Value"]
predictors_2 = ["Pclass", "Gender", "ClassAge","Fare", "Title_Value"]

algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors_1],
              [LogisticRegression(random_state=1),predictors_2 ]]

#************ Training on train dataset ************
kf = KFold(data_feature.shape[0], n_folds=10, random_state=1)
predictions = []
for train, test in kf:
    train_target = data_feature["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:  # Make predictions for each algorithm on each fold
        alg.fit(data_feature[predictors].iloc[train,:], train_target) # Fit the algorithm on the training data.
        test_predictions = alg.predict_proba(data_feature[predictors].iloc[test,:].astype(float))[:,1] # Select and predict on the test fold.# The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2     # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions[test_predictions <= .5] = 0    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == data_feature["Survived"]]) / len(predictions)# Compute accuracy by comparing to the training data.
print "Accuracy of the model = ",accuracy ##0.827160493827
## n_estimators=25 - 0.827160493827
## n_estimators=200 -0.836139169473

#************ Prediction on test dataset ************
algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors_1],
              [LogisticRegression(random_state=1), predictors_2]]
full_predictions = []
for alg, predictors in algorithms:
    alg.fit(data_feature[predictors], data_feature["Survived"])# Fit the algorithm using the full training data.
    predictions = alg.predict_proba(data_feature_test[predictors].astype(float))[:,1]# Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission_ensemble = pd.DataFrame({"PassengerId": data_feature_test["PassengerId"],
                                     "Survived": predictions})
submission_ensemble.to_csv("output_ensemble1.csv", index=False)
##n_estimators=25 - 0.78469
## n_estimators=200 - 0.76555

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 1 + full_predictions[1]*3) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission_ensemble = pd.DataFrame({"PassengerId": data_feature_test["PassengerId"],
                                     "Survived": predictions})
submission_ensemble.to_csv("output_ensemble2.csv", index=False) ##0.76555

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 2 + full_predictions[1]*2) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
submission_ensemble = pd.DataFrame({"PassengerId": data_feature_test["PassengerId"],
                                     "Survived": predictions})
submission_ensemble.to_csv("output_ensemble3.csv", index=False) ##0.77033