#todos:
#D-plot the error rate by num trees used for the RF in ErrorRates.png
#D-plot the feature importance in FeatureImportances.png
#D-Make this a git repo
#-add one feature at a time to see how it improves my score
    #D-with baseline features (Pclass,Sex,Age,SibSp,Parch): error=0.193042, score=0.72727
    #D-Fare: error=0.186308, score=0.71770
    #D-Title: error=0.180696, score=0.70813
    #-Family size:
    #-Embarked
    #-Deck
    #-Mother
    #-Child
    #-Surname
#D-read kaggle blog post for tips on working with python
#D-plot the learning curve (I have high variance, so try smaller set of features?, or adjust params for RF, or try different ML model)
#-Do it in R to see how easy it is
#-get the same results as the kaggle blogger




import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
import pylab

#Load training data
df_train = pd.read_csv('train.csv', header=0) #columns=PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
df_test = pd.read_csv('test.csv', header=0) #columns=PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

#Store passenger ids of test data before dropping the column
pids = df_test['PassengerId'].values

#Remove unused columns
df_train = df_train.drop(['PassengerId', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_test = df_test.drop(['PassengerId', 'Ticket', 'Cabin', 'Embarked'], axis=1)

#Features to use=Survived,Pclass,Name,Sex,Age,SibSp,Parch,Fare

#----------Feature engineering----------
#convert Sex to category
df_train['Sex'] = df_train.Sex.map({ 'male': 0, 'female': 1 }).astype(int)
df_test['Sex'] = df_test.Sex.map({ 'male': 0, 'female': 1 }).astype(int)

#fill in Age with median based on gender and class
num_classes = len(df_train.Pclass.unique())
for i in range(0, 2):
    for j in range(0, num_classes):
        median_trian = df_train.Age.dropna()[(df_train.Sex == i) & (df_train.Pclass == (j + 1))].median()
        df_train.loc[(df_train.Sex == i) & (df_train.Pclass == (j + 1)) & (df_train.Age.isnull()), 'Age'] = median_trian

        median_test = df_test.Age.dropna()[(df_test.Sex == i) & (df_test.Pclass == (j + 1))].median()
        df_test.loc[(df_test.Sex == i) & (df_test.Pclass == (j + 1)) & (df_test.Age.isnull()), 'Age'] = median_test

#fill in Fare with median values based on class
for i in range(0, num_classes):
    median_train = df_train.Fare.dropna()[(df_train.Pclass == (i + 1))].median()
    df_train.loc[(df_train.Pclass == (i + 1)) & (df_train.Fare.isnull()), 'Fare'] = median_train

    median_test = df_test.Fare.dropna()[(df_test.Pclass == (i + 1))].median()
    df_test.loc[(df_test.Pclass == (i + 1)) & (df_test.Fare.isnull()), 'Fare'] = median_test

#extract Title from Name and convert it to category
df_train['Title'] = df_train.Name.str.extract(',\s(.*?)\.') \
        .replace('Ms', 'Miss').replace('Mlle', 'Miss') \
        .astype('category').cat.codes
df_test['Title'] = df_test.Name.str.extract(',\s(.*?)\.') \
        .replace('Ms', 'Miss').replace('Mlle', 'Miss') \
        .astype('category').cat.codes


#remove Name before fitting the random forest
df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)


#-----------Random Forest Classifier-----------
train_data = df_train.values
columnNames = df_train.columns.values[1::]
X = train_data[0::,1::]
y = train_data[0::,0]
test_data = df_test.values


print 'Training...'
# Create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators=100, oob_score=True)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X, y)

print 'OOB Error Rate: %2f' % (1 - forest.oob_score_)

print 'Plotting Feature Importances...'
feature_importances = forest.feature_importances_
order = feature_importances.argsort()
feature_importances.sort()
pylab.barh(range(len(columnNames)), feature_importances, tick_label=columnNames[order], align='center')
pylab.title('Feature Importances')
#pylab.savefig('FeatureImportances2.png')
pylab.show()

print 'Predicting...'
# Take the same decision trees and run it on the test data
output = forest.predict(test_data).astype(int)

print 'Writing to file out.csv...'
predictions_file = open('out.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['PassengerId', 'Survived'])
open_file_object.writerows(zip(pids, output))
predictions_file.close()

print 'Done.'
