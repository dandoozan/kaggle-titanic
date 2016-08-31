import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier


#Read in and manipulate training data
df = pd.read_csv('train.csv', header=0)
df['Gender'] = df.Sex.map( { 'female': 0, 'male': 1 } ).astype(int)

#todo: fill the 2 na values with something other than 'C' (I chose C at random)
df['Embarked'] = df.Embarked.fillna('C').map({ 'C': 0, 'Q': 1, 'S': 2 }).astype(int)

#fill in age with median based on gender and class
num_classes = len(df.Pclass.unique())
for i in range(0, 2):
    for j in range(0, num_classes):
        median = df.Age.dropna()[(df.Gender == i) & (df.Pclass == (j + 1))].median()
        df.loc[(df.Gender == i) & (df.Pclass == (j + 1)) & (df.Age.isnull()), 'Age'] = median


#drop unneeded columns
df = df.drop(['Name', 'Cabin', 'Ticket', 'PassengerId', 'Sex'], axis=1)

#convert df to array
train_data = df.values




#Do the same things for test data
test_df = pd.read_csv('test.csv', header=0)

test_df['Gender'] = test_df.Sex.map( { 'female': 0, 'male': 1 } ).astype(int)

#todo: fill the 2 na values with something other than 'C' (I chose C at random)
test_df['Embarked'] = test_df.Embarked.fillna('C').map({ 'C': 0, 'Q': 1, 'S': 2 }).astype(int)

#fill in Fare with median values based on class
num_classes = len(test_df.Pclass.unique())
for i in range(0, num_classes):
    median = test_df.Fare.dropna()[(test_df.Pclass == (i + 1))].median()
    test_df.loc[(test_df.Pclass == (i + 1)) & (test_df.Fare.isnull()), 'Fare'] = median


#fill in age with median based on gender and class
for i in range(0, 2):
    for j in range(0, num_classes):
        median = test_df.Age.dropna()[(test_df.Gender == i) & (test_df.Pclass == (j + 1))].median()
        test_df.loc[(test_df.Gender == i) & (test_df.Pclass == (j + 1)) & (test_df.Age.isnull()), 'Age'] = median

#Store passenger ids of test data before dropping the column
ids = test_df['PassengerId'].values

#drop unneeded columns
test_df = test_df.drop(['Name', 'Cabin', 'Ticket', 'PassengerId', 'Sex'], axis=1)

#convert df to array
test_data = test_df.values





#---- Do the random forest here-----#
print 'Training...'
# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

print 'Predicting...'
# Take the same decision trees and run it on the test data
output = forest.predict(test_data).astype(int)
predictions_file = open('randomforest.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['PassengerId', 'Survived'])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'




