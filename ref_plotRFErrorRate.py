import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
import pylab

#Load training data
df_train = pd.read_csv('train.csv', header=0) #columns=PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

#Remove unused columns
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], axis=1)

#Features to use=Survived,Pclass,Sex,Age,SibSp,Parch

#----------Feature engineering----------
#1. convert sex to 0s and 1s
df_train['Sex'] = df_train.Sex.map({ 'male': 0, 'female': 1 }).astype(int)

#fill in age with median based on gender and class
num_classes = len(df_train.Pclass.unique())
for i in range(0, 2):
    for j in range(0, num_classes):
        median_trian = df_train.Age.dropna()[(df_train.Sex == i) & (df_train.Pclass == (j + 1))].median()
        df_train.loc[(df_train.Sex == i) & (df_train.Pclass == (j + 1)) & (df_train.Age.isnull()), 'Age'] = median_trian

train_data = df_train.values
X = train_data[0::,1::]
y = train_data[0::,0]



print 'Plotting error rates...'
#plot the OOB error rate by num trees
min_num_trees = 1
max_num_trees = 500

indices = []
error_rates = []
forest = RandomForestClassifier(oob_score=True)
for i in range(min_num_trees, max_num_trees + 1):
    if i%100 == 0:
        print 'On numTrees=%d' % i
    forest.set_params(n_estimators=i)
    forest = forest.fit(X, y)
    oob_error = 1 - forest.oob_score_
    indices.append(i)
    error_rates.append(oob_error)

pylab.plot(indices, error_rates)
pylab.ylim(ymin=0)
pylab.xlabel('num trees')
pylab.ylabel('OOB error rate')
pylab.title('Error Rates')
#pylab.savefig('ErrorRates.png')
pylab.show()
