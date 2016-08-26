import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pylab

#Load training data
df = pd.read_csv('train.csv', header=0) #columns=PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
df_train, df_cv = train_test_split(df, test_size = 0.2)

#Remove unused columns
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_cv = df_cv.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

#Features to use=Survived,Pclass,Sex,Age,SibSp,Parch,Fare

#----------Feature engineering----------
#convert Sex to category
df_train['Sex'] = df_train.Sex.map({ 'male': 0, 'female': 1 }).astype(int)
df_cv['Sex'] = df_cv.Sex.map({ 'male': 0, 'female': 1 }).astype(int)

#fill in Age with median based on gender and class
num_classes = len(df_train.Pclass.unique())
for i in range(0, 2):
    for j in range(0, num_classes):
        median_trian = df_train.Age.dropna()[(df_train.Sex == i) & (df_train.Pclass == (j + 1))].median()
        df_train.loc[(df_train.Sex == i) & (df_train.Pclass == (j + 1)) & (df_train.Age.isnull()), 'Age'] = median_trian

        median_cv = df_cv.Age.dropna()[(df_cv.Sex == i) & (df_cv.Pclass == (j + 1))].median()
        df_cv.loc[(df_cv.Sex == i) & (df_cv.Pclass == (j + 1)) & (df_cv.Age.isnull()), 'Age'] = median_cv

#fill in Fare with median values based on class
for i in range(0, num_classes):
    median_train = df_train.Fare.dropna()[(df_train.Pclass == (i + 1))].median()
    df_train.loc[(df_train.Pclass == (i + 1)) & (df_train.Fare.isnull()), 'Fare'] = median_train

    median_cv = df_cv.Fare.dropna()[(df_cv.Pclass == (i + 1))].median()
    df_cv.loc[(df_cv.Pclass == (i + 1)) & (df_cv.Fare.isnull()), 'Fare'] = median_cv


#-----------Random Forest Classifier-----------
train_data = df_train.values
cv_data = df_cv.values


trainErrors = []
cvErrors = []
indices = []

forest = RandomForestClassifier(n_estimators=100, oob_score=True)
for i in range(5, len(train_data), 5):
    print 'Fitting test set size=%d' % i
    train_data_subset = train_data[:i]
    X_train = train_data_subset[0::,1::]
    y_train = train_data_subset[0::,0]
    forest = forest.fit(X_train, y_train)
    trainScore = forest.score(X_train, y_train)
    trainErrors.append(1 - trainScore)

    X_cv = cv_data[0::,1::]
    y_cv = cv_data[0::,0]
    cvScore = forest.score(X_cv, y_cv)
    cvErrors.append(1 - cvScore)

    indices.append(i)

pylab.plot(indices, trainErrors, color='b', label='Train')
pylab.plot(indices, cvErrors, color='r', label='CV')
pylab.ylim(ymin=0, ymax=1)
pylab.title('Learning Curve')
pylab.xlabel('Training Set Size')
pylab.ylabel('Error (Accuracy)')
pylab.legend()
pylab.savefig('LearningCurve.png')
pylab.show()

print 'Done.'
