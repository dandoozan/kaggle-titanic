import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pylab

#load data
df_train = pd.read_csv('train.csv', header=0)


#plot basic bar chart
print 'Plotting basic bar chart'
pylab.bar(0, df_train.Survived[df_train.Sex == 'male'].sum(), color='b', label='Male')
pylab.bar(1, df_train.Survived[df_train.Sex == 'female'].sum(), color='r', label='Female')
pylab.title('Survival By Gender')
pylab.ylabel('Num Survivors')
pylab.legend()
pylab.show()


print 'Plotting grouped bar chart'
numSurvivedByFamilySize = []
numPerishedByFamilySize = []
for i in range(7):
    numSurvivedByFamilySize.append(df_train.Survived[(df_train.Parch == i) & (df_train.Survived == 1)].count())
    numPerishedByFamilySize.append(df_train.Survived[(df_train.Parch == i) & (df_train.Survived == 0)].count())
bar_width = 0.35
index = np.arange(7) #there are 7 family sizes (0 -> 6)
pylab.bar(index, numSurvivedByFamilySize, width=bar_width, color='b', label='Survived')
pylab.bar(index + bar_width, numPerishedByFamilySize, width=bar_width, color='r', label='Perished')
pylab.title('Survival By Family Size')
pylab.xlabel('Family size')
pylab.ylabel('Num people')
pylab.legend()
pylab.show()


print 'Plotting horizontal bar chart'
train_data = df_train.values
columnNames = df_train.columns.values[1::]
X = train_data[0::,1::]
y = train_data[0::,0]
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(X, y)
feature_importances = forest.feature_importances_
pylab.barh(range(len(columnNames)), feature_importances, tick_label=columnNames, align='center')
pylab.title('Feature Importances')
pylab.show()