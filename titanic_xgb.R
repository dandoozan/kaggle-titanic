#todo:
#-do simple xgboost: r_xgb: final train-error=0.017957, score=0.72727
#-plot train-error
#-plot feature importances
#-use xgboost cv feature
#-plot learning curve
#-do more complex xgboost
#-perhaps adjust the threshold for 1 vs 0 (from 0.5 to 0.7 or something?)


#Remove all objects from the current workspace
rm(list = ls())

library(xgboost)

#Globals
FILENAME = 'r_xgb'
PROD_RUN = T

#============= Main ================

#get data: this gives me train, test, and full, all fully feature engineered
source('_getData.R')


#one hot encode factor variables, and convert to matrix
set.seed(634)
trainSparseMatrix = sparse.model.matrix(Survived~.-1, data=subset(train, select=-c(PassengerId, Name, Ticket, Cabin)))
testSparseMatrix = sparse.model.matrix(~.-1, data=subset(test, select=-c(PassengerId, Name, Ticket, Cabin)))

#create model
print('Creating Model...')
set.seed(754)
model = xgboost(data=trainSparseMatrix,
                 label=train$Survived,
                 max.depth=4,
                 eta=1,
                 nthread=2,
                 nround=100,
                 verbose=0,
                 objective='binary:logistic')

if (PROD_RUN) {
  #Output solution
  prediction = predict(model, testSparseMatrix)
  prediction = as.numeric(prediction > 0.5)
  solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  print(paste('Writing solution to file:', outputFilename, '...'))
  write.csv(solution, file=outputFilename, row.names=F)
}

print('Done!')
