#todo:
#D-do simple xgboost: r_xgb: finalTrainError=0.017957, finalTestError=?, score=0.72727
#D-use xgb.cv to tune hyperparameters: r_xgb_hyperparam: 0.166388, 0.172898, 0.78947
#D-plot learning curve
#D-figure out what's wrong with the learning curve
#D-plot feature importances
#-do more complex xgboost (read: http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html)
#-perhaps adjust the threshold for 1 vs 0 (from 0.5 to 0.7 or something?)
#-maybe implement early stopping

#Remove all objects from the current workspace
rm(list = ls())

library(xgboost)
library(Matrix) #sparse.model.matrix
library(caret) #createDataPartition
library(Ckmeans.1d.dp) #xgb.plot.importance

#Globals
FILENAME = 'r_xgb_hyperparam'
PROD_RUN = F
THRESHOLD = 0.5

#================= Functions ===================

plotCV = function(data, label, params, nrounds, verbose) {
  print('Plotting Error Rates...')
  set.seed(754)
  cvRes <- xgb.cv(data=data,
                  label=label,
                  params=params,
                  nfold=5,
                  nrounds=nrounds,
                  verbose=verbose)
  print(paste('Final Train Error:', cvRes$train.error.mean[length(cvRes$train.error.mean)]))
  print(paste('Final Test Error:', cvRes$test.error.mean[length(cvRes$test.error.mean)]))

  #plot train error vs test error
  plot(cvRes$train.error.mean, type='l', ylim = c(min(cvRes$train.error.mean, cvRes$test.error.mean), max(cvRes$train.error.mean, cvRes$test.error.mean)), col='blue', main='Train Error vs. Test Error', xlab='Num Rounds', ylab='Error')
  lines(cvRes$test.error.mean, col='red')
  legend(x='topright', legend=c('train', 'test'), fill=c('blue', 'red'), inset=0.02, text.width=15)
}

plotLearningCurve = function(data, params, nrounds) {
  print('Plotting Learning Curve...')

  #split data into train and cv
  set.seed(837)
  trainIndex = createDataPartition(data$Survived, p=0.8, list=FALSE)
  train = data[trainIndex,]
  cv = data[-trainIndex,]

  #one hot encode train and cv
  set.seed(634)
  trainSparseMatrix = sparse.model.matrix(Survived~.-1, data=subset(train, select=-c(PassengerId, Name, Ticket, Cabin)))
  cvSparseMatrix = sparse.model.matrix(~.-1, data=subset(cv, select=-c(Survived, PassengerId, Name, Ticket, Cabin)))

  incrementSize = 5
  increments = seq(incrementSize, nrow(train), incrementSize)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)

  count = 1
  for (i in increments) {
    if (i %% 100 == 0) print(paste('On training example', i))
    trainSubset = trainSparseMatrix[1:i,]
    labelSubset = train$Survived[1:i]

    set.seed(754)
    model = xgboost(data=trainSubset,
                    label=labelSubset,
                    params=params,
                    nrounds=nrounds,
                    verbose=0)

    trainPrediction = predict(model, trainSubset)
    trainPrediction = as.numeric(trainPrediction > THRESHOLD)
    trainErrors[count] = mean(trainPrediction != labelSubset)

    cvPrediction = predict(model, cvSparseMatrix)
    cvPrediction = as.numeric(cvPrediction > THRESHOLD)
    cvErrors[count] = mean(cvPrediction != cv$Survived)

    count = count + 1
  }

  #png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trainErrors, col='red', type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors, col='blue')
  legend('topright', legend=c('train', 'cv'), fill=c('red', 'blue'), inset=.02, text.width=100)
  #dev.off()
}

plotFeatureImportances = function(model, data) {
  print('Plotting Feature Importances...')

  importances = xgb.importance(feature_names=data@Dimnames[[2]], model=model)
  print(xgb.plot.importance(importance_matrix=importances))
}

#============= Main ================

#get data: this gives me train, test, and full, all fully feature engineered
source('_getData.R')

#one hot encode factor variables, and convert to matrix
set.seed(634)
trainSparseMatrix = sparse.model.matrix(Survived~.-1, data=subset(train, select=-c(PassengerId, Name, Ticket, Cabin)))
testSparseMatrix = sparse.model.matrix(~.-1, data=subset(test, select=-c(PassengerId, Name, Ticket, Cabin)))


#set hyper params
nrounds = 100
verbose = 0
xgbParams <- list(
    'objective'='binary:logistic',

    #range=[0,1], default=0.3, toTry=0.01,0.015,0.025,0.05,0.1
    'eta'=0.001, #learning rate. Lower value=less overfitting

    #range=[1,∞], default=6, toTry=3,5,7,9,12,15,17,25
    'max_depth'=3, #Lower value=less overfitting

    #range=[0,∞], default=1, toTry=1,3,5,7
    'min_child_weight'=1,  #Larger value=less overfitting

    #range=(0,1], default=1, toTry=0.6,0.7,0.8,0.9,1.0
    'subsample'=1, #ratio of sample of data to use for each instance (eg. 0.5=50% of data). Lower value=less overfitting

    #range=(0,1], default=1, toTry=0.6,0.7,0.8,0.9,1.0
    'colsample_bytree'=0.6 #ratio of cols (features) to use in each tree
  )

#plot cv
#plotCV(trainSparseMatrix, train$Survived, xgbParams, nrounds, verbose)
#plotLearningCurve(train, xgbParams, nrounds)

#create model
print('Creating Model...')
set.seed(754)
model = xgboost(data=trainSparseMatrix,
                  label=train$Survived,
                  params=xgbParams,
                  nrounds=nrounds,
                  verbose=verbose)

#plot feature importances
plotFeatureImportances(model, trainSparseMatrix)

if (PROD_RUN) {
  #Output solution
  prediction = predict(model, testSparseMatrix)
  prediction = as.numeric(prediction > THRESHOLD)
  solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  print(paste('Writing solution to file:', outputFilename, '...'))
  write.csv(solution, file=outputFilename, row.names=F)
}

print('Done!')
