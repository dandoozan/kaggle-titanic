#todo:
#D-do simple xgboost: r_xgb: finalTrainError=0.017957, finalCvError=?, score=0.72727
#D-use xgb.cv to tune hyperparameters: r_xgb_hyperparam: 0.166388, 0.172898, 0.78947
#D-plot learning curve
#D-figure out what's wrong with the learning curve
#D-plot feature importances
#D-try xgboost without additional features: r_xgb_simple: 0.168359, 0.194152, 0.76077
#D-use xgb.train instead of xgboost: r_xgb_simple2: 0.168359, 0.194152, 0.76077
#-try different booster options
#-perhaps adjust the threshold for 1 vs 0 (from 0.5 to 0.7 or something?)
#-maybe implement early stopping


#Remove all objects from the current workspace
rm(list = ls())

library(xgboost)
library(Matrix) #sparse.model.matrix
library(caret) #createDataPartition
library(Ckmeans.1d.dp) #xgb.plot.importance

#Globals
FILENAME = 'r_xgb_simple2'
PROD_RUN = T
THRESHOLD = 0.5

#================= Functions ===================

plotCVErrorRates = function(cvRes) {
  print('Plotting CV Error Rates...')
  png(paste0('ErrorRates_', FILENAME, '.png'), width=500, height=350)
  plot(cvRes$train.error.mean, type='l', ylim = c(min(cvRes$train.error.mean, cvRes$test.error.mean), max(cvRes$train.error.mean, cvRes$test.error.mean)), col='blue', main='Train Error vs. CV Error', xlab='Num Rounds', ylab='Error')
  lines(cvRes$test.error.mean, col='red')
  legend(x='topright', legend=c('train', 'cv'), fill=c('blue', 'red'), inset=0.02, text.width=15)
  dev.off()
}

plotLearningCurve = function(data, params, nrounds) {
  print('Plotting Learning Curve...')

  #split data into train and cv
  set.seed(837)
  trainIndex = createDataPartition(data$Survived, p=0.8, list=FALSE)
  train = data[trainIndex,]
  cv = data[-trainIndex,]

  #one hot encode cv
  set.seed(634)
  cvSparseMatrix = sparse.model.matrix(~.-1, data=subset(cv, select=-c(Survived, PassengerId, Name, Ticket, Cabin)))
  cvDMatrix <- xgb.DMatrix(data=cvSparseMatrix, label=cv$Survived)

  incrementSize = 5
  increments = seq(incrementSize, nrow(train), incrementSize)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)

  count = 1
  for (i in increments) {
    if (i %% 100 == 0) print(paste('On training example', i))
    trainSubset = train[1:i,]

    #one hot encode train subset
    set.seed(634)
    trainSparseMatrix = sparse.model.matrix(Survived~.-1, data=subset(trainSubset, select=-c(PassengerId, Name, Ticket, Cabin)))
    trainDMatrix <- xgb.DMatrix(data=trainSparseMatrix, label=trainSubset$Survived)

    set.seed(754)
    model = xgb.train(data=trainDMatrix,
                    params=params,
                    nrounds=nrounds,
                    verbose=0)

    trainPrediction = predict(model, trainDMatrix)
    trainPrediction = as.numeric(trainPrediction > THRESHOLD)
    trainErrors[count] = mean(trainPrediction != getinfo(trainDMatrix, 'label'))

    cvPrediction = predict(model, cvDMatrix)
    cvPrediction = as.numeric(cvPrediction > THRESHOLD)
    cvErrors[count] = mean(cvPrediction != getinfo(cvDMatrix, 'label'))

    count = count + 1
  }

  png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trainErrors, col='blue', type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors, col='red')
  legend('topright', legend=c('train', 'cv'), fill=c('blue', 'red'), inset=.02, text.width=100)
  dev.off()
}

plotFeatureImportances = function(model, dataAsSparseMatrix, save=FALSE) {
  print('Plotting Feature Importances...')

  importances = xgb.importance(feature_names=dataAsSparseMatrix@Dimnames[[2]], model=model)
  if (save) png(paste0('Importances_', FILENAME, '.png'), width=500, height=350)
  print(xgb.plot.importance(importance_matrix=importances))
  if (save) dev.off()
}

#============= Main ================

source('_getData.R')
data = getData(simple=TRUE)
train = data$train
test = data$test
full = data$full

#one hot encode factor variables, and convert to matrix
set.seed(634)
trainSparseMatrix = sparse.model.matrix(Survived~.-1, data=subset(train, select=-c(PassengerId, Name, Ticket, Cabin)))
trainDMatrix = xgb.DMatrix(data=trainSparseMatrix, label=train$Survived)
testSparseMatrix = sparse.model.matrix(~.-1, data=subset(test, select=-c(PassengerId, Name, Ticket, Cabin)))

#set hyper params
nrounds = 100
xgbParams = list(
    #range=[0,1], default=0.3, toTry=0.01,0.015,0.025,0.05,0.1
    'eta'=0.001, #learning rate. Lower value=less overfitting, but increase nrounds when lowering eta

    #range=[1,∞], default=6, toTry=3,5,7,9,12,15,17,25
    'max_depth'=3, #Lower value=less overfitting

    #range=[0,∞], default=1, toTry=1,3,5,7
    'min_child_weight'=1, #Larger value=less overfitting

    #range=(0,1], default=1, toTry=0.6,0.7,0.8,0.9,1.0
    'subsample'=1, #ratio of sample of data to use for each instance (eg. 0.5=50% of data). Lower value=less overfitting

    #range=(0,1], default=1, toTry=0.6,0.7,0.8,0.9,1.0
    'colsample_bytree'=0.6, #ratio of cols (features) to use in each tree

    #values=gbtree|gblinear|dart, default=gbtree, toTry=gbtree,gblinear
    #'booster'='gblinear', #gbtree/dart=tree based, gblinear=linear function. Remove eta when using gblinear

    'objective'='binary:logistic'
  )

#run cv
set.seed(754)
cvRes <- xgb.cv(data=trainDMatrix,
                params=xgbParams,
                nfold=5,
                nrounds=nrounds,
                verbose=0)
print(paste('Final Train Error:', cvRes$train.error.mean[length(cvRes$train.error.mean)]))
print(paste('Final CV Error:', cvRes$test.error.mean[length(cvRes$test.error.mean)]))

if (PROD_RUN) {
  #plots
  plotCVErrorRates(cvRes)
  plotLearningCurve(train, xgbParams, nrounds)
}

#create model
print('Creating Model...')
set.seed(754)
model = xgb.train(data=trainDMatrix,
                  params=xgbParams,
                  nrounds=nrounds,
                  verbose=0)

#plot feature importances
plotFeatureImportances(model, trainSparseMatrix, save=PROD_RUN)

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
