#todo:
#D-plot learning curve with basic features: Pclass, Sex, Age, Sibsp, Parch
#D-create output file
#-Impute missing values
  #D-Age by median: 0.74641
  #-Age by more clever means
#-Create Child feature (age<18)
#-Create Mother feature (sex=female & age>18 & parch>0 & Title != 'Miss')
#-add FamilySize feature (1 + parch + sibsp)
#-Create Title feature from Name
#-combine rare titles in Title
#-Fill in missing Embarkment values


library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('caret') #for data-splitting

#It seems absurd that they don't have a function to get this, but I can't
#find it, so create my own until I find a built-in way to get the OOB Error
getError = function(confusionMatrix) {
  return (sum(confusionMatrix[2:3]) / sum(confusionMatrix[1:4]))
}

getRandomForest = function(data) {
  return (randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch, 
                      ntree = 100,
                      data = data))
}

plotLearningCurve = function(data) {
  print('Plotting Learning Curve...')
  
  #split data into train and cv
  trainIndex = createDataPartition(data$Survived, p=0.8, list=FALSE)
  train = data[trainIndex,]
  cv = data[-trainIndex,]
  
  incrementSize = 5
  increments = seq(incrementSize, nrow(train), incrementSize)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)
  
  #tbx
  start = proc.time()
  
  count = 1
  for (i in increments) {
    if (i %% 100 == 0) {
      print(paste('On training example', i))
    }
    trainSubset = train[1:i,]
    rf = getRandomForest(trainSubset)
    trainPrediction = predict(rf, trainSubset)
    trainErrors[count] = getError(table(trainSubset$Survived, trainPrediction))
    
    cvPrediction = predict(rf, cv)
    cvErrors[count] = getError(table(cv$Survived, cvPrediction))
    count = count + 1
  }
  
  #tbx
  print(proc.time() - start)
  
  plot(increments, trainErrors, type='l', ylim = c(0, 1), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors)
}



#============= Main ================
train = read.csv('data/train.csv')
test = read.csv('data/test.csv')

#Impute missing Age values
train = na.roughfix(train)
test = na.roughfix(test)

set.seed(624)

#plotLearningCurve(train)


#Output solution
rf = getRandomForest(train)
prediction = predict(rf, test)
solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file = 'out.csv', row.names=F)
