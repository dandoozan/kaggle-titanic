#todo:
#D-plot learning curve
#D-create output file
#-Submit baseline using roughfix (feautures=Pclass,Sex,Age,SibSp,Parch): r_rf_PclassSexAgeSibspParch: OOBAccuracy=0.8249, score=0.74641
#-Add Fare and Embarked as features: r_rf_PclassSexAgeSibspParchFareEmbarked, 0.8305, 0.75598
#-Fill in missing Embarkment values
#-Create Mother feature (sex=female & age>18 & parch>0 & Title != 'Miss')
#-Create Child feature (age<18)
#-add FamilySize feature (1 + parch + sibsp)
#-Create Title feature from Name
#-combine rare titles in Title
#-Fill in Age more cleverly

library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('caret') #for data-splitting
library('ggthemes') # visualization

#It seems absurd that they don't have a function to get this, but I can't
#find it, so create my own until I find a built-in way to get the OOB Error
getError = function(confusionMatrix) {
  return (sum(confusionMatrix[2:3]) / sum(confusionMatrix[1:4]))
}

getRandomForest = function(data) {
  return (randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
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
  
  plot(increments, trainErrors, type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors)
}

#I do not understand any of this code, I borrowed it from a kaggler
plotImportances = function(rf) {
  # Get importance
  importance = importance(rf)
  varImportance = data.frame(Variables = row.names(importance),
                              Importance = round(importance[ ,'MeanDecreaseGini'],2))

  # Create a rank variable based on importance
  rankImportance <- varImportance %>%
    mutate(Rank = paste0('#',dense_rank(desc(Importance))))

  # Use ggplot2 to visualize the relative importance of variables
  print(ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                             y = Importance, fill = Importance)) +
    geom_bar(stat='identity') +
    geom_text(aes(x = Variables, y = 0.5, label = Rank),
              hjust=0, vjust=0.55, size = 4, colour = 'red') +
    labs(x = 'Variables') +
    coord_flip() +
    theme_few())
}

#============= Main ================
train = read.csv('data/train.csv', stringsAsFactors=F)
test = read.csv('data/test.csv', stringsAsFactors=F)
full = bind_rows(train, test)

#remove unnecessary cols
full = subset(full, select=-c(Name, Ticket, Cabin))

#manually create factors from some cols
full$Sex = factor(full$Sex)
full$Embarked = factor(full$Embarked)

#Impute missing values in Age, Fare, Embarked
full = na.roughfix(full)

#split the data back into train and test
train = full[1:nrow(train),]
test = full[(nrow(train)+1):nrow(full),]

set.seed(343)

#plot learning curve
#plotLearningCurve(train)

rf = getRandomForest(train)

#plot importances
plotImportances(rf)


#Output solution
prediction = predict(rf, test)
print(paste('OOB Accuracy:', (1-getError(rf$confusion))))
solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file = 'out.csv', row.names=F)
