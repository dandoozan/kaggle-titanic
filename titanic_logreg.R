#todo:
#D-initial logistic regression: r_logreg: cvAccuracy=0.84831, score=0.77512
#D-plot learning curve
#D-print train accuracy
#D-set seeds to their initial values so that my results are consistent: r_logreg: cvAccuracy=0.84831, score=0.77512
#-use trainCv model for final model

#Remove all objects from the current workspace
rm(list = ls())


library(dplyr) #bind_rows
library(caret) #createDataPartition


#Globals
FILENAME = 'r_logreg2'
PROD_RUN = F


plotLearningCurve = function(data) {
  print('Plotting Learning Curve...')

  #split data into train and cv
  set.seed(837)
  index = createDataPartition(data$Survived, p=0.8, list=FALSE)
  trainCv = data[index,]
  cv = data[-index,]

  incrementSize = 5
  startIndex = 120 #I have to start at 120 so that all values of each factor are represented, otherwise predict(cv) throws an error
  increments = seq(startIndex, nrow(trainCv), incrementSize)
  numIterations = length(increments)
  trainErrors = numeric(numIterations)
  cvErrors = numeric(numIterations)
  count = 1
  for (i in increments) {
    if (i %% 100 == 0) print(paste('On training example', i))

    trainSubset = trainCv[1:i,]

    set.seed(754)
    model = glm(Survived ~., family=binomial(link='logit'), data=trainSubset)
    trainCvPrediction = predict(model, type='response')
    trainCvPrediction = ifelse(trainCvPrediction > 0.5, 1, 0)
    trainErrors[count] = mean(trainCvPrediction != trainSubset$Survived)

    cvPrediction = predict(model, newdata=subset(cv, select=-c(Survived)), type='response')
    cvPrediction = ifelse(cvPrediction > 0.5, 1, 0)
    cvErrors[count] = mean(cvPrediction != cv$Survived)
    count = count + 1
  }

  #save plot
  #png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trainErrors, type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors)
  #dev.off()


  #print final train and cv accuracies
  set.seed(754)
  model = glm(Survived ~., family=binomial(link='logit'), data=trainCv)

  #print train accuracy
  trainCvPrediction = predict(model, type='response')
  trainCvPrediction = ifelse(trainCvPrediction > 0.5, 1, 0)
  print(paste('TrainCv accuracy:', (1 - mean(trainCvPrediction != trainCv$Survived))))

  #print cv accuracy
  cvPrediction = predict(model, newdata=subset(cv, select=-c(Survived)), type='response')
  cvPrediction = ifelse(cvPrediction > 0.5, 1, 0)
  print(paste('CV accuracy:', (1 - mean(cvPrediction != cv$Survived))))
}

#============= Main ================

source('_getData.R') #this gives me train, test, and full, all fully feature engineered

#remove unused cols from train so that the logistic regression call is smoother
train = subset(train, select=-c(PassengerId, Name, Ticket, Cabin))

plotLearningCurve(train)

set.seed(754)
model = glm(Survived ~., family=binomial(link='logit'), data=train)

#print train accuracy
trainPrediction = predict(model, type = 'response')
trainPrediction = ifelse(trainPrediction > 0.5, 1, 0)
print(paste('Train accuracy:', (1 - mean(trainPrediction != train$Survived))))

if (PROD_RUN) {
  #Output solution
  prediction = predict(model, newdata=test, type='response')
  prediction = ifelse(prediction > 0.5, 1, 0)
  solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  print(paste('Writing solution to file:', outputFilename, '...'))
  write.csv(solution, file=outputFilename, row.names=F)
}

print('Done!')
