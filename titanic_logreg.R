#todo:
#D-initial logistic regression: r_logreg: cvAccuracy=0.84831, score=0.77512
#D-plot learning curve
#D-print train accuracy
#D-set seeds to their initial values so that my results are consistent: r_logreg: cvAccuracy=0.84831, score=0.77512
#-use trainCv model for final model


library(dplyr) # data manipulation
library(mice) # imputation
library(randomForest) # classification algorithm
library(caret) #for data-splitting
library(ggplot2) #visualization
library(ggthemes) # visualization
library(pscl) #for pR2
library(caret) #for data splitting
library(ROCR) #for ROC plot

#Globals
FILENAME = 'r_logreg2'
PROD_RUN = T


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

train = read.csv('data/train.csv', stringsAsFactors=F, na.strings=c(''))
test = read.csv('data/test.csv', stringsAsFactors=F, na.strings=c(''))
full = bind_rows(train, test)

#manually create factors from some cols
full$Sex = factor(full$Sex)
full$Embarked = factor(full$Embarked)
full$PassengerId = factor(full$PassengerId)
full$Pclass = factor(full$Pclass)

#create Title feature from Name
full$Title = gsub('(.*, )|(\\..*)', '', full$Name)
full$Title[full$Title == 'Mlle' | full$Title == 'Ms'] = 'Miss'
full$Title[full$Title == 'Mme'] = 'Mrs'
full$Title[full$Title %in% c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')] = 'Rare_Title'
full$Title = factor(full$Title)

#create FamilySize feature
familySize = (1 + full$SibSp + full$Parch)

#discretize FamilySize: 1=Single, 2-4=Small, >5=Large (these values were arrived at by
#manually examining the data; families of size 2-4 seem to have a better chance of
#survival than singletons or large families)
full$FamilySizeDiscrete = cut(familySize, breaks=c(0, 1, 4, 1000), labels=c('Single', 'Small', 'Large'))


#impute missing values in Age, Fare, Embarked
print('Imputing missing values...')
set.seed(129)
mice_imp = mice(subset(full, select=-c(PassengerId, Name, Ticket, Cabin, Survived)), method='rf', printFlag=F)
mice_output = complete(mice_imp)
full$Age = mice_output$Age
#full$Age = na.roughfix(full$Age) #tbx
full$Fare[1044] = 8.05
full$Embarked[c(62, 830)] = 'C'


#discretize Fare
full$FareDiscrete = cut(full$Fare, c(-1, 50, 10000), labels=c('Low', 'High'))

#create Child feature
full$Child[full$Age < 18] = 'Child'
full$Child[full$Age >= 18] = 'Adult'
full$Child = factor(full$Child)

#create AgeDiscrete feature: 0-6=Young, 7-12=Middle, 13-18=Teen, >18=Adult
full$AgeDiscrete = cut(full$Age, breaks=c(0, 6, 12, 18, 1000), labels=c('Young', 'Middle', 'Teen', 'Adult'))

#create Mother feature
full$Mother = 'NotMother'
full$Mother[full$Sex == 'female' & full$Age > 18 & full$Parch > 0 & full$Title != 'Miss'] = 'Mother'
full$Mother = factor(full$Mother)


#split the data back into train and test
train = full[1:nrow(train),]
test = full[(nrow(train)+1):nrow(full),]


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
  prediction = predict(model, newdata=subset(test, select=-c(Survived)), type='response')
  prediction = ifelse(prediction > 0.5, 1, 0)
  solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  print(paste('Writing solution to file:', outputFilename, '...'))
  write.csv(solution, file=outputFilename, row.names=F)
}