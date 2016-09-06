#todo:
#D-plot learning curve
#D-create output file
#D-Submit baseline using roughfix (feautures=Pclass,Sex,Age,SibSp,Parch): r_rf_PclassSexAgeSibspParch: OOBAccuracy=0.8249, score=0.74641
#D-Add Fare and Embarked as features: r_rf_PclassSexAgeSibspParchFareEmbarked, 0.8305, 0.75598
#D-Add FamilySize feature (1 + parch + sibsp): r_rf_PclassSexAgeSibspParchFareEmbarkedFamilysize: 0.8227, 0.77990
#D-Create Child feature (age<18): r_rf_PclassSexAgeSibspParchFareEmbarkedFamilysizeChild: 0.83053, 0.77033
#D-Discretize Age into Young (0-6), Middle (7-12), Teen (13-18), Adult (19-):r_rf_PclassSexAgeSibspParchFareEmbarkedFamilysizeChildAgediscrete: 0.83389, 0.77990
#D-Create Title feature from Name: r_rf_+Title: 0.83053, 0.77990
#D-Combine rare titles in Title: r_rf_+Title2: 0.83165, 0.76555
#D-Create Mother feature (sex=female & age>18 & parch>0 & Title!='Miss'): r_rf_+Mother: 0.83053, 0.77033
#D-Discretize family size into Single, Small, Large: r_rf_+FamilySizeDiscrete: 0.83389, 0.78947
#D-Impute missing values in Age, Fare, and Embarked using MICE: r_rf_Mice, 0.83389, 0.78947
#D-Remove Child: r_rf_-Child: 0.82155, 0.78947
#D-use Title when imputing values: r_rf_Mice2: 0.83389, 0.77512
#D-Remove FamilySize: r_rf_-FamilySize: 0.83389, 0.77990
#D-Remove Mother: r_rf_-Mother: 0.83951, 0.76077
#D-Get exactly what kaggler got:r_rf_SameAsK: 0.83726, 0.77990
  #D-Impute Embarked NAs with 'C': 0.83614
  #D-Impute Fare NA with 8.05: 0.83614
  #D-Make Pclass a factor: 0.82716
  #D-Change mice method, excluded cols, and seed: 0.83053
  #D-Make PassengerId a factor: 0.83053
  #D-Don't delete Ticket and Cabin cols: 0.83053
  #D-Add Deck: 0.83053
  #D-Use FamilySize, FamilySizeDiscrete, and Deck in mice: 0.83389
  #***Note: Ages are still different with mice, but they're close; maybe the difference is due to the seed
  #D-Make Child and Mother string factors: 0.83389
  #D-Remove AgeDiscrete: 0.83838
  #D-Add back Child, Mother: 0.83277
  #D-Use default ntree for rf: 0.83165
  #D-set.seed(754) for rf: 0.83726
#D-Add back AgeDiscrete: r_rf_+AgeDiscrete: 0.83502, 0.78947
#D-Remove Deck b/c it makes mice slow: r_rf_-Deck: 0.83277, 0.79426
#-Add FareDiscrete (low=0-50, high=>50):
#-Remove rare titles?



library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('caret') #for data-splitting
library('ggplot2') #visualization
library('ggthemes') # visualization


#Globals
FILENAME = 'r_rf_-Deck'
SEED_NUMBER = 343
PROD_RUN = T

#It seems absurd that they don't have a function to get this, but I can't
#find it, so create my own until I find a built-in way to get the OOB Error
getError = function(confusionMatrix) {
  return (sum(confusionMatrix[2:3]) / sum(confusionMatrix[1:4]))
}

getRandomForest = function(data) {
  #set.seed(SEED_NUMBER)
  return (randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch +
                         Fare + Embarked + Title + FamilySizeDiscrete + Child + Mother +
                         AgeDiscrete,
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
    if (i %% 100 == 0) print(paste('On training example', i))
    trainSubset = train[1:i,]
    rf = getRandomForest(trainSubset)
    trainPrediction = predict(rf, trainSubset)
    trainErrors[count] = getError(table(trainSubset$Survived, trainPrediction))

    cvPrediction = predict(rf, cv)
    cvErrors[count] = getError(table(cv$Survived, cvPrediction))
    count = count + 1
  }

  png(paste0('LearningCurve_', FILENAME, '.png'), width=500, height=350)
  plot(increments, trainErrors, type='l', ylim = c(0, max(cvErrors)), main='Learning Curve', xlab = "Number of Training Examples", ylab = "Error")
  lines(increments, cvErrors)
  dev.off()
}

#I do not understand any of this code, I borrowed it from a kaggler
plotImportances = function(rf, save=FALSE) {
  print('Plotting Feature Importances...')

  # Get importance
  importance = importance(rf)
  varImportance = data.frame(Variables = row.names(importance),
      Importance = round(importance[ ,'MeanDecreaseGini'],2))

  # Create a rank variable based on importance
  rankImportance = varImportance %>%
      mutate(Rank = paste0('#',dense_rank(desc(Importance))))

  if (save) {
    png(paste0('Importances_', FILENAME, '.png'), width=500, height=350)
  }
  print(ggplot(rankImportance, aes(x = reorder(Variables, Importance),
          y = Importance, fill = Importance)) +
      geom_bar(stat='identity') +
      geom_text(aes(x = Variables, y = 0.5, label = Rank),
          hjust=0, vjust=0.55, size = 4, colour = 'red') +
      labs(title='Feature Importances', x='Features') +
      coord_flip() +
      theme_few())
  if (save) {
    dev.off()
  }
}


#============= Main ================
#set.seed(SEED_NUMBER)

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
full$FamilySize = (1 + full$SibSp + full$Parch)

#discretize FamilySize: 1=Single, 2-4=Small, >5=Large (these values were arrived at by
#manually examining the data; families of size 2-4 seem to have a better chance of
#survival than singletons or large families)
full$FamilySizeDiscrete = cut(full$FamilySize, breaks=c(0, 1, 4, 1000), labels=c('Single', 'Small', 'Large'))


#impute missing values in Age, Fare, Embarked
print('Imputing missing values...')
#mice_imp = mice(subset(full, select=-c(Survived)), seed=SEED_NUMBER, printFlag=F)
set.seed(129)
mice_imp = mice(full[, !names(full) %in% c('PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived')], method='rf', printFlag=F)
mice_output = complete(mice_imp)
full$Age = mice_output$Age #263 occurrences
#full$Fare = mice_output$Fare #1 occurrence (row 1044=8.05)
#full$Embarked = mice_output$Embarked #2 occurrences (row 62=S, row 830=C)
full$Fare[1044] = 8.05
full$Embarked[c(62, 830)] = 'C'


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

if (PROD_RUN) {
  #plot learning curve
  #plotLearningCurve(train)
}

set.seed(754)
rf = getRandomForest(train)
plotImportances(rf, save=PROD_RUN)
print(paste('OOB Accuracy:', (1-getError(rf$confusion))))


if (PROD_RUN) {
  #Output solution
  prediction = predict(rf, test)
  solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  print(paste('Writing solution to file:', outputFilename, '...'))
  write.csv(solution, file=outputFilename, row.names=F)
}
