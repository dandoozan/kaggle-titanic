featureEngineer = function(data, useMice=FALSE) {
  #create Title feature from Name
  data$Title = gsub('(.*, )|(\\..*)', '', data$Name)
  data$Title[data$Title == 'Mlle' | data$Title == 'Ms'] = 'Miss'
  data$Title[data$Title == 'Mme'] = 'Mrs'
  data$Title[data$Title %in% c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')] = 'Rare_Title'
  data$Title = factor(data$Title)
  
  #create FamilySize feature
  familySize = (1 + data$SibSp + data$Parch)
  
  #discretize FamilySize: 1=Single, 2-4=Small, >5=Large (these values were arrived at by
  #manually examining the data; families of size 2-4 seem to have a better chance of
  #survival than singletons or large families)
  data$FamilySizeDiscrete = cut(familySize, breaks=c(0, 1, 4, 1000), labels=c('Single', 'Small', 'Large'))
  
  
  #impute missing values in Age, Fare, Embarked
  print('Imputing missing values...')
  if (useMice) {
    require(mice)
    set.seed(129)
    mice_imp = mice(subset(data, select=-c(PassengerId, Name, Ticket, Cabin, Survived)), method='rf', printFlag=F)
    mice_output = complete(mice_imp)
    data$Age = mice_output$Age
  } else {
    data$Age = na.roughfix(data$Age)
  }
  data$Fare[1044] = 8.05
  data$Embarked[c(62, 830)] = 'C'
  
  #discretize Fare
  data$FareDiscrete = cut(data$Fare, c(-1, 50, 10000), labels=c('Low', 'High'))
  
  #create Child feature
  data$Child[data$Age < 18] = 'Child'
  data$Child[data$Age >= 18] = 'Adult'
  data$Child = factor(data$Child)
  
  #create AgeDiscrete feature: 0-6=Young, 7-12=Middle, 13-18=Teen, >18=Adult
  data$AgeDiscrete = cut(data$Age, breaks=c(0, 6, 12, 18, 1000), labels=c('Young', 'Middle', 'Teen', 'Adult'))
  
  #create Mother feature
  data$Mother = 'NotMother'
  data$Mother[data$Sex == 'female' & data$Age > 18 & data$Parch > 0 & data$Title != 'Miss'] = 'Mother'
  data$Mother = factor(data$Mother)
  
  return(data)
}