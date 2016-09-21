#This code is inspired by this post: http://amunategui.github.io/sparse-matrix-glmnet/

#todo:
#D-do simple glmnet: r_glmnet: error=?, score=0.78947
#-read: https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet_beta.html
#-output the cv error/accuracy
#-plot learning curve
#-do more complex glmnet
#-change the s param in predict to use other metrics (other than lambda.min)


#Remove all objects from the current workspace
rm(list = ls())

library(glmnet)
library(Matrix) #sparse.model.matrix

#Globals
FILENAME = 'r_glmnet'
PROD_RUN = T

#============= Main ================

#get data: this gives me train, test, and full, all fully feature engineered
source('_getData.R')

#one hot encode factor variables, and convert it to matrix
set.seed(634)
trainSparseMatrix = sparse.model.matrix(Survived~.-1, data=subset(train, select=-c(PassengerId, Name, Ticket, Cabin)))
testSparseMatrix = sparse.model.matrix(~.-1, data=subset(test, select=-c(PassengerId, Name, Ticket, Cabin)))

#create model
print('Creating Model...')
set.seed(754)
fit = glmnet(trainSparseMatrix, train$Survived)

# use cv.glmnet to find best lambda/penalty - choosing small nfolds for cv due to...
# s is the penalty parameter
set.seed(754)
cv = cv.glmnet(trainSparseMatrix, train$Survived, nfolds=3)
#todo: experiment with different params for s
prediction = predict(fit, testSparseMatrix, type='response', s=cv$lambda.min)
prediction = as.numeric(prediction > 0.5)


if (PROD_RUN) {
  #Output solution
  solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)
  outputFilename = paste0(FILENAME, '.csv')
  print(paste('Writing solution to file:', outputFilename, '...'))
  write.csv(solution, file=outputFilename, row.names=F)
}

print('Done!')
