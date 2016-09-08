#Plot the NA values in a dataset

library(Amelia)

train = read.csv('data/train.csv', stringsAsFactors=F, na.strings=c(''))
test = read.csv('data/test.csv', stringsAsFactors=F, na.strings=c(''))
full = bind_rows(train, test)

missmap(full, main = "NAs by Column")