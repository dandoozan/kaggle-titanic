#Plot the missing data for each column

library('VIM')
train = read.csv('data/train.csv', stringsAsFactors=F)
aggr_plot <- aggr(train, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(train), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
