library('rpart')
library('nnet')
library('randomForest')
library('e1071')
library('caret')
library('dplyr')
library('ggplot2')
library('ROCR')
library('neuralnet')
library('NeuralNetTools')

source('src/helper.R')
source('src/validate.R')
source('src/crossval.R')
source('src/nnet_plot.R')

### Load Data

numerai_training_data <- read.csv("data/numerai_training_data.csv")
test <- read.csv("data/numerai_tournament_data.csv")
row.names(test) <- test$t_id
test$t_id <- NULL
numerai_example_predictions <- read.csv("submissions/numerai_example_predictions.csv")

### Benchmark Prediction (Score: 0.4984)

nrowTest <- nrow(numerai_example_predictions)
pred <- sample(0:1, nrowTest, replace = T)
submission(pred, 'benchmark.csv')

### Data Preparation

numerai_training_data$target <- as.factor(numerai_training_data$target)
train <- numerai_training_data[numerai_training_data$validation == 0,]
valid <- numerai_training_data[numerai_training_data$validation == 1,]

### Initial Analysis

summary(train)
summary(valid)
summary(test)

### Preprocessing

train.numeric <- train[,1:14]
valid.numeric <- valid[,1:14]
test.numeric <- test[,1:14]
train.numeric$target <- as.numeric(train$target) - 1
valid.numeric$target <- as.numeric(valid$target) - 1
pca_pp <- preProcess(train.numeric, method = c("center", "scale", "pca"))
train.pca <- predict(pca_pp, train.numeric)
valid.pca <- predict(pca_pp, valid.numeric)
test.pca <- predict(pca_pp, test.numeric)
train <- cbind(train, train.pca)
valid <- cbind(valid, valid.pca)
test <- cbind(test, test.pca)
train <- tbl_df(train)
valid <- tbl_df(valid)
test <- tbl_df(test)
grouped <- group_by(train, target)

### Formulae

formula.basic <- target ~ .
formula.orig <- target ~ f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + c1
formula.numeric <- target ~ f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14
formula.pca <- target ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + c1
formula <- formula.pca

### Basic Model

# rpart (Score: 0.5165) PCA : (Score: 0.5149)
fit <- rpart(formula, train)
acc <- validate(fit)
print(acc)
print( crossval(formula, train, model = "rpart") )
pred <- predict(fit, test, type = "prob")
submission(pred[,2], 'submissions/rpartPCA.csv')

# nnet (Score: 0.4973) PCA : (Score: 0.5137)
fit <- nnet(formula, train, size=10, maxit=250)
print( validate(fit, model = "nnet") )
print( crossval(formula, train, model = "nnet") )
pred <- predict(fit, test, type="raw")
submission(pred, 'submissions/nnetPCA.csv')

# randomForest (Score: 0.5253), randomForestPCA : 5158
fit <- randomForest(formula, train)
print( validate(fit) )
print( crossval(formula, train, model = "rf") )
pred <- predict(fit, test, type = "prob")
submission(pred[,2], 'submissions/randomForestPCA.csv')

# SVM (Score: 0.5111)
fit <- svm(formula, train, probability = TRUE)
print( validate(fit) )
print( crossval(formula, train, model = "svm") )
pred <- predict(fit, test, probability = TRUE)
submission(pred[,2], 'submissions/svm.csv')

# neuralnet (Score: 0.5049)
fit <- neuralnet(formula.numeric, data=train.numeric, hidden = c(10,5))
acc <- print( validate(fit) )
print(acc)
# print( crossval(formula, train, model = "svm") )
pred <- compute(fit, test.numeric)$net.result
submission(pred, 'submissions/neuralnet3.csv')
