crossval <- function(form, x, fold = 10, cp = 0.01, model='nnet') {
  n <- nrow(x)
  prop <- n %/% fold
  set.seed(7)
  newseq <- rank(runif(n))
  k <- as.factor((newseq - 1)%/%prop + 1)
  y <- unlist(strsplit(as.character(form), " "))[2]
  vec.accuracy <- vector(length = fold)
  
  for (i in seq(fold)) {
    
    if(model == 'rpart'){
      # Decision Tree
      fit <- rpart(form, data = x[k != i,], method = "class")
      # print(fit)
      fcast <- predict(fit, newdata = x[k == i,], type = "class")
      cm <- ifelse(x[k == i, y] == fcast,1,0)
    }
    if(model == 'nnet'){
      # Neural Net
      fit <- nnet(form, data = x[k != i,], size=10, maxit=250)    
      # print(fit)
      fcast <- predict(fit, newdata = x[k == i,], type = "class")
      cm <- ifelse(x[k == i, y] == fcast,1,0)
    }
    if(model == 'rf'){
      # Random Forest
      fit <- randomForest(form, data = x[k != i,], ntree=500)    
      print(fit)
      fcast <- predict(fit, newdata = x[k == i,], type = "class")
      cm <- ifelse(x[k == i, y] == fcast,1,0)
    }
    if(model == 'svm'){
      # Random Forest
      fit <- svm(form, data = x[k != i,])    
      # print(fit)
      fcast <- predict(fit, newdata = x[k == i,], type = "class")
      cm <- ifelse(x[k == i, y] == fcast,1,0)
    }
    accuracy <- sum(cm)/length(cm)
    vec.accuracy[i] <- accuracy
    
  }
  
  avg.accuracy <- mean(vec.accuracy)
  avg.error <- 1 - avg.accuracy
  cv <- data.frame(Accuracy = avg.accuracy, Error = avg.error)
  
  return(cv)
}
