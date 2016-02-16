validate <- function( fit, model = "" ) {  
  if( model == "nnet" ){
  	fcast <- predict(fit, valid, type = "class")
  }
  else{
  	fcast <- predict(fit, valid)[,2]
  }
  predOb <- prediction(fcast, valid$target)
  auc <- performance(predOb, measure = "auc")
  return(auc)
}