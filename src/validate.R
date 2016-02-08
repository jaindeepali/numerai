validate <- function( fit, model = "" ) {  
  if( model == "nnet" ){
  	fcast <- predict(fit, valid, type = "class")
  }
  else{
  	fcast <- predict(fit, valid)
  }
  cm <- ifelse(valid[, target] == fcast,1,0)
  accuracy <- sum(cm)/length(cm)
  return(accuracy)
}