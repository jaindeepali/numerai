submission <- function( pred, name ) {
  prediction <- numerai_example_predictions
  prediction$probability <- pred
  write.table(prediction, file = name, sep = ",", row.names = FALSE, col.names = TRUE)
}