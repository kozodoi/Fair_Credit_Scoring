#' This file contains a cost function for trainControl to enable EMP-driven
#' meta-parameter tuning of the base classifiers within the caret framework.
#' 
#' @param data: data set with caret results
#' 
#' @return EMP value

# load EMP library
library(EMP)

# cost function
creditSummary <- function(data,
                          lev   = NULL, 
                          model = NULL) {
  
  # error handling
  lvls <- levels(data$obs)
  if (length(lvls) > 2) 
    stop(paste('Your outcome has', length(lvls), 
               'levels. The assignmentSummary() function is not appropriate.'))
  if (!all(levels(data[, 'pred']) == lvls)) 
    stop('levels of observed and predicted data do not match')

  # compute EMP
  out        <- EMP::empCreditScoring(scores = data$Good, classes = data$obs)
  out        <- out$EMPC        
  names(out) <- 'EMP'
  out
}