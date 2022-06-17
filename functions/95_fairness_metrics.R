#' This file contains helper functions that evaluate fairness of a classifier using 
#' three criteria: independence, separation and sufficiency.
#' 
#' @param sens.attr:      vector of sensitive attribute values
#' @param target.attr:    vector of target values
#' @param predicted.attr: vector of classifier predictions
#' 
#' @return fairness metric value


###################################
#                                 
#         INDEPENDENCE            
#                                 
###################################

statParDiff <- function(sens.attr   = df$AGE, 
                        target.attr = df$TARGET){
  
  # encode as factors
  sens.var    <- as.factor(sens.attr)
  target.var  <- as.factor(target.attr)
  sens.lvls   <- levels(sens.var)
  target.lvls <- levels(target.var)
  
  # construct data
  data <- cbind(sens.var, target.var)
  
  # compute counts
  total.count   <- nrow(data)
  target1.count <- nrow(data[target.var == target.lvls[1], ])
  
  # compute terms
  p_1 <- (nrow(data[sens.var == sens.lvls[1] & target.var == target.lvls[1], ]) / target1.count) * 
    (target1.count/total.count) / (nrow(data[sens.var == sens.lvls[1], ]) / total.count)
  p_2 <- (nrow(data[sens.var == sens.lvls[2] & target.var == target.lvls[1], ]) / target1.count) * 
    (target1.count/total.count) / (nrow(data[sens.var == sens.lvls[2], ]) / total.count)
  
  # compute difference
  if (length(p_1 - p_2) > 0) {
    return(abs(p_1 - p_2))
  }else{
    return(NA)
  }
}


###################################
#                                 
#         SEPARATION            
#                                 
###################################

### SEPARATION
avgOddsDiff <- function(sens.attr      = df$AGE,
                        target.attr    = df$TARGET, 
                        predicted.attr = df$class){
  
  # encode as factor
  sens.attr <- as.factor(sens.attr)
  
  # construct DF
  data <- data.frame(sens.attr, target.attr, predicted.attr, stringsAsFactors = T)
  colnames(data) <- c('sens', 'target', 'pred')

  # unprivileged group error rates
  data_un <- data[data[, 'sens'] == levels(data[, 'sens'])[1],]
  FN_un   <- nrow(data_un[data_un[,'target'] == 'Bad'  & data_un[,'pred'] == 'Good', ])
  FP_un   <- nrow(data_un[data_un[,'target'] == 'Good' & data_un[,'pred'] == 'Bad',  ])
  TP_un   <- nrow(data_un[data_un[,'target'] == 'Bad'  & data_un[,'pred'] == 'Bad',  ])
  TN_un   <- nrow(data_un[data_un[,'target'] == 'Good' & data_un[,'pred'] == 'Good', ])
  FPR_un  <- FP_un/(TN_un+FP_un)
  TPR_un  <- TP_un/(TP_un+FN_un)
  
  # privileged group error rates
  data_priv <- data[data[, 'sens'] == levels(data[, 'sens'])[2],]
  FN_priv   <- nrow(data_priv[data_priv[,'target'] == 'Bad'  & data_priv[,'pred'] == 'Good', ])
  FP_priv   <- nrow(data_priv[data_priv[,'target'] == 'Good' & data_priv[,'pred'] == 'Bad',  ])
  TP_priv   <- nrow(data_priv[data_priv[,'target'] == 'Bad'  & data_priv[,'pred'] == 'Bad',  ])
  TN_priv   <- nrow(data_priv[data_priv[,'target'] == 'Good' & data_priv[,'pred'] == 'Good', ])
  FPR_priv  <- FP_priv / (TN_priv+FP_priv)
  TPR_priv  <- TP_priv / (TP_priv+FN_priv)
  
  # compute difference
  # note that TPR_un - TPR_priv is equivalent to FNR_priv - FNR_un
  if (length(((FPR_un - FPR_priv) + (TPR_un - TPR_priv))/2) > 0) {
    return(abs(((FPR_un - FPR_priv) + (TPR_un - TPR_priv))/2))
  }else{
    return(NA)
  }
}



###################################
#                                 
#         SUFFICIENCY            
#                                 
###################################

predParDiff <- function(sens.attr      = df$AGE, 
                        target.attr    = df$TARGET, 
                        predicted.attr = df$class){
  
  # encode as factor
  sens.attr <- as.factor(sens.attr)
  
  # construct DF
  data <- data.frame(sens.attr, target.attr, predicted.attr, stringsAsFactors = T)
  colnames(data) <- c('sens', 'target', 'pred')
  
  # unprivileged group computations
  data_un <- data[data[, 'sens'] == levels(data[, 'sens'])[1], ]
  pp_un <- nrow(data_un[data_un[,'target'] == 'Good' & data_un[,'pred'] == 'Good', ]) / 
    nrow(data_un[data_un[,'pred'] == 'Good',])
  
  # privileged group computations
  data_priv <- data[data[, 'sens'] == levels(data[, 'sens'])[2],]
  pp_priv <- nrow(data_priv[data_priv[,'target'] == 'Good' & data_priv[,'pred'] == 'Good', ]) / 
    nrow(data_priv[data_priv[,'pred'] == 'Good',])
  
  # compute difference
  if (length(pp_un - pp_priv) > 0) {
    return(abs(pp_un - pp_priv))
  }else{
    return(NA)
  }
}
