#' This function performs evaluation of the classifier predictions in terms of 
#' predictive performance, profitability and fairness.
#' 
#' @param class_preds: vector of predicted class labels
#' @param score_preds: vector of predicted scores 
#' @param targets:     vector of true class labels
#' @param amounts:     vector of loan amounts
#' @param age:         vector of applicant age labels
#' @param r:           total interest rate
#' 
#' @return matrix with performance values

evaluate <- function(class_preds, 
                     score_preds,
                     targets, 
                     amounts, 
                     age,
                     r = 0.2644) {
  
  
  ##### PERFORMANCE

  # AUC
  AUC <- as.numeric(roc(targets, score_preds)$auc)
  
  
  # BACC
  cm <- confusionMatrix(data      = as.factor(class_preds),
                        reference = as.factor(targets))
  balAccuracy <- cm$byClass[['Balanced Accuracy']]
  
  
  ##### PROFITABILITY
  
  # EMP
  EMPCS <- empCreditScoring(scores  = score_preds, 
                            classes = targets)
  EMP <- EMPCS$EMPC
  
  # acceptance rate
  acceptedLoans <- 1 - EMPCS$EMPCfrac
  
  # class preds
  cutoff       <- quantile(score_preds, EMPCS$EMPCfrac)
  cutoff_label <- sapply(score_preds, function(x) ifelse(x > cutoff, 'Good', 'Bad'))
  
  # random subset if working with class preds
  if (length(unique(score_preds)) == 2) {
    set.seed(1)
    preds_df <- data.frame(score_preds = score_preds, cutoff_label = factor('Good', levels = c('Good', 'Bad')), index = 1:length(score_preds))
    preds_df <- preds_df[order(preds_df$score_preds, decreasing = F), ]
    preds_df$cutoff_label[1:round(EMPCS$EMPCfrac * nrow(preds_df))] <- 'Bad'
    preds_df     <- preds_df[order(preds_df$index, decreasing = F), ]
    cutoff_label <- preds_df$cutoff_label
  }
  
  # profit
  profits <- compute_profit(class_preds = cutoff_label,
                            targets     = targets,
                            amounts     = amounts,
                            r           = r)

  
  ###### FAIRNESS 
  
  # fairness criteria
  statParityDiff  <- statParDiff(sens.attr = age, target.attr = class_preds)
  averageOddsDiff <- avgOddsDiff(sens.attr = age, target.attr = targets, predicted.attr = class_preds)
  predParityDiff  <- predParDiff(sens.attr = age, target.attr = targets, predicted.attr = class_preds)
  

  ##### OUTPUT
  
  # merge results
  test_eval <- rbind(AUC, 
                     balAccuracy, 
                     EMP, 
                     acceptedLoans,
                     profits$profit, 
                     profits$profitPerLoan,
                     profits$profitPerEUR,
                     statParityDiff, 
                     averageOddsDiff, 
                     predParityDiff)
  rownames(test_eval)[5:7] <- c('profit', 'profitPerLoan', 'profitPerEUR')
  return(test_eval)
}
