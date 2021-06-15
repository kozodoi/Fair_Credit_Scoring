#' This function computes profit of a financial institution from a classifier.
#' 
#' @param class_preds: vector of predicted class labels
#' @param targets:     vector of true class labels
#' @param amounts:     vector of loan amounts
#' @param r:           total interest rate
#' 
#' @return list with profit, profit per loan and profit per EUR issued

# function for profit computation
compute_profit <- function(class_preds, targets, amounts, r = 0.2644) {
  
  # placeholder
  loanprofit <- NULL
  
  # go through loan applications
  for (i in 1:length(targets)) {
    
    # label and target
    pred_label <- class_preds[i]
    true_label <- targets[i]
    amount     <- amounts[i]
    
    # compute profit
    if (pred_label == "Bad" & true_label == "Bad") {
      p = 0
    } else if (pred_label == "Good" & true_label == "Bad") {
      p = -(0.55*0 + 0.10*1 + 0.35*0.5)*amount
    } else if (pred_label == "Good" & true_label == "Good") {
      p = amount  * r
    }else if (pred_label == "Bad" & true_label == "Good") {
      p = -amount * r
    }
    
    # sum profit
    loanprofit <- c(loanprofit, p)
  }
  
  # total profit
  profit        <- sum(loanprofit)
  profitPerLoan <- profit / length(targets)
  profitPerEUR  <- profitPerLoan / mean(amounts)
  
  # output
  return(list(profit        = profit, 
              profitPerLoan = profitPerLoan,
              profitPerEUR  = profitPerEUR))
}
