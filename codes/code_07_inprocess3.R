###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code performs meta-parameter tuning of three in-processors: 
# - Prejudice Remover
# - Meta-Fair Algorithm
# - Adversarial Debiasing
#
# The code compares the EMP of these in-processors on validation folds. 
# The intermediate results are imported from the files exported in `code_04_inprocess1.ipynb`
# and `code_05_inprocess2.ipynb`. The code saves final results for in-processors
# in `results`.



###################################
#                                 
#             SETTINGS            
#                                 
###################################

# clearing the memory
rm(list = ls())

# installing pacman
if (require(pacman) == F) install.packages('pacman')
library(pacman)

# libraries
p_load(caret, doParallel, kernlab, randomForest, nnet, 
       xgboost, foreach, e1071, pROC, EMP)

# working directory
cd <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dirname(cd))



###################################
#                                 
#           PARAMETERS            
#                                 
###################################

# paths
source(file.path(cd, 'code_00_working_paths.R'))

# data 
data <- 'taiwan'

# partitioning
num_folds <- 5
seed      <- 1

# adversary loss weight
all_ad_adversary_loss_weight <- c(0.1, 0.01, 0.001)

# options
set.seed(seed)
options(scipen = 10)



##################################
#                                
#          MODELING      
#                                
##################################

# helper functions
source(file.path(func_path, '94_evaluate.R'))
source(file.path(func_path, '95_fairness_metrics.R'))
source(file.path(func_path, '96_emp_summary.R'))
source(file.path(func_path, '99_compute_profit.R'))

# modeling
for (fold in seq(0, num_folds - 1)) {
  
  # feedback
  print('----------------------------------------')
  print(paste0('FOLD: ', fold))
  print('----------------------------------------')
  
  # read data
  dtest          <- read.csv(file.path(data_path, 'prepared/', paste0(data, '_scaled_', fold, '_test.csv')))
  dval           <- read.csv(file.path(data_path, 'prepared/', paste0(data, '_scaled_', fold, '_valid.csv')))
  dtest_unscaled <- read.csv(file.path(data_path, 'prepared/', paste0(data, '_orig_test.csv')))
  
  # factor encoding
  dval$TARGET            = as.factor(ifelse(dval$TARGET  == 1,          'Good', 'Bad'))
  dtest$TARGET           = as.factor(ifelse(dtest$TARGET == 1,          'Good', 'Bad'))
  dtest_unscaled$TARGET  = as.factor(ifelse(dtest_unscaled$TARGET == 1, 'Good', 'Bad'))
  dval$AGE               = as.factor(ifelse(dval$AGE  == 1,             'Old',  'Young'))
  dtest$AGE              = as.factor(ifelse(dtest$AGE == 1,             'Old',  'Young'))
  dtest_unscaled$AGE     = as.factor(ifelse(dtest_unscaled$AGE == 1,    'Old',  'Young'))


  #-------------------------- PREJUDICE REMOVER ----------------------------------
  
  # feedback
  print('- PREJUDICE REMOVER...')
  
  # load preds
  dval_pred  <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_PR_predictions_valid.csv')))
  dtest_pred <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_PR_predictions_test.csv')))
  
  #---- THRESHOLDING ----
  
  # Find optimal cutoff based on validation set
  empVals <- NULL
  for (col in 1:ncol(dval_pred)){
    empVal  <- empCreditScoring(dval_pred[,col], dval$TARGET)
    empVals <- unlist(c(empVals, empVal['EMPC']))
  }
  bestPrediction <- dval_pred[, which(empVals == max(empVals))[1]]
  best_eta       <- colnames(dval_pred)[which(empVals == max(empVals))[1]]
  
  # Define cutoff
  EMP    <- empCreditScoring(scores = bestPrediction, classes = dval$TARGET)
  cutoff <- EMP$EMPCfrac
    
  #---- TESTING ----
  
  # extract preds and scores
  pred         <- dtest_pred[, best_eta]
  cutoff       <- quantile(pred, cutoff)
  cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Good', 'Bad'))
  
  # evaluation
  res <- evaluate(class_preds = cutoff_label, 
                  score_preds = pred,
                  targets     = dtest_unscaled$TARGET, 
                  amounts     = dtest_unscaled$CREDIT_AMNT, 
                  age         = dtest_unscaled$AGE,
                  r           = 0.2644)

  # save results
  write.csv(res, file.path(res_path, 'final', paste0(data, '_', fold, '_PR_results.csv')), row.names = T)
  
  
  #-------------------------- META ALGORITHM ----------------------------------

  # feedback
  print('- META ALGORITHM...')

  # load preds
  dval_pred  <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_MA_predictions_valid.csv')))
  dtest_pred <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_MA_predictions_test.csv')))

  #---- THRESHOLDING ----

  # find optimal cutoff
  empVals <- NULL
  for (col in 1:ncol(dval_pred)){
    empVal <- empCreditScoring(dval_pred[,col], dval$TARGET)
    empVals <- unlist(c(empVals, empVal['EMPC']))
  }
  bestPrediction <- dval_pred[, which(empVals == max(empVals))[1]]
  best_eta <- colnames(dval_pred)[which(empVals == max(empVals))[1]]

  # define cutoff
  EMP    <- empCreditScoring(scores = bestPrediction, classes = dval$TARGET)
  cutoff <- EMP$EMPCfrac

  #---- TESTING ----

  # extract preds and scores
  pred         <- dtest_pred[, best_eta]
  cutoff       <- quantile(pred, cutoff)
  cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Good', 'Bad'))

  # evaluation
  res <- evaluate(class_preds = cutoff_label,
                  score_preds = pred,
                  targets     = dtest_unscaled$TARGET,
                  amounts     = dtest_unscaled$CREDIT_AMNT,
                  age         = dtest_unscaled$AGE,
                  r           = 0.2644)

  # save results
  write.csv(res, file.path(res_path, 'final', paste0(data, '_', fold, '_MA_results.csv')), row.names = T)

  
  #-------------------------- ADVERSARIAL DEBIASING ------------------------------
  
  # feedback
  print('- ADVERSARIAL DEBIASING...')
  
  #---- TUNING ----
  
  # placeholder
  emp_dval <- NULL
  
  # tune meta-parameter
  for (ad_adversary_loss_weight in all_ad_adversary_loss_weight) {
    
      # import preds
      dval_pred  <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_AD_', ad_adversary_loss_weight, '_predictions_valid.csv')))

      # write EMP
      EMP <- empCreditScoring(dval_pred[, 'scores'], 2 - dval_pred[, 'targets'])$EMPC
      emp_dval <- rbind(emp_dval, c(as.numeric(ad_adversary_loss_weight), EMP))
  }
  
  # format results
  emp_dval <- data.frame(emp_dval)
  colnames(emp_dval) <- c('adversary_loss_weight', 'EMP')
  
  # find optimal adversary loss weight
  adversary_loss_weight <- emp_dval$adversary_loss_weight[which.max(emp_dval$EMP)]
  
  # import relevant preds
  dval_pred  <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_AD_', adversary_loss_weight, '_predictions_valid.csv')))
  dtest_pred <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_AD_', adversary_loss_weight, '_predictions_test.csv')))
  
  #---- THRESHOLDING ----
  
  # find optimal cutoff
  EMP    <- empCreditScoring(dval_pred[, 'scores'], 2 - dval_pred[, 'targets'])
  cutoff <- EMP$EMPCfrac
  
  #---- TESTING ----
  
  # extract preds and scores
  pred         <- dtest_pred[, 'scores']
  cutoff       <- quantile(pred, cutoff)
  cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Good', 'Bad'))
  
  # evaluation
  res <- evaluate(class_preds = cutoff_label,
                  score_preds = pred,
                  targets     = dtest_unscaled$TARGET,
                  amounts     = dtest_unscaled$CREDIT_AMNT,
                  age         = dtest_unscaled$AGE,
                  r           = 0.2644)
  
  # save results
  write.csv(res, file.path(res_path, 'final', paste0(data, '_', fold, '_AD_results.csv')), row.names = T)
}
