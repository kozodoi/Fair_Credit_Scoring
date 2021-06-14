###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements Platt scaling. The intermediate results are imported from 
# the files saved in `code_08_postprocess1.R`. The code saves final results for 
# Platt scaling in `results`.



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
source(file.path(func_path, '97_caret_settings.R'))
source(file.path(func_path, '98_param_grids.R'))
source(file.path(func_path, '99_compute_profit.R'))

# read data
dtest_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_test.csv')))
dtest_unscaled <- subset(dtest_unscaled, select = c(CREDIT_AMNT,AGE, TARGET))

# factor encoding
dtest_unscaled$TARGET <- as.factor(ifelse(dtest_unscaled$TARGET == 1, 'Good', 'Bad'))
dtest_unscaled$AGE    <- as.factor(ifelse(dtest_unscaled$AGE == 1,    'Old',  'Young'))

# modeling
for (fold in seq(0, num_folds - 1)) {
  
  # feedback
  print('----------------------------------------')
  print(paste0('FOLD: ', fold))
  print('----------------------------------------')
  
  # import data
  dval_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_', fold, '_valid.csv')))
  dval_unscaled <- subset(dval_unscaled, select = c(AGE, TARGET))
  
  # factor encoding
  dval_unscaled$TARGET <- as.factor(ifelse(dval_unscaled$TARGET == 1,  'Good', 'Bad'))
  dval_unscaled$AGE    <- as.factor(ifelse(dval_unscaled$AGE  == 1,    'Old',  'Young'))
  
  # import preds
  dval_training_results  <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_POST_training_results_dval.csv')))
  dtest_training_results <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_POST_training_results_dtest.csv')))

  
  # ---- PLATT SCALING PER GROUP ----
  
  # reload girds
  source(file.path(func_path, '98_param_grids.R'))
  
  # loop through sensitive groups
  for (i in c('Young', 'Old')) {
    
    # subset data
    dval_target  <- dval_unscaled$TARGET[dval_unscaled$AGE    == i]
    dval_scores  <- dval_training_results[dval_unscaled$AGE   == i,]
    dtest_scores <- dtest_training_results[dtest_unscaled$AGE == i,]
    dtest_subset <- dtest_unscaled[dtest_unscaled$AGE         == i,]
    dval_subset  <- dval_unscaled[dval_unscaled$AGE           == i,]
    platt_scores       <- NULL
    platt_valid_scores <- NULL
    
    # perform scaling
    for (m in model.names) {
      
      # train logistic model with Yval ~ Y^val --> model_val
      dataframe_valid   <- data.frame(x = dval_scores[, paste0(m, '_scores')], y = dval_target)
      dataframe_valid$y <- ifelse(dataframe_valid$y == 'Good', 1, 0)
      model_val         <- glm(y~x, data = dataframe_valid, family = binomial)
      
      # predict scores
      valid_scores       <- predict(model_val, newdata = dataframe_valid, type = 'response')
      platt_valid_scores <- cbind(platt_valid_scores, valid_scores)

      # use model_val to predict ytest
      dataframe_test <- data.frame(x = dtest_scores[, paste0(m, '_scores')])
      test_scores    <- predict(model_val, newdata = dataframe_test, type = 'response')
      platt_scores   <- cbind(platt_scores, test_scores)
    }
    colnames(platt_scores)       <- model.names
    colnames(platt_valid_scores) <- model.names
    assign(paste0('platt_scores_', which(c('Young', 'Old') == i) - 1), 
           cbind(platt_scores, dtest_subset))
    assign(paste0('platt_valid_scores_', which(c('Young', 'Old') == i) - 1), 
           cbind(platt_valid_scores, dval_subset))
  }
  platt_results       <- rbind(platt_scores_0,       platt_scores_1)
  platt_valid_results <- rbind(platt_valid_scores_0, platt_valid_scores_1)
  
  
  #----- THRESHOLDING ----
  
  # find optimal cutoff based on validation set
  for (m in model.names){
    pred <- platt_valid_results[, m]
    EMP  <- empCreditScoring(scores = pred, classes = platt_valid_results$TARGET)
    assign(paste0('cutoff.', m), EMP$EMPCfrac)
  }
  
  
  #---- TESTING ----
  
  # assess test results
  test_results <- NULL
  for (m in model.names) {
    
    # load preds and scores
    pred         <- platt_results[, m]
    cutoff       <- quantile(pred, get(paste0('cutoff.', m)))
    cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Good', 'Bad'))
    
    # evaluation
    res <- evaluate(class_preds = cutoff_label, 
                    score_preds = pred,
                    targets     = platt_results$TARGET, 
                    amounts     = platt_results$CREDIT_AMNT, 
                    age         = platt_results$AGE,
                    r           = 0.2644)
    test_results <- cbind(test_results, res)
  }
  
  # save results
  colnames(test_results) <- c(model.names)
  write.csv(test_results, file.path(res_path, 'final', paste0(data, '_', fold, '_PL_results.csv')), row.names = T)
}
