###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements Equalized Odds Processor. The intermediate results are 
# imported from the files saved in `code_08_postprocess1.R`. The code saves final 
# results for equalized odds processor in `results`.



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
  dval <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_', fold, '_valid.csv')))
  
  # factor encoding
  dval$TARGET <- as.factor(ifelse(dval$TARGET == 1, 'Good', 'Bad'))
  dval$AGE    <- as.factor(ifelse(dval$AGE == 1,    'Old',  'Young'))

  # import preds
  val_pred  <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_POST_training_results_dval.csv')))
  val_pred  <- cbind(val_pred, AGE = dval$AGE, TARGET = dval$TARGET)
  test_pred <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_POST_training_results_dtest.csv')))
  test_pred <- cbind(test_pred, AGE = dtest_unscaled$AGE, TARGET = dtest_unscaled$TARGET, CREDIT_AMNT = dtest_unscaled$CREDIT_AMNT)
  
  # reload grids
  source(file.path(func_path, '98_param_grids.R'))

  # loop through model names
  for (m in model.names) {
    
    # extract preds
    pred_0 <- val_pred[val_pred$AGE == 'Young', paste0(m, '_scores')]
    pred_1 <- val_pred[val_pred$AGE == 'Old',   paste0(m, '_scores')]

    # find threshold that optimizes the unprivileged group via EMP
    EMP <- empCreditScoring(scores = pred_0, classes = val_pred$TARGET[val_pred$AGE == 'Young'])
    assign(paste0('0_cutoff.', m), EMP$EMPCfrac)
    cutoff_label <- sapply(pred_0, function(x) ifelse(x <= quantile(pred_0, get(paste0('0_cutoff.', m))), 'Bad', 'Good'))
    
    # compute sensitivity
    cm     <- confusionMatrix(data = as.factor(cutoff_label), reference = as.factor(val_pred$TARGET[val_pred$AGE == 'Young']))
    sens_0 <- cm$byClass[['Sensitivity']]

    # find the threshold for the privileged group with the same sensitivity
    roc_curve <- roc(val_pred$TARGET[val_pred$AGE == 'Old'], pred_1)
    my.coords <- coords(roc = roc_curve, x = 'all', transpose = F)
    percent   <- ecdf(pred_1)
    assign(paste0('1_cutoff.', m), 1 - percent(my.coords[which.min(abs(my.coords$sensitivity - sens_0)), ]$threshold))
    cutoff_label <- sapply(pred_1, function(x) ifelse(x <= quantile(pred_1, get(paste0('1_cutoff.', m))), 'Bad', 'Good'))
  }
  
  # TEST RESULTS
  test_results <- NULL
  for (m in model.names) {
    
    # Assess test results
    assign(paste0('0_cutoff.', m), quantile(test_pred[test_pred$AGE == 'Young', paste0(m, '_scores')], get(paste0('0_cutoff.', m))))
    assign(paste0('1_cutoff.', m), quantile(test_pred[test_pred$AGE == 'Old',   paste0(m, '_scores')], get(paste0('1_cutoff.', m))))
    cutoff_label_0 <- sapply(test_pred[test_pred$AGE == 'Young', paste0(m, '_scores')], function(x) ifelse(x <= get(paste0('0_cutoff.',m)), 'Bad', 'Good'))
    cutoff_label_1 <- sapply(test_pred[test_pred$AGE == 'Old',   paste0(m, '_scores')], function(x) ifelse(x <= get(paste0('1_cutoff.',m)), 'Bad', 'Good'))
    cutoff_label <- c(cutoff_label_0, cutoff_label_1)
    test_label <- c(as.character(test_pred$TARGET[test_pred$AGE == 'Young']), 
                    as.character(test_pred$TARGET[test_pred$AGE == 'Old']))
    test_label <- as.factor(test_label)
    credit <- c(test_pred$CREDIT_AMNT[test_pred$AGE == 'Young'], 
                test_pred$CREDIT_AMNT[test_pred$AGE == 'Old'])
    age <- c(rep(0, length(cutoff_label_0)), rep(1, length(cutoff_label_1)))
    
    # evaluation
    res <- evaluate(class_preds = cutoff_label, 
                    score_preds = ifelse(cutoff_label == 'Good', 1, 0),
                    targets     = test_label, 
                    amounts     = credit, 
                    age         = age,
                    r           = 0.2644)
    test_results <- cbind(test_results, res)
  }  
  
  colnames(test_results) <- c(model.names)
  write.csv(test_results, file.path(res_path, 'final', paste0(data, '_', fold, '_EOP_results.csv')), row.names = T)
}
