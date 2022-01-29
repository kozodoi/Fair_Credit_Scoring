###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code processes intermediate results of Reject Option Classification imported 
# from the files saved in `code_11_postprocess4.ipynb`. The code saves final results for 
# reject option classification in `results`.



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

# metric bound
roc_bound <- 0.1

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

# factor encoding
dtest_unscaled$TARGET <- as.factor(ifelse(dtest_unscaled$TARGET == 1, 'Good', 'Bad'))
dtest_unscaled$AGE    <- as.factor(ifelse(dtest_unscaled$AGE == 1,    'Old',  'Young'))

# modeling
for (fold in seq(0, num_folds - 1)) {
  
  # feedback
  print('----------------------------------------')
  print(paste0('FOLD: ', fold))
  print('----------------------------------------')
 
  # load data and results
  POST_results <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_ROC_', roc_bound, '_predictions_test.csv')))
  
  
  #---- TESTING ----
  
  # reload grids
  source(file.path(func_path, '98_param_grids.R'))

  # placeholder
  test_results <- NULL
  
  # loop through base models
  for (m in model.names) {
    
    # load preds and scores
    cutoff_label <- POST_results[, paste0(m, '_fairLabels')]
    cutoff_label <- factor(as.character(cutoff_label), levels = c('Good', 'Bad'))
    scores       <- sapply(cutoff_label, function(x) ifelse(x == 'Good', 1, 0))
    
    # evaluation
    res <- evaluate(class_preds = cutoff_label, 
                    score_preds = scores,
                    targets     = dtest_unscaled$TARGET, 
                    amounts     = dtest_unscaled$CREDIT_AMNT, 
                    age         = dtest_unscaled$AGE,
                    r           = 0.2644)
    test_results <- cbind(test_results, res)
  }  
  
  # save results
  colnames(test_results) <- c(model.names)
  write.csv(test_results, file.path(res_path, 'final', paste0(data, '_', fold, '_ROC_results.csv')), row.names = T)
}
