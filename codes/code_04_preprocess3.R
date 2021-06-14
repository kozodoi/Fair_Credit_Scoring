###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code performs meta-parameter tuning of disparate impact remover by comparing
# multiple variants of this pre-processor in terms of the EMP on validation folds.
# The intermediate results are imported from the files exported in `code_02_preprocess2.R`.
# The code saves final results for disparate impact remover in `results`.



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

# repair level
all_di_repair_level <- c('0.5', '0.6', '0.7', '0.8', '0.9', '1.0')

# options
set.seed(seed)
options(scipen = 10)



##################################
#                                
#            MODELING      
#                                
##################################

# method
method <- 'DI'

# modeling
for (fold in seq(0, num_folds - 1)) {
  
  # feedback
  print('----------------------------------------')
  print(paste0('FOLD: ', fold))
  print('----------------------------------------')
  
  # placeholder
  emp_dval <- NULL
  
  # load EMP
  for (di_repair_level in all_di_repair_level) {
    
    # load image
    load(file.path(res_path, 'intermediate', 
                   paste0('IMAGE_PRE_', data, '_', method, '_', fold, '_', di_repair_level, '.Rdata')))
    
    # write EMP
    emp_dval <- rbind(emp_dval, c(di_repair_level, 
                                  EMP.glm, EMP.rf, EMP.xgbTree, EMP.nnet))
    
    # reset working paths
    cd <- dirname(rstudioapi::getActiveDocumentContext()$path)
    setwd(dirname(cd))
    source(file.path(cd, 'code_00_working_paths.R'))
  }
  
  # format results
  emp_dval <- data.frame(emp_dval)
  colnames(emp_dval) <- c('di_repair_level', model.names)
  
  # placeholder 
  test_results <- NULL
  
  # tune meta-parameter
  for (m in model.names) {
    
    # find optimal repair level
    repair_level <- emp_dval$di_repair_level[which.max(emp_dval[[m]])]
    
    # load relevant predictions
    test_result_model <- read.csv(file.path(res_path, 'intermediate', paste0(data, '_', fold, '_', method, '_', repair_level, '_results.csv')))
    
    # write predictions
    if (which(model.names == m) == 1) {
      test_results <- test_result_model[, c('X', m)]
    }else{
      test_results <- cbind(test_results, test_result_model[, m])
    }
  }
  
  # update colnames
  rownames(test_results) <- test_results$X
  test_results           <- test_results[, 2:ncol(test_results)]
  colnames(test_results) <- model.names
  
  # save predictions
  write.csv(test_results, file.path(res_path, 'final', 
                                    paste0(data, '_', fold, '_', method, '_results.csv')), row.names = T)
}
