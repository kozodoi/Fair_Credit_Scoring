###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code implements a set of base classifiers using fair pre-processed data
# by two pre-processors: Reweighting and Disparate Impact Remover. Pre-processed  
# data are imported from the files saved in `code_01_preprocess1.ipynb`. The code
# saves intermediate results for Disparate Impact Remover and final results for 
# Reweighting in the corresponding subfolders in `results`.



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

# cores
cores <- 8

# repair level
di_repair_level <- 0.5 # one of [0.5, 0.6, 0.7, 0.8, 0.9, '1.0']

# options
set.seed(seed)
options(scipen = 10)

# parallel computing
nrOfCores <- cores
registerDoParallel(cores = nrOfCores)
message(paste('\n Registered number of cores:\n',nrOfCores,'\n'))



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

# modeling loop
for (method in c('RW', 'DI')) {
  for (fold in seq(0, 4)) {

    #---- PREPARATIONS ----

    # feedback
    print('----------------------------------------')
    print(paste0('METHOD: ', method, ' | FOLD: ', fold))
    print('----------------------------------------')      
  
    # read data
    if (method == 'DI') {
      dtest  <- read.csv(file.path(data_path, 'prepared', paste0(data, '_', fold, '_pre_test_',  method, '_', di_repair_level, '.csv')))
      dval   <- read.csv(file.path(data_path, 'prepared', paste0(data, '_', fold, '_pre_valid_', method, '_', di_repair_level, '.csv')))
      dtrain <- read.csv(file.path(data_path, 'prepared', paste0(data, '_', fold, '_pre_train_', method, '_', di_repair_level, '.csv')))
    }else{
      dtest  <- read.csv(file.path(data_path, 'prepared', paste0(data, '_', fold, '_pre_test_',  method, '.csv')))
      dval   <- read.csv(file.path(data_path, 'prepared', paste0(data, '_', fold, '_pre_valid_', method, '.csv')))
      dtrain <- read.csv(file.path(data_path, 'prepared', paste0(data, '_', fold, '_pre_train_', method, '.csv')))
    }

    # load original test data
    dtest_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_test.csv')))
    dtest_unscaled <- subset(dtest_unscaled, select = c(CREDIT_AMNT, AGE, TARGET))
    
    # load original train data
    dtrain_unscaled <- read.csv(file.path(data_path, 'prepared', paste0(data, '_orig_', fold, '_train.csv')))
    dtrain_unscaled <- subset(dtrain_unscaled, select = c(CREDIT_AMNT, AGE, TARGET))

    # factor encoding
    dtrain$TARGET         <- as.factor(ifelse(dtrain$TARGET == 1,         'Good', 'Bad'))
    dval$TARGET           <- as.factor(ifelse(dval$TARGET   == 1,         'Good', 'Bad'))
    dtest$TARGET          <- as.factor(ifelse(dtest$TARGET  == 1,         'Good', 'Bad'))
    dtest_unscaled$TARGET <- as.factor(ifelse(dtest_unscaled$TARGET == 1, 'Good', 'Bad'))
    dtrain$AGE            <- as.factor(ifelse(dtrain$AGE == 1,            'Old',  'Young'))
    dval$AGE              <- as.factor(ifelse(dval$AGE   == 1,            'Old',  'Young'))
    dtest$AGE             <- as.factor(ifelse(dtest$AGE  == 1,            'Old',  'Young'))
    dtest_unscaled$AGE    <- as.factor(ifelse(dtest_unscaled$AGE == 1,    'Old',  'Young'))


    #---- TRAINING ----

    # grid search params
    source(file.path(func_path, '97_caret_settings.R'))
    source(file.path(func_path, '98_param_grids.R'))

    # train models and save result to model.'name'
    for (m in model.names) {
      print(paste0('-- ', m, '...'))
      grid <- get(paste('param.', m, sep = ''))
      args.train <- list(TARGET ~ .,
                         data      = dtrain,
                         method    = m,
                         tuneGrid  = grid,
                         metric    = 'EMP', 
                         trControl = model.control)
      args.model <- c(args.train, get(paste('args.', m, sep = '')))
      assign(paste('model.', m, sep = ''), do.call(caret::train, args.model))
      print(paste('-- model', m, 'finished training:', Sys.time(), sep = ' '))
    }

    # clean up
    for (m in model.names) {
      rm(list = c(paste0('args.', m), paste0('param.', m)))
    }
    gc()
    rm(args.model, args.train, model.control)


    #---- THRESHOLDING ----

    # Find optimal cutoff based on validation set
    for (m in model.names){
      pred <- predict(get(paste('model.', m, sep = '')), newdata = dval, type = 'prob')$Good
      EMP  <- empCreditScoring(scores = pred, classes = dval$TARGET)
      assign(paste0('cutoff.', m), EMP$EMPCfrac)
      assign(paste0('EMP.',    m), EMP$EMPC)
    }
    
    
    #---- TESTING ----
    
    # save image
    if (method == 'DI') {
      save.image(file.path(res_path, 'intermediate', paste0('IMAGE_PRE_', data, '_', method, '_', fold, '_', di_repair_level, '.Rdata')))
    }else{
      save.image(file.path(res_path, 'intermediate', paste0('IMAGE_PRE_', data, '_', method, '_', fold, '.Rdata')))
    }

    # reload helper functions
    source(file.path(func_path, '94_evaluate.R'))
    source(file.path(func_path, '95_fairness_metrics.R'))
    source(file.path(func_path, '96_emp_summary.R'))
    source(file.path(func_path, '97_caret_settings.R'))
    source(file.path(func_path, '98_param_grids.R'))
    source(file.path(func_path, '99_compute_profit.R'))
    
    # assess test results
    test_results <- NULL
    for (m in model.names) {
      
      # extract preds and scores
      pred         <- predict(get(paste0('model.', m)), newdata = dtest, type = 'prob')$Good
      cutoff       <- quantile(pred, get(paste0('cutoff.', m)))
      cutoff_label <- sapply(pred, function(x) ifelse(x > cutoff, 'Good', 'Bad'))
      
      # evaluation
      res <- evaluate(class_preds = cutoff_label, 
                      score_preds = pred,
                      targets     = dtest$TARGET, 
                      amounts     = dtest$CREDIT_AMNT * max(dtrain_unscaled$CREDIT_AMNT),
                      age         = dtest$AGE,
                      r           = 0.2644)
      test_results <- cbind(test_results, res)
    }
    
    # save results
    colnames(test_results) <- c(model.names)
    if (method == 'DI') {
      write.csv(test_results, file.path(res_path, 'intermediate', paste0(data, '_', fold, '_', method, '_', di_repair_level, '_results.csv')), row.names = T)
    }else{
      write.csv(test_results, file.path(res_path, 'final', paste0(data, '_', fold, '_', method, '_results.csv')), row.names = T)
    }
  }
}

# close cluster
stopImplicitCluster()
