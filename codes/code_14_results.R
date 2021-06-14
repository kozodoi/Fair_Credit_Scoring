###################################
#                                 
#             OVERVIEW            
#                                 
###################################

# This code performs aggregation and analysis of the empirical results. The 
# results are imported from individual files produced by different modeling
# scripts and saved as a 5-dimensional tensor (data x fold x base classifier x 
# fairness processor x evaluation metric) in `aggregated_results.RDS`. The code 
# produces the following outputs based on the analysis of the empirical results:
# - tables displaying mean performance estimates on data set level
# - tables displaying mean processor performance gains across the data sets
# - correlation matrix indicating agreement between performance and fairness
# - Pareto frontiers depicting trade-off between profit and fairness



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
p_load(stargazer, ecr)

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

# names
datasets <- c('german',
              'bene',
              'taiwan',
              'uk',
              'pkdd',
              'gmsc',
              'homecredit')
metrics <- c('AUC',
             'acceptedLoans',
             'profitPerEUR',
             'statParityDiff', 
             'averageOddsDiff', 
             'predParityDiff')
models <- c('glm',
            'rf',
            'xgbTree',
            'nnet')
methods <- c('pre_reweighing',
             'pre_di_remover',
             'in_prej_remover',
             'in_adv_debias',
             'in_meta_alg',
             'post_roc',
             'post_eop',
             'post_platt',
             'bench_maxprof')

# params
num_data    <- length(datasets)
num_metrics <- length(metrics)
num_methods <- length(methods)
num_folds   <- 5



##################################
#                                
#         IMPORT RESULTS      
#                                
##################################

# model names
source(file.path(func_path, '98_param_grids.R'))

# placeholders
results <- array(NA, dim = c(num_data, num_folds, length(models), num_methods, num_metrics))

# extract results
for (data in datasets) {
  
  # feedback
  print(paste0('- ', data, '...'))
  
  # loop through folds and base models
  for (fold in seq(0, num_folds - 1)) {
    for (model in models) {
  
      # pre-processors
      results[which(datasets == data), fold + 1, which(models == model), 1, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold, '_RW_results.csv')), row.names = 1)[metrics, model]
      results[which(datasets == data), fold + 1, which(models == model), 2, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold, '_DI_results.csv')), row.names = 1)[metrics, model]
      
      # in-processors
      results[which(datasets == data), fold + 1, which(models == model), 3, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold,'_PR_results.csv')), row.names = 1)[metrics, 1]
      results[which(datasets == data), fold + 1, which(models == model), 4, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold,'_AD_results.csv')), row.names = 1)[metrics, 1]
      results[which(datasets == data), fold + 1, which(models == model), 5, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold,'_MA_results.csv')), row.names = 1)[metrics, 1]
      
      # post-processors
      results[which(datasets == data), fold + 1, which(models == model), 6, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold, '_ROC_results.csv')), row.names = 1)[metrics, model]
      results[which(datasets == data), fold + 1, which(models == model), 7, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold, '_EOP_results.csv')), row.names = 1)[metrics, model]
      results[which(datasets == data), fold + 1, which(models == model), 8, ] <-
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold, '_PL_results.csv')), row.names = 1)[metrics, model]
      
      # benchmarks
      results[which(datasets == data), fold + 1, which(models == model), 9, ] <- 
        read.csv(file.path(res_path, 'final', paste0(data, '_', fold, '_bench_maxprof_results.csv')), row.names = 1)[metrics, model]
    }
  }
}

# save results
saveRDS(results, file.path(res_path, 'final', 'aggregated_results.RDS'))



##################################
#                                
#    PERFORMANCE PER DATA SET   
#                                
##################################

# load results
results <- readRDS(file.path(res_path, 'final', 'aggregated_results.RDS'))

# displayed metrics
rel_metrics <- c('AUC', 
                 'profitPerEUR',
                 'acceptedLoans',
                 'statParityDiff', 
                 'averageOddsDiff', 
                 'predParityDiff')

# extract results
for (data in datasets) {

  # placeholders
  tmp_results_me           <- matrix(nrow = num_methods, ncol = length(rel_metrics))
  rownames(tmp_results_me) <- methods
  colnames(tmp_results_me) <- rel_metrics
  tmp_results_se           <- tmp_results_me
  
  # extract results
  for (metric in rel_metrics) {
    vals <- NULL
    for (model in models) {
        tmp_res <- data.frame(results[which(datasets == data), , which(models == model), , which(metrics == metric)])
        vals    <- rbind(vals, tmp_res)
    }
    colnames(vals) <- methods
    tmp_results_me[ , which(rel_metrics == metric)] <- apply(vals, 2, function(x) mean(x, na.rm = T))
    tmp_results_se[ , which(rel_metrics == metric)] <- apply(vals, 2, function(x) sd(x / sqrt(length(x)), na.rm = T))
  }
  
  # prepare a table
  stargazer(tmp_results_me, type = 'text', title = data, digits = 4, 
            out = file.path(out_path, paste0('results_', data, '.html')))
}



##################################
#                                
#    AGGREGATED PERFORMANCE
#                                
##################################

# load results
results <- readRDS(file.path(res_path, 'final', 'aggregated_results.RDS'))

# displayed metrics
rel_metrics <- c('AUC', 
                 'profitPerEUR',
                 'statParityDiff', 
                 'averageOddsDiff', 
                 'predParityDiff')

# placeholders
tmp_results           <- matrix(nrow = num_methods, ncol = length(rel_metrics))
rownames(tmp_results) <- methods
colnames(tmp_results) <- rel_metrics

# extract results
for (metric in rel_metrics) {
  vals <- NULL
  for (model in models) {
    for (data in datasets) {
      tmp_res  <- data.frame(results[which(datasets == data), , which(models == model), , which(metrics == metric)])
      tmp_res  <- na.omit(tmp_res)
      vals     <- rbind(vals, tmp_res)
    }
  }
  colnames(vals) <- methods
  gaps  <- (vals - vals[, 'bench_maxprof']) / vals[, 'bench_maxprof'] * 100
  gaps  <- gaps[is.finite(rowSums(gaps)), ]
  tmp_results[ , which(rel_metrics == metric)] <- apply(gaps, 2, function(x) mean(x, na.rm = T))
}

# invert signs for fairness
tmp_results[, c('statParityDiff', 'averageOddsDiff', 'predParityDiff')] <- 
  -tmp_results[, c('statParityDiff', 'averageOddsDiff', 'predParityDiff')]

# add mean gains
mean_gains  <- apply(tmp_results[1:nrow(tmp_results)-1, ], 2, function(x) mean(x, na.rm = T))
tmp_results <- rbind(tmp_results[1:nrow(tmp_results)-1, ], mean_gains)

# export table
stargazer(tmp_results, title = 'Performance Gains)', type = 'text',  digits = 2, 
          out = file.path(out_path, 'performance_gains.html'))



##################################
#                                
#       CORRELATION ANALYSIS      
#                                
##################################

# load results
results <- readRDS(file.path(res_path, 'final', 'aggregated_results.RDS'))

# displayed metrics
rel_metrics <- c('AUC', 
                 'profitPerEUR',
                 'statParityDiff', 
                 'averageOddsDiff', 
                 'predParityDiff')

# placeholders
tmp_results_me           <- matrix(nrow = num_methods, ncol = num_metrics)
rownames(tmp_results_me) <- methods
colnames(tmp_results_me) <- metrics

# extract results
for (data in datasets) {
  res <- list()
  for (metric in rel_metrics) {
    vals <- NULL
    for (model in models) {
        tmp_res <- data.frame(results[which(datasets == data), , which(models == model), , which(metrics == metric)])
        vals    <- rbind(vals, apply(tmp_res, 2, function(x) mean(x, na.rm = T)))
    }
    colnames(vals) <- methods
    res[[metric]]  <- vals
  }

  # transform data
  cor_methods <- methods[!(methods %in% 'base')]
  cor_vals    <- NULL
  res_tmp     <- list()
  for (method in cor_methods) {
    for (metric in metrics) {
      res_tmp[[metric]] <- res[[metric]][, method]
    }
    cor_vals <- rbind(cor_vals, data.frame(res_tmp))
  }
  
  # store dataset-specific correlation
  if (which(datasets == data) == 1) {
    cors <- cor(cor_vals, use = 'pairwise.complete.obs', method = 'spearman') / length(datasets)
  }else{
    cors <- cors + cor(cor_vals, use = 'pairwise.complete.obs', method = 'spearman') / length(datasets)
  }
}

# construct and export correlation matrix
colnames(cors) <- rel_metrics
rownames(cors) <- colnames(cors) 
cors[upper.tri(cors)] <- NA
stargazer(cors, type = 'text',  digits = 4, out = file.path(out_path, 'correlations.html'))



##################################
#                                
#         PARETO FRONTIERS      
#                                
##################################

# helper function
source(file.path(func_path, '93_plot_pareto_frontier.R'))

# load results
results <- readRDS(file.path(res_path, 'final', 'aggregated_results.RDS'))

# relevant criterion
criterion <- 'averageOddsDiff'
if (criterion == 'statParityDiff')  {axt = 'Independence'}
if (criterion == 'averageOddsDiff') {axt = 'Separation'}
if (criterion == 'predParityDiff')  {axt = 'Sufficiency'}

# plot settings
par(mfrow = c(2,4), mar = c(4, 4, 2, 1) + 0.1)

# placeholder
per_02  <- NULL
per_min <- NULL
sep_min <- NULL

# loop through data sets
for (data in datasets) {
  
  # extract results
  data_vals <- NULL
  for (model in models) {
    for (method in methods) {
      tmp_res   <- data.frame(t(results[which(datasets == data), , which(models == model), which(methods == method), ]))
      data_vals <- rbind(data_vals, t(tmp_res))
    }
  }
  
  # column names
  colnames(data_vals) <- metrics  
  
  # preparation
  vals <- data_vals[, c(criterion, 'profitPerEUR')]
  vals <- na.omit(vals)
  vals <- as.matrix(vals, ncol = 2)
  vals <- vals[order(vals[, 1], decreasing = T), ]
  vals <- vals[!duplicated(vals), ]
  
  # drop non-dominated solutions
  vals[, 2] <- 1 - vals[, 2]
  vals      <- vals[nondominated(t(vals)), ]
  vals[, 2] <- 1 - vals[, 2]
  
  # compute profit percentage required to reduce fairness to 0.2
  per_02  <- c(per_02,  ((vals[1, 2] - vals[vals[, 1] <= 0.2, ][1, 2]) / vals[1, 2]))
  per_min <- c(per_min, ((vals[1, 2] - vals[nrow(vals), 2]) / vals[1, 2]))
  sep_min <- c(sep_min, vals[nrow(vals), 1])
  
  # plot settings for edge subplots
  par(mar = c(4,1.5,2,1.5) + 0.1)
  if (data %in% c('german', 'pkdd')) {par(mar = c(4, 3.5, 2, 1.5) + 0.1)}
  if (data %in% c('uk'))             {par(mar = c(4, 1.5, 2, 0.8) + 0.1)}
  
  # data id
  if (data == 'pkdd') {
    data.id <- 'pakdd'
  }else{
    data.id <- data
  }
  
  # plot Pareto front
  PlotParetoFront(vals, minimize = c(F, T), main = paste0('Data: ', data.id), objectives = c('', ''), cex = 2, pch = 21)
  mtext(cex = 0.85, side = 1, text = axt, line = 2.4)
  if (data %in% c('german', 'pkdd')) {
    mtext(cex = 0.80, side = 2, text = 'Profit per EUR Issued', line = 2.5)
  }
}

# export plot
dev.copy(pdf, file.path(out_path, paste0('pareto_frontiers.pdf')), width = 10, height = 5)
dev.off()

# display mean profit reductions
print(paste0('Profit reduction to reduce SEP to 0.2: ', round(100*mean(per_02), 2),  '%'))
print(paste0('Profit reduction to reduce SEP to min (', round(mean(sep_min), 2), '): ', round(100*mean(per_min), 2), '%'))
