#' This file defines meta-parameter grids used in the grid search.

# logistic regression
param.glm <- NULL
args.glm  <- list(family = "binomial")

# random forest
param.rf <- expand.grid(mtry = c(5, 10, 15))
args.rf  <- list(ntree = 500)

# gradient boosting
param.xgbTree <- expand.grid(
  nrounds          = c(100, 500, 1000),
  max_depth        = c(5, 10),
  gamma            = 0,
  eta              = 0.1,
  colsample_bytree = c(0.5, 1),
  subsample        = c(0.5, 1),
  min_child_weight = c(0.5, 1, 3)
)
args.xgbTree <- list()

# neural network
param.nnet <- expand.grid(decay = c(0.1, 0.5, 1, 1.5, 2),
                          size = c(5))
args.nnet <- list(maxit = 1000, trace = F)

# create vector of model names to call parameter grid in for-loop
model.names <- c(
  "glm",
  "rf",
  "xgbTree",
  "nnet")