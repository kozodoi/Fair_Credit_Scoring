#' This function plots Pareto frontier with non-dominated solutions for two objectives.
#' 
#' @param pareto.set: matrix with solution fitness in two objectives
#' @param objectives: vector of two objective names
#' @param minimize: logical vector (whether the objectives should be minimized)
#' @param main: plot title
#' @param pch: shape of points on the frontier
#' @param cex: size of points on the frontier
#' 
#' @return plots the Pareto frontier

# plotting pareto frontier
PlotParetoFront <- function(pareto.set, pch = 16, objectives = c('Objective I', 'Objective II'),
                            minimize = c(T, T), main = 'Pareto Front', cex = 1.2, ...) {
  
  # plot the points
  plot(pareto.set, xlab = objectives[1], ylab = objectives[2], main = main, 
       type = 'p', pch = pch, col = 'red', cex = cex, ...)
  
  # put points to front
  points(pareto.set, pch = pch, col = 'black', cex = cex, bg = 'red')
  
  # connect the points
  if (nrow(pareto.set) > 1) {
    
    # go throw all points
    for (k in 1:(nrow(pareto.set) - 1)) {
      
      # find coordinates
      x = c(min(pareto.set[k:(k + 1), 1]), max(pareto.set[k:(k + 1), 1]))
      y = c(min(pareto.set[k:(k + 1), 2]), max(pareto.set[k:(k + 1), 2]))
      
      # draw lines
      if (minimize[1] == T) {
        lines(x = rep(x[2], 2), y = y, lwd = 0.6, lty = 2)
      }else{
        lines(x = rep(x[1], 2), y = y, lwd = 0.6, lty = 2)
      }
      if (minimize[2] == T) {
        lines(y = rep(y[2], 2), x = x, lwd = 0.6, lty = 2)
      }else{
        lines(y = rep(y[1], 2), x = x, lwd = 0.6, lty = 2)
      }
    }
  }
  
  # put points to front
  points(pareto.set, pch = pch, col = 'black', cex = cex, bg = 'red')
}