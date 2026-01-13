# Load necessary libraries
# Install packages if necessary using install.packages("MASS")

# Logistic regression using Newton-Raphson method
logistic_regression_newton_raphson <- function(X, y, tol = 1e-6, max_iter = 100) {
  
  # Add an intercept column (a column of 1's) to X
  X <- cbind(1, X)  # Add intercept (bias term)
  
  # Initialize coefficients (beta) to zero
  beta <- rep(0, ncol(X))
  
  # Define the sigmoid function
  sigmoid <- function(z) {
    return(1 / (1 + exp(-z)))
  }
  
  # Iterate using Newton-Raphson
  for (iter in 1:max_iter) {
    # Compute the predicted probabilities
    p <- sigmoid(X %*% beta)
    
    # Compute the gradient (first derivative of the log-likelihood)
    gradient <- t(X) %*% (y - p)
    
    # Compute the Hessian matrix (second derivative of the log-likelihood)
    W <- diag(as.vector(p * (1 - p)))  # Diagonal weight matrix
    Hessian <- -t(X) %*% W %*% X
    
    # Update beta using the Newton-Raphson update rule
    beta_new <- beta - solve(Hessian) %*% gradient
    
    # Check for convergence (if change in beta is small)
    if (max(abs(beta_new - beta)) < tol) {
      cat("Convergence reached after", iter, "iterations.\n")
      return(beta_new)
    }
    
    # Update beta for the next iteration
    beta <- beta_new
  }
  
  cat("Max iterations reached without full convergence.\n")
  return(beta)
}

# Example usage:

# Generate some sample data (X = predictors, y = outcome)
set.seed(123)
X <- matrix(rnorm(100), nrow = 50, ncol = 2)  # 50 samples, 2 predictors
y <- rbinom(50, 1, prob = 0.5)  # Binary outcome

# Fit the logistic regression model using Newton-Raphson method
beta_estimates <- logistic_regression_newton_raphson(X, y)

# Print the estimated coefficients
print(beta_estimates)
