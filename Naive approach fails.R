#Naive approach fails
library(MASS)
n <- nrow(X_train)
p <- ncol(X_train)
K <- ncol(B_true)

X_train_intercept <- cbind(
  Intercept = 1,
  X_train
)

X_test_intercept <- cbind(
  Intercept = 1,
  X_test
)

# One-hot encode training response
Y_train <- model.matrix(~ factor(y_train) - 1)

# One-hot encode test response
Y_test <- model.matrix(~ factor(y_test) - 1)

colnames(Y_train) <- paste0("Class", 1:K)
colnames(Y_test) <- paste0("Class", 1:K)

# Fit OLS using Moore-Penrose generalized inverse
beta_ols <- ginv(X_train_intercept) %*% Y_train

# Training predictions
train_scores <- X_train_intercept %*% beta_ols

# Test predictions
test_scores <- X_test_intercept %*% beta_ols

# Predict class with highest score
train_pred <- max.col(train_scores)
test_pred <- max.col(test_scores)

# Classification accuracy
train_accuracy <- mean(train_pred == y_train)
test_accuracy <- mean(test_pred == y_test)

cat("Training accuracy:", train_accuracy, "\n")
cat("Test accuracy:", test_accuracy, "\n")

# SVD
svd_result <- svd(
  X_train_intercept,
  nu = 0,
  nv = ncol(X_train_intercept)
)

# Numerical rank
tolerance <- max(dim(X_train_intercept)) *
  max(svd_result$d) *
  .Machine$double.eps

rank_X <- sum(svd_result$d > tolerance)

# Null space
null_space <- svd_result$v[
  ,
  (rank_X + 1):ncol(X_train_intercept),
  drop = FALSE
]

cat("Rank of design matrix:", rank_X, "\n")

cat("Null space dimension:",
    ncol(null_space), "\n")

# Select one null-space direction
v <- null_space[, 1]

# Create alternative coefficient matrix
beta_ols_2 <- beta_ols + 10 * v

# Compare training scores
train_scores_1 <- X_train_intercept %*% beta_ols

train_scores_2 <- X_train_intercept %*% beta_ols_2

# Compare coefficient matrices
cat("Coefficient difference:",
    sqrt(sum((beta_ols - beta_ols_2)^2)),
    "\n")

# Compare training predictions
cat("Maximum training score difference:",
    max(abs(train_scores_1 - train_scores_2)),
    "\n")

signal_strength <- 0.05
for (i in 1:n_train) {
  X_train[i, ] <- X_train[i, ] +
    signal_strength * B_true[, y_train[i]]
}

for (i in 1:n_test) {
  X_test[i, ] <- X_test[i, ] +
    signal_strength * B_true[, y_test[i]]
}

set.seed(123)

R <- 100

test_accuracy <- numeric(R)

for (r in 1:R) {
  
  # Generate independent predictors
  Xtr <- matrix(rnorm(60 * 2000), 60, 2000)
  Xte <- matrix(rnorm(1000 * 2000), 1000, 2000)
  
  # Balanced class labels
  ytr <- sample(rep(1:4, length.out = 60))
  yte <- sample(rep(1:4, length.out = 1000))
  
  # Add intercept
  Xtr <- cbind(1, Xtr)
  Xte <- cbind(1, Xte)
  
  # One-hot encoding
  Ytr <- model.matrix(
    ~ factor(ytr, levels = 1:4) - 1
  )
  
  # OLS
  beta <- MASS::ginv(Xtr) %*% Ytr
  
  # Test predictions
  scores <- Xte %*% beta
  
  pred <- max.col(scores)
  
  test_accuracy[r] <- mean(pred == yte)
}

# Results
cat("Mean test accuracy:",
    mean(test_accuracy), "\n")

cat("SD of test accuracy:",
    sd(test_accuracy), "\n")

cat("Chance accuracy:", 0.25, "\n")

# Plot
hist(
  test_accuracy,
  breaks = 15,
  main = "OLS Test Accuracy",
  xlab = "Test Accuracy",
  col = "lightblue"
)

abline(
  v = 0.25,
  col = "red",
  lwd = 2,
  lty = 2
)

