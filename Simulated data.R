
#simulation part
set.seed(123)
n_train <- 60
n_test <- 1000

p <- 2000; K <- 4; s <- 8
X_train <- matrix(rnorm(n_train*p), nrow = n_train, ncol = p)
X_test <- matrix(rnorm(n_test*p), nrow = n_test, ncol = p)

B_true <- matrix(0, nrow = p, ncol = K)
true_variables <- c(5,20,100,250,400,650,900,1200)
class_variables <- split(true_variables, rep(1:K, each = 2))
true_effects <- c(3,-2.5,2,-3,2.5,-3.5,3.5,-2)
for(k in 1:K){
  idx <- class_variables[[k]]
  B_true[idx, k] <- true_effects[
    ((k-1)*2+1):(k*2)
  ]
}

y_train <- rep(1:K, length.out = n_train)
y_test <- rep(1:K, length.out = n_test)
y_train <- sample(y_train)
y_test <- sample(y_test)

for (i in 1:n_train) {
  X_train[i,] <- X_train[i,]+
    B_true[, y_train[i]]
}
for (i in 1:n_test) {
  X_test[i,] <- X_test[i,]+
    B_true[, y_test[i]]
}

colnames(X_train) <- paste0("Gene",1:p)
colnames(X_test) <- paste0("Gene",1:p)

train_data <- data.frame(
  Class = factor(y_train),
  X_train
)

test_data <- data.frame(
  Class = factor(y_test),
  X_test
)

cat("Training demension:", dim(train_data), "\n")
cat("Testing demension:", dim(test_data), "\n")
cat("Training predictor demension:", dim(X_train), "\n")
cat("Testing predictor demension:", dim(X_test), "\n")
cat("True nonzero coefficients:", sum(B_true!=0),"\n")

save(X_train, X_test, y_train, y_test, B_true, true_variables, train_data,
     test_data, file = "simulated_data.RData")
#load("simulated_data.RData")
