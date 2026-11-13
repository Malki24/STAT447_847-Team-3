#svm
#install.packages(c("caret", "kernlab"))

library(caret)
library(kernlab)

# Make sure class labels are factors
y_train <- factor(y_train, levels = 1:K)
y_test <- factor(y_test, levels = 1:K)

# Convert predictors to data frames
X_train_df <- as.data.frame(X_train)
X_test_df <- as.data.frame(X_test)

# Add gene names if necessary
colnames(X_train_df) <- paste0("Gene", 1:p)
colnames(X_test_df) <- paste0("Gene", 1:p)

ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 10,
  classProbs = FALSE,
  savePredictions = "final"
)

cost_grid <- expand.grid(
  C = 10^seq(-5, 5, length.out =  9)
)

svm_repeated <- train(
  x = X_train_df,
  y = y_train,
  method = "svmLinear",
  metric = "Accuracy",
  trControl = ctrl,
  tuneGrid = cost_grid,
  preProcess = c("center", "scale")
)

# Best tuning parameter
best_cost <- svm_repeated$bestTune$C
cat("Optimal cost:", best_cost, "\n")

# Cross-validation results
print(svm_repeated$results)

plot(svm_repeated)

best_svm <- svm_repeated

# Training predictions
train_pred <- predict(
  best_svm,
  newdata = X_train_df
)

# Test predictions
test_pred <- predict(
  best_svm,
  newdata = X_test_df
)

train_accuracy <- mean(train_pred == y_train)

test_accuracy <- mean(test_pred == y_test)

cat("Training accuracy:", train_accuracy, "\n")
cat("Test accuracy:", test_accuracy, "\n")

confusionMatrix(
  data = test_pred,
  reference = y_test
)

library(ISLR2)

data(Khan)

X_train_khan <- as.data.frame(Khan$xtrain)
X_test_khan <- as.data.frame(Khan$xtest)

y_train_khan <- factor(Khan$ytrain)
y_test_khan <- factor(
  Khan$ytest,
  levels = levels(y_train_khan)
)

svm_khan_repeated <- train(
  x = X_train_khan,
  y = y_train_khan,
  method = "svmLinear",
  metric = "Accuracy",
  trControl = ctrl,
  tuneGrid = cost_grid,
  preProcess = c("center", "scale")
)

# Optimal cost
best_cost_khan <- svm_khan_repeated$bestTune$C

cat("Optimal Khan cost:", best_cost_khan, "\n")

# Test predictions
khan_test_pred <- predict(
  svm_khan_repeated,
  newdata = X_test_khan
)

# Test accuracy
khan_accuracy <- mean(
  khan_test_pred == y_test_khan
)

cat("Khan test accuracy:", khan_accuracy, "\n")

# Confusion matrix
confusionMatrix(
  khan_test_pred,
  y_test_khan
)

svm_repeated$results[
  svm_repeated$results$C == best_cost_khan,
  c("C", "Accuracy", "Kappa", "AccuracySD")
]

svm_repeated$results[
  order(-svm_repeated$results$Accuracy),
]

# Calculate SVM variable importance
svm_importance <- varImp(
  best_svm,
  scale = FALSE
)

# Display top 20 genes
svm_importance_df <- svm_importance$importance

svm_importance_df$Gene <- rownames(svm_importance_df)

svm_importance_df <- svm_importance_df[
  order(
    apply(svm_importance_df[, setdiff(
      names(svm_importance_df), "Gene"
    ), drop = FALSE], 1, mean),
    decreasing = TRUE
  ),
]

head(svm_importance_df, 20)

top_svm <- head(svm_importance_df, 20)

ggplot(
  top_svm,
  aes(
    x = reorder(Gene, -Overall),
    y = Overall
  )
) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Top 20 Important Genes - SVM",
    x = "Genes",
    y = "Importance"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 90,
      hjust = 1
    )
  )


# Training class proportions
train_prop <- prop.table(table(y_train_khan))

# Test class proportions
test_prop <- prop.table(table(y_test_khan))

# Random guessing based on training proportions
random_baseline <- sum(
  train_prop[names(test_prop)] * test_prop
)

# Majority class
majority_class <- names(which.max(train_prop))

# Majority-class test accuracy
majority_baseline <- mean(
  y_test_khan == majority_class
)

cat("Random baseline:", random_baseline, "\n")

cat("Majority-class baseline:", majority_baseline, "\n")

