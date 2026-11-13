#Random forest
#install.packages("randomForest")
set.seed(123)
library(randomForest)
library(ggplot2)

X_train_df <- as.data.frame(Khan$xtrain)
y_train <- factor(Khan$ytrain)

X_test_df <- as.data.frame(Khan$xtest)
y_test <- factor(Khan$ytest)

# Assign identical gene names
gene_names <- paste0("Gene_", seq_len(ncol(X_train_df)))

colnames(X_train_df) <- gene_names
colnames(X_test_df) <- gene_names


rf_grid <- expand.grid(
  mtry = c(5, 15, 30, 50, 100)
)

rf_repeated <- train(
  x = X_train_df,
  y = y_train,
  method = "rf",
  metric = "Accuracy",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 500,
  importance = TRUE
)

# Best tuning parameter
best_mtry <- rf_repeated$bestTune$mtry

cat("Optimal mtry:", best_mtry, "\n")

# Cross-validation results
print(rf_repeated$results)

#plot(rf_repeated)

best_rf <- rf_repeated

# Training predictions
rf_train_pred <- predict(
  best_rf,
  newdata = X_train_df
)

# Test predictions
rf_test_pred <- predict(
  best_rf,
  newdata = X_test_df
)

# Accuracy
rf_train_accuracy <- mean(
  rf_train_pred == y_train
)

rf_test_accuracy <- mean(
  rf_test_pred == y_test
)

cat("RF Training Accuracy:",
    rf_train_accuracy, "\n")

cat("RF Test Accuracy:",
    rf_test_accuracy, "\n")

# Confusion matrix
rf_cm <- confusionMatrix(
  data = rf_test_pred,
  reference = y_test
)

print(rf_cm)

# Calculate variable importance
rf_importance <- varImp(
  best_rf,
  scale = FALSE
)

# Extract importance
rf_importance_df <- rf_importance$importance

# Add gene names
rf_importance_df$Gene <- rownames(
  rf_importance_df
)

# Sort by overall importance
rf_importance_df <- rf_importance_df[
  order(
    rf_importance_df$Overall,
    decreasing = TRUE
  ),
]

# Display top 20 genes
head(rf_importance_df, 20)

top_rf <- head(rf_importance_df, 20)

ggplot(
  top_rf,
  aes(
    x = reorder(Gene, Overall),
    y = Overall
  )
) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Top 20 Important Genes - Random Forest",
    x = "Genes",
    y = "Importance"
  ) +
  theme_minimal()

comparison <- data.frame(
  Model = c("SVM", "Random Forest"),
  
  CV_Accuracy = c(
    max(svm_repeated$results$Accuracy),
    max(rf_repeated$results$Accuracy)
  ),
  
  Test_Accuracy = c(
    train_accuracy * 0 + test_accuracy,
    rf_test_accuracy
  )
)

print(comparison)

comparison$Test_Accuracy <- c(
  test_accuracy,
  rf_test_accuracy
)

print(comparison)

comparison_long <- data.frame(
  Model = rep(
    comparison$Model,
    2
  ),
  Accuracy = c(
    comparison$CV_Accuracy,
    comparison$Test_Accuracy
  ),
  Type = rep(
    c("Cross-validation", "Test"),
    each = 2
  )
)

ggplot(
  comparison_long,
  aes(
    x = Model,
    y = Accuracy,
    fill = Type
  )
) +
  geom_col(
    position = "dodge"
  ) +
  labs(
    title = "SVM vs Random Forest Performance",
    x = "Machine Learning Model",
    y = "Accuracy"
  ) +
  theme_minimal()