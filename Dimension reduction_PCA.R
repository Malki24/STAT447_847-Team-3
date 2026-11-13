#Comment
library(ISLR2)
library(tidymodels)

data(Khan)
set.seed(847)

#Prepare training and test datasets
train <- as.data.frame(Khan$xtrain)
test <- as.data.frame(Khan$xtest)

names(train) <- paste0("gene_", seq_len(ncol(train)))
names(test) <- names(train)

#Response for cancer classes
train$Class <- factor(Khan$ytrain)
test$Class <- factor(Khan$ytest,
                     levels = levels(train$Class))

#Examine the dataset
dim(train)
dim(test)
table(train$Class)

#Extract gene expression predictors
x_train <- sapply(
  train[,setdiff(names(train), "class"), drop = FALSE],
  function(x) as.numeric(as.character(x))
)

#Remove any constant predistors
keep <- apply(x_train,2,sd) > 0 
x_train <- x_train[,keep,drop = FALSE]

#Fit PCA using training data only
pca <- prcomp(x_train,
              center = TRUE,
              scale. = TRUE)

#Percentage of variance explained
variance_explained <- 100*pca$sdev^2/sum(pca$sdev^2)

# First 10 cumulative percentages
cumulative_variance <- cumsum(variance_explained)
cumulative_variance[1:10]

# Plot the first 10 components
barplot(variance_explained[1:10],
        names.arg = 1:10,
        xlab = "Principal component",
        ylab = "Variance explained (%)",
        main = "PCA: Variance Explained")

# Visualize the first two components
plot(pca$x[, 1], pca$x[, 2],
     col = as.integer(train$Class),
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "Khan Data: First Two PCs")

legend("topright",
       legend = levels(train$Class),
       col = seq_along(levels(train$Class)),
       pch = 19)

# PCA preprocessing
pca_recipe <- recipe(Class ~ ., data = train) |>
  step_zv(all_predictors()) |>
  step_normalize(all_predictors()) |>
  step_pca(all_predictors(),
           num_comp = tune()) |>
  step_normalize(all_predictors())

# Multinomial classification model
pca_model <- multinom_reg() |>
  set_engine("nnet",
             trace = FALSE,
             maxit = 1000)

# Combine PCA and classification model
pca_workflow <- workflow() |>
  add_recipe(pca_recipe) |>
  add_model(pca_model)

# Repeated 5-fold cross-validation
set.seed(847)

folds <- vfold_cv(
  train,
  v = 5,
  repeats = 10,
  strata = Class
)

# Candidate numbers of components
pca_grid <- tibble(num_comp = 1:20)

# Tune the number of components
pca_cv <- tune_grid(
  pca_workflow,
  resamples = folds,
  grid = pca_grid,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = TRUE)
)

# View cross-validation results
collect_metrics(pca_cv)

collect_metrics(pca_cv) |>
  select(num_comp, mean, std_err)


# Plot predictive accuracy against components
autoplot(pca_cv)


# # Select the best number of components
# best_pca <- select_best(
#   pca_cv,
#   metric = "accuracy"
# )
# 
# print(best_pca)

best_pca <- tibble(num_comp = 15)

# Fit the final model on all training observations
final_pca <- finalize_workflow(
  pca_workflow,
  best_pca
) |>
  fit(data = train)

# Predict held-out test observations
test_predictions <- bind_cols(
  test[, "Class", drop = FALSE],
  predict(final_pca, new_data = test)
)

# Test accuracy
accuracy(test_predictions,
         truth = Class,
         estimate = .pred_class)

# Confusion matrix
conf_mat(test_predictions,
         truth = Class,
         estimate = .pred_class)

# Record reproducibility information
sessionInfo()

#######################################
# Extract cross-validation results
cv_results <- collect_metrics(pca_cv) |>
  filter(.metric == "accuracy")

# Plot mean cross-validation accuracy
ggplot(cv_results,
       aes(x = num_comp, y = mean)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  geom_vline(xintercept = 15,
             linetype = "dashed",
             color = "red") +
  geom_point(
    data = cv_results |> filter(num_comp == 15),
    color = "red",
    size = 3
  ) +
  labs(
    title = "PCA: Cross-Validation Accuracy",
    subtitle = "Selected model: 15 principal components",
    x = "Number of Principal Components",
    y = "Mean CV Accuracy"
  ) +
  theme_minimal()

# Generate confusion matrix
cm <- conf_mat(
  test_predictions,
  truth = Class,
  estimate = .pred_class
)

# Plot confusion matrix
autoplot(cm, type = "heatmap") +
  labs(
    title = "Confusion Matrix: PCA with 15 Components",
    subtitle = "Khan Test Dataset"
  ) +
  theme_minimal()

# Extract the fitted preprocessing recipe
trained_recipe <- extract_recipe(
  final_pca,
  estimated = TRUE
)

# Transform the test data using fitted PCA
test_pca <- bake(
  trained_recipe,
  new_data = test
)

# Prepare plotting data
plot_data <- bind_cols(
  test_pca |> select(PC01, PC02),
  test_predictions
) |>
  mutate(
    Correct = if_else(
      Class == .pred_class,
      "Correct",
      "Incorrect"
    )
  )

# Plot test observations
ggplot(plot_data,
       aes(x = PC01, y = PC02,
           color = Class, shape = Correct)) +
  geom_point(size = 3) +
  scale_shape_manual(
    values = c("Correct" = 16,
               "Incorrect" = 4)
  ) +
  labs(
    title = "PCA Classification: Khan Test Data",
    subtitle = "Selected model: 15 PCs",
    x = "First Principal Component",
    y = "Second Principal Component",
    color = "Actual Class",
    shape = "Prediction"
  ) +
  theme_minimal()

# Predict probabilities for the four cancer classes
test_probabilities <- predict(
  final_pca,
  new_data = test,
  type = "prob"
)

# Reshape probabilities for plotting
prob_plot <- test_probabilities |>
  mutate(Patient = factor(row_number())) |>
  pivot_longer(
    cols = starts_with(".pred_"),
    names_to = "Class",
    values_to = "Probability"
  )

# Plot probabilities
ggplot(prob_plot,
       aes(x = Patient,
           y = Probability,
           fill = Class)) +
  geom_col() +
  labs(
    title = "Predicted Cancer Class Probabilities",
    subtitle = "PCA Classification with 15 PCs",
    x = "Test Observation",
    y = "Predicted Probability"
  ) +
  theme_minimal()
#########################################

# library(dplyr)
# library(tidyr)
# library(ggplot2)
# 
# # Get accuracy for every fold and repetition
# fold_results <- collect_metrics(
#   pca_cv,
#   summarize = FALSE
# ) |>
#   filter(.metric == "accuracy")
# 
# print(fold_results)
# 
# repeat_results <- fold_results |>
#   group_by(num_comp, id) |>
#   summarise(
#     repeat_accuracy = mean(.estimate),
#     .groups = "drop"
#   )
# 
# print(repeat_results)
# 
# uncertainty_results <- repeat_results |>
#   group_by(num_comp) |>
#   summarise(
#     mean_accuracy = mean(repeat_accuracy),
#     sd_accuracy = sd(repeat_accuracy),
#     min_accuracy = min(repeat_accuracy),
#     max_accuracy = max(repeat_accuracy),
#     .groups = "drop"
#   )
# 
# print(uncertainty_results)
# 
# ggplot(uncertainty_results,
#        aes(x = num_comp, y = mean_accuracy)) +
#   geom_line() +
#   geom_point() +
#   geom_errorbar(
#     aes(
#       ymin = pmax(0, mean_accuracy - sd_accuracy),
#       ymax = pmin(1, mean_accuracy + sd_accuracy)
#     ),
#     width = 0.2
#   ) +
#   labs(
#     x = "Number of Principal Components",
#     y = "Mean CV Accuracy",
#     title = "PCA Classification: Accuracy and Variability",
#     subtitle = "Error bars represent +/- 1 SD across repetitions"
#   ) +
#   theme_minimal()
# 
# comparison <- repeat_results |>
#   filter(num_comp %in% c(15, 18)) |>
#   select(id, num_comp, repeat_accuracy) |>
#   pivot_wider(
#     names_from = num_comp,
#     values_from = repeat_accuracy,
#     names_prefix = "PC_"
#   ) |>
#   mutate(
#     difference = PC_18 - PC_15
#   )
# 
# print(comparison)
# 
# comparison |>
#   summarise(
#     mean_difference = mean(difference),
#     sd_difference = sd(difference)
#   )

