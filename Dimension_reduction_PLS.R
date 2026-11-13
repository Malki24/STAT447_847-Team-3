# Install once if needed
install.packages("BiocManager")
BiocManager::install("mixOmics")

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

gene_cols <- setdiff(names(train), "Class")

all(sapply(train[gene_cols], is.numeric))

# PLS preprocessing
pls_recipe <- recipe(Class ~ ., data = train) |>
  step_zv(all_predictors()) |>
  step_normalize(all_predictors()) |>
  step_pls(all_predictors(),
           outcome = Class,
           num_comp = tune(),
           options = list(scale = FALSE)) |>
  step_normalize(all_predictors())

# Multinomial classification model
pls_model <- multinom_reg() |>
  set_engine("nnet",
             trace = FALSE,
             maxit = 1000)
# Combine PLS recipe and classification model
pls_workflow <- workflow() |>
  add_recipe(pls_recipe) |>
  add_model(pls_model)

set.seed(847)

folds <- vfold_cv(
  train,
  v = 5,
  repeats = 10,
  strata = Class
)

# Candidate numbers of PLS components
pls_grid <- tibble(num_comp = 1:20)

# Tune PLS components
pls_cv <- tune_grid(
  pls_workflow,
  resamples = folds,
  grid = pls_grid,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = TRUE)
)

# View CV results
collect_metrics(pls_cv)

# Plot CV accuracy against number of components
autoplot(pls_cv)

# Select number of components with highest CV accuracy
best_pls <- select_best(
  pls_cv,
  metric = "accuracy"
)

print(best_pls)

# Fit the final model
final_pls <- finalize_workflow(
  pls_workflow,
  best_pls
) |>
  fit(data = train)

# Predict the 20 test observations
pls_predictions <- bind_cols(
  test[, "Class", drop = FALSE],
  predict(final_pls, new_data = test)
)

# Test accuracy
accuracy(
  pls_predictions,
  truth = Class,
  estimate = .pred_class
)

# Confusion matrix
conf_mat(
  pls_predictions,
  truth = Class,
  estimate = .pred_class
)

# PCA CV results
pca_results <- collect_metrics(pca_cv) |>
  mutate(Method = "PCA")

# PLS CV results
pls_results <- collect_metrics(pls_cv) |>
  mutate(Method = "PLS")

# Combine
comparison <- bind_rows(
  pca_results,
  pls_results
)

# Plot
ggplot(comparison,
       aes(x = num_comp,
           y = mean,
           color = Method)) +
  geom_line() +
  geom_point() +
  labs(
    x = "Number of Components",
    y = "Mean CV Accuracy",
    title = "PCA vs PLS: Cross-Validation Accuracy"
  ) +
  theme_minimal()
