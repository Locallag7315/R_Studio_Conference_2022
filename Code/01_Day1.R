#package install
install.packages(c("doParallel", "embed", "forcats",
                   "lme4", "ranger", "remotes", "rpart", 
                   "rpart.plot", "stacks", "tidymodels",
                   "vetiver", "xgboost"))
remotes::install_github("topepo/ongoal@v0.0.2")

##################################
########## Begin coding ##########
##################################

#Load Libraries
library(tidymodels)

#Load Data
data("tree_frogs",package = "stacks")

#Transform Data
#t_o_d is time of day
#latency is how long is takes the frogs to hatch
tree_frogs <- tree_frogs %>%
  mutate(t_o_d = factor(t_o_d),
         age = age / 86400) %>%
  filter(!is.na(latency)) %>%
  select(-c(clutch, hatched))

####################################
########## Splitting Data ##########
####################################

#Split data
set.seed(123)
frog_split <- initial_split(tree_frogs)
frog_split

#Get train and test set
frog_train <- training(frog_split)
frog_test <- testing(frog_split)

#re-splitting to hold 20% of data in the test set (as opposed to 25% default)
set.seed(7315)
new_frog_split <- initial_split(tree_frogs, prop = 4/5)
frog_train_80 <- training(new_frog_split)
frog_test_20 <- testing(new_frog_split)

#####################################
########## Exploring  Data ##########
#####################################

#Descriptive Stats
summary(frog_train_80$treatment)
summary(frog_train_80$reflex)
summary(frog_train_80$age)
summary(frog_train_80$t_o_d)
summary(frog_train_80$latency)

plot(frog_train_80$age)
plot(frog_train_80$latency)
table(frog_train_80$treatment)
table(frog_train_80$reflex)
table(frog_train_80$t_o_d)

#How does latency differ amongst categorical variables
ggplot(frog_train_80, aes(latency)) +
  geom_histogram(bins = 20)
ggplot(frog_train_80, aes(latency, treatment, fill = treatment)) +
  geom_boxplot(alpha = 0.5, show.legend = FALSE)
frog_train_80 %>%
  ggplot(aes(latency, reflex, fill = reflex)) +
  geom_boxplot(alpha = 0.3, show.legend = FALSE)
ggplot(frog_train, aes(age, latency, color = reflex)) +
  geom_point(alpha = .8, size = 2)

#########################################################
########## Splitting  Data with stratification ##########
#########################################################
#set seed to be reproducible
set.seed(123)
#adding strata = latency makes it sample within bins that are created for a variable (we choose latency as this is our dv)
frog_split_strat <- initial_split(tree_frogs, prop = 0.8, strata = latency)
frog_split_strat
frog_train_strat <- training(frog_split_strat)
frog_test_strat <- testing(frog_split_strat)

##############################
########## Modeling ##########
##############################

#Engine (lm is linear regression). lm is in r, but doesn't have to be in r
linear_reg() %>% 
  set_engine("lm") #other options can be set here. Example "glmnet"

#Mode - tells what type of problems we should solve (regression, classification, sensored regression, etc.)
decision_tree() %>% 
  set_mode("regression")

#Model Specifics - Linear Regression
lm_spec <-
  linear_reg() %>% 
  set_mode("regression")

#Fit Regression Model
lm_fit <-
  workflow(latency ~ ., tree_spec) %>% 
  fit(data = frog_train_strat) 

#Predict
predict(lm_fit, new_data = frog_test_strat) #Returns predicted latency for the test set

#Augment - prediction appended to the test set
augment(lm_fit, new_data = frog_test_strat)

#Decision tree
#Model Specifics - Linear Regression
tree_spec <-
  decision_tree() %>% 
  set_mode("regression")
tree_wflow <- workflow() %>%
  add_formula(latency ~ .) %>%
  add_model(tree_spec)


#Fit Regression Model
tree_fit <-
  workflow(latency ~ ., tree_spec) %>% 
  fit(data = frog_train_strat) 

#Predict
predict(tree_fit, new_data = frog_test_strat) #Returns predicted latency for the test set

#Augment - prediction appended to the test set
augment(tree_fit, new_data = frog_test_strat)

library(rpart.plot)
tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint = FALSE)

#Extract the engine from my fitted workflow
extract_fit_engine(lm_fit)
extract_fit_engine(tree_fit)

#Summary of lm
extract_fit_engine(lm_fit) %>% summary()

######################################
########## Deploy the Model ##########
######################################
library(vetiver)
v <- vetiver_model(tree_fit, "frog_hatching") #Vetiver contains info needed to deploy
v
library(plumber)
pr <- pr() %>%
  vetiver_api(v)

pr_run(pr)

#######################################
########## Model Performance ##########
#######################################
augment(tree_fit, new_data = frog_test_strat) %>%
  metrics(latency, .pred)

#You can get metrics by group
augment(tree_fit, new_data = frog_test_strat) %>%
  group_by(reflex) %>%
  rmse(latency, .pred)

#Metrics set
frog_metrics <- metric_set(rmse, msd)
augment(tree_fit, new_data = frog_test_strat) %>%
  frog_metrics(latency, .pred)

#Predicting on the training set
tree_fit %>%
  augment(frog_train_strat)
tree_fit %>%
  augment(frog_train_strat) %>%
  rmse(latency, .pred)

#Compare against test
tree_fit %>%
  augment(frog_test_strat) %>%
  rmse(latency, .pred)

#This is for demonstration purposes, don't use test until the very end

#Compute MAE for train and test
tree_fit %>%
  augment(frog_train_strat) %>%
  mae(latency, .pred)
tree_fit %>%
  augment(frog_test_strat) %>%
  mae(latency, .pred)

#R-Squared
tree_fit %>%
  augment(frog_train_strat) %>%
  rsq(latency, .pred)
tree_fit %>%
  augment(frog_test_strat) %>%
  rsq(latency, .pred)

#R-Squared lm
lm_fit %>%
  augment(frog_train_strat) %>%
  rsq(latency, .pred)
lm_fit %>%
  augment(frog_test_strat) %>%
  rsq(latency, .pred)

######################################
########## Cross Validation ##########
######################################
#Data not used in fitting is used in predicting
#This is done a bunch of time using re-sampling
#Do this using the training data, don't touch the test in this
# 10 fold is default v=10
#you can stratify the sample similar to when the data was originally sampled
set.seed(123)
frog_folds <- vfold_cv(frog_train_strat, strata = "latency")
frog_folds$splits[1:10]

#Fitting the re-samples
#Each fold will be a new model
tree_res <- fit_resamples(tree_wflow, frog_folds)
tree_res

#Evaluate performance across the folds
#This can be compared against the train and test metrics prior to cross-validation
tree_res %>%
  collect_metrics() %>% #The metrics are computed on the "test" parts of the cv analysis while the models are trained on the "training" part
  select(.metric, mean, n)

# Save the assessment set results
ctrl_frog <- control_resamples(save_pred = TRUE) #returns a predictions column back for the individual folds
tree_res <- fit_resamples(tree_wflow, frog_folds, control = ctrl_frog)

tree_preds <- collect_predictions(tree_res)
tree_preds

#Plot all predictions across the folds
tree_preds %>% 
  ggplot(aes(latency, .pred, color = id)) + 
  geom_abline(lty = 2, col = "gray", size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()

###################################
########## Bootstrapping ##########
###################################

#This results in equal analysis set sizes
#Re-sampling with replacement 
#default is 25 times
set.seed(3214)
bootstraps(frog_train_strat, times = 10, strata = "latency")

#Create another validation set using another re-sampling method
set.seed(1234)
validation_split(frog_train_strat, strata = "latency")

####################################
########## Random Forests ##########
####################################

#A bunch of decision trees
#All the trees make individual predictions
#An average of the decision trees is taken
#Boot-strapping is conducted so each tree has a different set of data

#Specify the model
rf_spec <- rand_forest(trees = 1000, mode = "regression")
rf_spec

#Make the workflow
rf_wflow <- workflow(latency ~ ., rf_spec)
rf_wflow

#Cross validation (10-fold)
set.seed(123)
frog_folds_rf <- vfold_cv(frog_train_strat, strata = "latency")

#Re-sample and keep predicted column
control_rf <- control_resamples(save_pred = TRUE)
rf_res <- fit_resamples(rf_wflow, frog_folds_rf, control = control_rf)
rf_preds <- collect_predictions(rf_res)

#Compute metrics
rf_res %>%
  collect_metrics() %>% #The metrics are computed on the "test" parts of the cv analysis while the models are trained on the "training" part
  select(.metric, mean, n)

#Plot predictions vs. actual
rf_preds %>% 
  ggplot(aes(latency, .pred, color = id)) + 
  geom_abline(lty = 2, col = "gray", size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()

########################################################
########## Comparing Multiple Model Workflows ##########
########################################################
#Use workflow set
#This will fit all 3, apply cv, and rank the results
workflow_set(list(latency~.),list(tree_spec, rf_spec, lm_spec)) %>% 
  workflow_map("fit_resamples", resamples = frog_folds) %>% 
  rank_results()

###################################
########## The Final Fit ##########
###################################
#assuming we use random forest we want to fit a final time on all training data
final_fit <- last_fit(rf_wflow, frog_split_strat)
final_fit

#metrics
collect_metrics(final_fit) #these are computed on the test set

#collect predictions
collect_predictions(final_fit) #these are predictions on the test set

#plot
collect_predictions(final_fit) %>%
  ggplot(aes(latency, .pred)) + 
  geom_abline(lty = 2, col = "deeppink4", size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()

#Extract Workflow for final fit for rf
extract_workflow(final_fit)
############################
########## Stacks ##########
############################
#Combine multiple model predictions into 1 prediction
#Need to save_pred and save_workflow both true
stack_ctrl <- control_resamples(save_pred = TRUE, save_workflow = TRUE)
#Linear regression
lr_res <- 
  # define model spec
  linear_reg() %>%
  set_mode("regression") %>%
  # add to workflow
  workflow(preprocessor = latency ~ .) %>%
  # fit to resamples
  fit_resamples(frog_folds, control = stack_ctrl)
#Random Forest
rf_res <- 
  # define model spec
  rand_forest() %>%
  set_mode("regression") %>%
  # add to workflow
  workflow(preprocessor = latency ~ .) %>%
  # fit to resamples
  fit_resamples(frog_folds, control = stack_ctrl)

#initialize the data stack object
library(stacks)
frog_st <- stacks()

#Add candidates to the stack
frog_st <- frog_st %>%
  add_candidates(lr_res) %>%
  add_candidates(rf_res)

#Combine the predictions
frog_st_res <- frog_st %>%
  blend_predictions()

#Fit the final weights
frog_st_res <- frog_st_res %>%
  fit_members()

#Predict on new data
frog_test %>%
  select(latency) %>%
  bind_cols(
    predict(frog_st_res, frog_test)
  ) %>%
  ggplot(aes(latency, .pred)) + 
  geom_abline(lty = 2, 
              col = "deeppink4", 
              size = 1.5) +
  geom_point(alpha = 0.5) +
  coord_obs_pred()

