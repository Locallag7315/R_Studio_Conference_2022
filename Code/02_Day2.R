#package install
install.packages(c("doParallel", "embed", "forcats",
                   "lme4", "ranger", "remotes", "rpart", 
                   "rpart.plot", "stacks", "tidymodels",
                   "vetiver", "xgboost"))
remotes::install_github("topepo/ongoal@v0.0.2")

##################################
########## Begin coding ##########
##################################
#Building a model to see if a shot is on goal
#Load Libraries
library(tidymodels)
library(ongoal)

#Prioritize tidymodels
tidymodels_prefer()

#Take a peek at the data
glimpse(season_2015)

####################################
########## Splitting Data ##########
####################################
#We will split into train and test, and then split the train into train and validation for further testing
#Because we have over 12,000 observations we don't feel the need to do 10-fold cross validation

#Set seed
set.seed(23)

#Split the data
nhl_split <- initial_split(season_2015, prop = 3/4)
nhl_split

#Split into training/val vs. test
nhl_train_and_val <- training(nhl_split)
nhl_test  <- testing(nhl_split)

## not testing
nrow(nhl_train_and_val)
#9110

## testing
nrow(nhl_test)
#3037

#Split into val set
set.seed(234)
nhl_val <- validation_split(nhl_train_and_val, prop = 0.80)
nhl_val

#Actual Training data from the val split
nhl_train <- analysis(nhl_val$splits[[1]])

##################################
########## Explore Data ##########
##################################
set.seed(100)
nhl_train %>% 
  sample_n(200) %>%
  plot_nhl_shots(emphasis = position)

#Correlation
data_analysis_set <- nhl_train
data_analysis_set$on_goal <- ifelse(data_analysis_set$on_goal == "yes",1,0)
data_analysis_set$pp_ind <- ifelse(data_analysis_set$strength == "power_play",1,0)
data_analysis_set$ot_ind <- ifelse(data_analysis_set$period_type == "overtime",1,0)
cor(data_analysis_set$on_goal, data_analysis_set$coord_x)
cor(data_analysis_set$on_goal, data_analysis_set$pp_ind)
cor(data_analysis_set$on_goal, data_analysis_set$ot_ind)
cor(data_analysis_set$on_goal, data_analysis_set$offense_goal_diff)

#plot to explore
nhl_train %>% 
  ggplot(aes(x=position,fill=on_goal))+
  geom_bar()

nhl_train %>% 
  ggplot(aes(x=strength))+
  geom_bar() +
  facet_wrap(~on_goal)+
  scale_y_log10()
  
#####################################
########## Recipes Package ##########
#####################################
#First recipe
nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train)

#Get summary
summary(nhl_rec)

#Create indicator (dummy variables) for all nominal predictors
#Use all_nominal_predictors as opposed to all_nominal because we want to keep our dv a factor
nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors())

#Filter out constant columns
#This would get rid of any columns that have no unique information
#This makes things faster
nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())

#Normalization
#This centers and scales all predictors
nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

#Reduce correlation
#Get rid of any predictors that are at least 90% correlated
nhl_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.9)

#Independent task
nhl_rec_test <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_zv(all_predictors())

#using a recipe in a workflow with logistic regression
nhl_indicators <-
  recipe(on_goal ~ ., data = nhl_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

set.seed(9)

#add recipe and logistic regression model to workflow
nhl_glm_wflow <-
  workflow() %>%
  add_recipe(nhl_indicators) %>%
  add_model(logistic_reg())

#Save the predictors
ctrl <- control_resamples(save_pred = TRUE)

#Resampling object is validation split after separating from training
#This trains on the train and tests on the test
nhl_glm_res <-
  nhl_glm_wflow %>%
  fit_resamples(nhl_val, control = ctrl)

#Evaluate performance on validation set
collect_metrics(nhl_glm_res)

#Fit workflow with a different recipe
#I used the recipe where we did a lot above (nhl_rec)
#I used a logistic regression model
nhl_glm_wflow_new <-
  workflow() %>%
  add_recipe(nhl_rec) %>%
  add_model(logistic_reg())

#Expect warning - this is because of linear dependencies between indicator variables (this is okay)
#We can remove this with step_lincomb(all_numeric_predictors())
nhl_glm_res_new <-
  nhl_glm_wflow_new %>%
  fit_resamples(nhl_val, control = ctrl)

#Get metrics
collect_metrics(nhl_glm_res_new) #got worse lol

#Get predictions
# Since we used `save_pred = TRUE`
glm_val_pred <- collect_predictions(nhl_glm_res)
glm_val_pred %>% slice(1:7)

#My predictions
glm_val_pred_new <- collect_predictions(nhl_glm_res_new)

#ROC Curve
# Assumes _first_ factor level is event; there are options to change that
#Truth is dv, estimate is what we would want to predict (could be yes or no)
roc_curve_points <- glm_val_pred %>% roc_curve(truth = on_goal, estimate = .pred_yes)
roc_curve_points %>% slice(1, 50, 100)
#AUC
glm_val_pred %>% roc_auc(truth = on_goal, estimate = .pred_yes)

#Plot ROC
autoplot(roc_curve_points)

#Plot ROC Curve for my model
roc_curve_me <- glm_val_pred_new %>% 
  roc_curve(truth = on_goal, estimate = .pred_yes)
autoplot(roc_curve_me)

#Get AUC
auc_me <- glm_val_pred_new %>% 
  roc_auc(truth = on_goal, estimate = .pred_yes)

#Including the players in the model
#We have roughly 600 players and can't include all of them in the model
#use effect encoding to replace the player column with the effect of each player on the outcome
#for a logistic regression you replace the player with their log odds of getting a shot on goal
#We use partial pooling to replace the raw rate with an estimate of being on goal. The raw rate is the actual rate on goal, which is great for a lot fo data but not great for someone who has like 1 shot and it went on goal (on goal rate of 100%). The average of the data is used to substiture a more accurate esimate for him
#Use the embed package for this
library(embed)
#Incorporate it into the recipe with step_lencode_mixed
#important to do before step dummy
nhl_effect_rec <-
  recipe(on_goal ~ ., data = nhl_train) %>%
  step_lencode_mixed(player, outcome = vars(on_goal)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

#Create workflow based on our old workflow and update recipe instead 
nhl_effect_wflow <-
  nhl_glm_wflow %>%
  update_recipe(nhl_effect_rec)

nhl_effect_res <-
  nhl_effect_wflow %>%
  fit_resamples(nhl_val, control = ctrl)

collect_metrics(nhl_effect_res) #this was not great still

#Incorporate the angle of the shot
#this is shots that were less than 25 degrees
nhl_angle_rec <-
  nhl_indicators %>%
  step_mutate(
    angle = abs(atan2(abs(coord_y), (89 - abs(coord_x))) * (180 / pi))
  )
#Distance
nhl_distance_rec <-
  nhl_angle_rec %>%
  step_mutate(
    distance = sqrt((89 - abs(coord_x))^2 + abs(coord_y)^2),
    distance = log(distance)
  )
#Behind the goal
nhl_behind_rec <-
  nhl_distance_rec %>%
  step_mutate(
    behind_goal_line = ifelse(abs(coord_x) >= 89, 1, 0)
  )
#Fit new recipe
set.seed(9)

nhl_glm_set_res <-
  workflow_set(
    list(`1_dummy` = nhl_indicators, `2_angle` = nhl_angle_rec, 
         `3_dist` = nhl_distance_rec, `4_bgl` = nhl_behind_rec),
    list(logistic = logistic_reg())
  ) %>%
  workflow_map(fn = "fit_resamples", resamples = nhl_val, verbose = TRUE, control = ctrl)

#How to see the models:
#below code isn't right
extract_model(nhl_glm_set_res[1,])


#Compare the recipes
library(forcats)
collect_metrics(nhl_glm_set_res) %>%
  filter(.metric == "roc_auc") %>%
  mutate(
    features = gsub("_logistic", "", wflow_id), 
    features = fct_reorder(features, mean)
  ) %>%
  ggplot(aes(x = mean, y = features)) +
  geom_point(size = 3) +
  labs(y = NULL, x = "ROC AUC (validation set)")

#Apply recipe transformations to data independently using prep if you need to troubleshoot
#Make predictions on the data with a recipe using prep
fitted_rec <- nhl_angle_rec %>%  prep()
baked <- fitted_rec %>% bake(new_data=nhl_train)

############################################
########## Tuning Hyperparameters ##########
############################################
#Determine what values are best for tunable paramters in your model
#The 2 main way to do this are grid search or iterative search

#Make a recipe using previous pieces and also include tuning over angle and distance
#We will assess if angle and distance have a non-linear relationship (spline) with the shots on goal (dv)
#Splines help to capture non-linear relationships
#The more spline terms the better the fit, but this can be overfit
#The tuning of the parameters can be optimized
glm_rec <-
  recipe(on_goal ~ ., data = nhl_train) %>%
  step_lencode_mixed(player, outcome = vars(on_goal)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_mutate(
    angle = abs(atan2(abs(coord_y), (89 - abs(coord_x))) * (180 / pi)),
    distance = sqrt((89 - abs(coord_x))^2 + abs(coord_y)^2),
    distance = log(distance),
    behind_goal_line = ifelse(abs(coord_x) >= 89, 1, 0)
  ) %>%
  step_rm(coord_x, coord_y) %>%
  step_zv(all_predictors()) %>%
  step_ns(angle, deg_free = tune("angle")) %>% #this shows you want to eventually tune angle on degrees of freedom
  step_ns(distance, deg_free = tune("distance")) %>% #this shows you want to eventually tune distance on degress of freedom
  step_normalize(all_numeric_predictors())

#Workflow adding the new recipe
glm_spline_wflow <-
  workflow() %>%
  add_model(logistic_reg()) %>%
  add_recipe(glm_rec)

#Create a grid using our workflow
glm_spline_wflow %>% 
  extract_parameter_set_dials()

set.seed(2)
grid <- 
  glm_spline_wflow %>% 
  extract_parameter_set_dials() %>% 
  grid_latin_hypercube(size = 25) #try and get 25 rows, could be 25 but will return less if it doesn't think any value will come from that
#the grid will try to cover the space of tunable parameters based on
grid

#independent task of creating regular grid
set.seed(123)
grid_ind <- 
  glm_spline_wflow %>% 
  extract_parameter_set_dials() %>% 
  grid_regular(levels=25)
grid_ind  #this will cover the full space, only 225 bc df only ranges from 1-15

#update parameter methods if you need to
set.seed(2)
grid <- 
  glm_spline_wflow %>% 
  extract_parameter_set_dials() %>% 
  update(angle = spline_degree(c(2L, 20L)), #2 to 20 is the range of possible splines that we can try
         distance = spline_degree(c(2L, 20L))) %>% 
  grid_latin_hypercube(size = 25) #25 is the number of models we want to build to cover the space. latin hypercube covers the space of 2 to 20 as effectively as possible with 25 dots

grid

#The results of the grid plotted (will result in a model per number of rows in grid)
grid %>% 
  ggplot(aes(angle, distance)) +
  geom_point(size = 4)

#Call the tuning function to do a grid search of all possible values
#This will create 25 models
set.seed(9)
ctrl <- control_grid(save_pred = TRUE, parallel_over = "everything")

glm_spline_res <-
  glm_spline_wflow %>%
  tune_grid(resamples = nhl_val, grid = grid, control = ctrl)
  
glm_spline_res

#Collect the metrics on the model results
collect_metrics(glm_spline_res)

#plot grid accuracy results on the tuned parameters
autoplot(glm_spline_res)

#show the best combination of parameters based on auc
show_best(glm_spline_res, metric = "roc_auc")

#Independent task
#Try to get a simpler model in terms of splines (less wiggly) that is still good
#Use select_by_pct_loss()
#this gives the best possible model with the least complex value of distance that is still within 2% auc of the best model
select_by_pct_loss(x=glm_spline_res, 
                   distance,
                   metric = "roc_auc",
                   limit = 2)

###################################
########## Boosted Trees ##########
###################################
#Typically one of the best performing algorithm
#Ensemble many decision tree models
#Each tree is dependent on the previous one
#These trees need to be tuned, the performance will not be good without tuning
#We use early stopping to stop boosting when we get worse results

#Boosting code
#Boosted Tree Model Specification
#We set the number of trees high, but are using early stopping with the validation proportion
#1/10 means that 1/10 of the training data (after splitting with val set) is used to evaluate if the trees are getting better
#We are tuning on 5 parameters
#Mode is classification
xgb_spec <-
  boost_tree(
    trees = 500, min_n = tune(), stop_iter = tune(), tree_depth = tune(),
    learn_rate = tune(), loss_reduction = tune()
  ) %>%
  set_mode("classification") %>% 
  set_engine("xgboost", validation = 1/10) # <- for better early stopping

#Recipe
#encoding on player data using partial pooling
#xgboost requires numbers so all nominal predictors are dummies
#remove all uniform predictors (all 0 or 1 - they will add nothing)
xgb_rec <- 
  recipe(on_goal ~ ., data = nhl_train) %>% 
  step_lencode_mixed(player, outcome = vars(on_goal)) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

#Workflow
xgb_wflow <- 
  workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(xgb_rec)

#Tune the model
set.seed(9)

xgb_res <-
  xgb_wflow %>%
  tune_grid(resamples = nhl_val, grid = 15, control = ctrl) # automatic grid now!

xgb_res

#plot results
autoplot(xgb_res)

#Regenerate the features we created earlier
#Recipe to handcraft features
coord_rec <- 
  xgb_rec %>%
  step_mutate(
    angle = abs(atan2(abs(coord_y), (89 - abs(coord_x))) * (180 / pi)),
    distance = sqrt((89 - abs(coord_x))^2 + abs(coord_y)^2),
    distance = log(distance),
    behind_goal_line = ifelse(abs(coord_x) >= 89, 1, 0)
  ) %>% 
  step_rm(coord_x, coord_y)

#Workflow for updated recipe
xgb_coord_wflow <- 
  workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(coord_rec)

#tune again with updated model and handcrafted workflow
set.seed(9)
xgb_coord_res <-
  xgb_coord_wflow %>%
  tune_grid(resamples = nhl_val, grid = 20, control = ctrl)

#Get best from original xgboost and the handcrafted xgboost tunings
show_best(xgb_res, metric = "roc_auc")
show_best(xgb_coord_res, metric = "roc_auc")

#Best Linear Regression Results from earlier
glm_spline_res %>% 
  show_best(metric = "roc_auc", n = 1) %>% 
  select(.metric, .estimator, mean, n, std_err, .config)

#Best Boosting Results
xgb_coord_res %>% 
  show_best(metric = "roc_auc", n = 1) %>% 
  select(.metric, .estimator, mean, n, std_err, .config)

#Logistic Regression is the best so we want it as our model
best_auc <- select_best(glm_spline_res, metric = "roc_auc")
best_auc

#Finalize workflow function updates the tunable parameters to the values we chose
glm_spline_wflow <-
  glm_spline_wflow %>% 
  finalize_workflow(best_auc)

glm_spline_wflow

#Final Fit time (oh yeah)
#This trains on the full train data (including the validation set) and uses the test data for assessment
test_res <- 
  glm_spline_wflow %>% 
  last_fit(split = nhl_split)

test_res

#Validation results
#Metrics on the validation set
glm_spline_res %>% 
  show_best(metric = "roc_auc", n = 1) %>% 
  select(.metric, mean, n, std_err)

#Metrics on the test data
test_res %>% collect_metrics() #this is good

#Final fitted workflow (this contains all the parameters in the model)
final_glm_spline_wflow <- 
  test_res %>% 
  extract_workflow()

# use this object to predict or deploy 
# You can deploy using vetiver
predict(final_glm_spline_wflow, nhl_test[1:3,])

