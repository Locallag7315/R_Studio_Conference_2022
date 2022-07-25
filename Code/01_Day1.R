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

#tree_spec editing
library(baguette)
tree_Spec <- bag_tree() %>% 
  set_mode("classification")


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


