# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# library(haven) 
library(tidyverse) 
library(caret) 
# remotes::install_version("xgboost", version = "1.6.0.1", repos = "https://cran.r-project.org") 
library(xgboost)
library(parallel) # added `paralell` package to conduct parallelized models
set.seed(123) 

# Data Import and Cleaning
gss_tbl <- haven::read_sav("../data/GSS2016.sav", user_na = T) %>% 
  haven::zap_missing() %>% 
  filter(!is.na(mosthrs)) %>% 
  rename(`work hours` = mosthrs) %>% 
  select(-c(hrs1, hrs2)) %>% 
  select(which(colMeans(is.na(.)) < 0.75)) %>% 
  mutate(across(where(haven::is.labelled), as.numeric)) %>% 
  mutate(`work hours` = as.numeric(`work hours`)) %>% 
  as_tibble() 

# Visualization
(ggplot(gss_tbl, aes(x = `work hours`)) + 
  geom_histogram(binwidth = 4)) %>%  
  ggsave(filename = "../figs/hours_histogram.png", width = 1920, height = 1080, units = "px", dpi = 300)

# Analysis 

## Define training and test sets
holdout_indices <- createDataPartition(gss_tbl$`work hours`, p = .25, list=F) 
gss_training <- gss_tbl[holdout_indices,] 
gss_holdout <- gss_tbl[-holdout_indices,] 

## Define consistent folds 
cv_control <- trainControl(
  method="cv", 
  number=10, 
  search = "random",
  verboseIter = T 
)

## Run k-fold testing hyperparameters on training set
# I modified each of the models below by adding `system.time` and storing the results in a new variable, which I then added to my table2_tbl
mod1_tm <- system.time({
  model1 <- train(
  `work hours` ~ ., 
  gss_training, 
  na.action = na.pass, 
  method = "lm", 
  preProcess = c("zv", "medianImpute","center","scale"), 
  trControl=cv_control 
  )
})

mod2_tm <- system.time({
  model2 <- train(
    `work hours` ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "glmnet", 
    preProcess = c("zv", "medianImpute","center","scale"),
    tuneGrid = expand.grid( 
      alpha = c(0,1), 
      lambda = seq(0.0001, 0.1, length = 10) 
    ),
    trControl = cv_control 
  )
})

mod3_tm <- system.time({
  model3 <- train(
    `work hours` ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("zv", "medianImpute","center","scale"),
    tuneGrid = expand.grid( 
      mtry = c(23, 178, 267, 535), 
      splitrule = c("variance", "extratrees"), 
      min.node.size = 5 
    ),
    trControl=cv_control 
  )
})

mod4_tm <- system.time({
  model4 <- train(
    `work hours` ~., 
    gss_training, 
    na.action = na.pass, 
    method = "xgbTree", 
    preProcess = c("zv", "medianImpute", "center", "scale"),
    tuneGrid = expand.grid( 
      nrounds = 300, 
      eta = c(0.01, 0.3),  
      max_depth = c(6, 9), 
      min_child_weight = c(1, 5), 
      gamma = c(0, 1), 
      colsample_bytree = c(.5, 1), 
      subsample = c(.5, 1) 
    ),
    trControl = cv_control 
  )
})

## Examine k-fold CV results
summary(resamples(list(model1,model2, model3, model4))) 
dotplot(resamples(list(model1,model2, model3, model4)), metric = "Rsquared")  

## Compare to holdout CV results
pred1 <- predict(model1, newdata = gss_holdout, na.action = na.pass) 
pred2 <- predict(model2, newdata = gss_holdout, na.action = na.pass)
pred3 <- predict(model3, newdata = gss_holdout, na.action = na.pass)
pred4 <- predict(model4, newdata = gss_holdout, na.action = na.pass)

# Publication
table1_tbl <- tibble( 
  algo = c(model1$method, model2$method, model3$method, model4$method), 
  cv_rsq = str_remove(round(c(getTrainPerf(model1)$TrainRsquared, 
                              getTrainPerf(model2)$TrainRsquared, 
                              getTrainPerf(model3)$TrainRsquared,
                              getTrainPerf(model4)$TrainRsquared), 2), "^0"),
  ho_rsq = str_remove(round(c(postResample(pred1, gss_holdout$`work hours`)["Rsquared"], 
                              postResample(pred2, gss_holdout$`work hours`)["Rsquared"], 
                              postResample(pred3, gss_holdout$`work hours`)["Rsquared"], 
                              postResample(pred4, gss_holdout$`work hours`)["Rsquared"]), 2), "^0")
) %>% 
  write_csv(file = "../figs/table1.csv") 

tbl2_tbl <- tibble(
  original = c(mod1_tm[[3]],mod2_tm[[3]],mod3_tm[[3]],mod4_tm[[3]]),
  parallelized = c(1,2,3,4)
) %>% 
  write_csv(file = "../figs/table2.csv") 

