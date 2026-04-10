# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse) 
library(caret) 
# remotes::install_version("xgboost", version = "1.6.0.1", repos = "https://cran.r-project.org") 
library(xgboost)
library(parallel) # added `parallel` package to conduct parallelized models
library(doParallel) # added `doParallel` package to conduct parallelized models
set.seed(123) 

# Data Import and Cleaning
gss_import_tbl <- haven::read_sav("../data/GSS2016.sav", user_na = T) %>% # I updated the import tibble as discussed in class outbrief
  haven::zap_missing() %>% 
  filter(!is.na(mosthrs))
gss_tbl <- gss_import_tbl %>% 
  select(-c(hrs1, hrs2)) %>% 
  select(which(colMeans(is.na(.)) < 0.75)) %>% 
  mutate(across(everything(), as.numeric)) %>% 
  as_tibble() 

# Visualization
(ggplot(gss_tbl, aes(x = mosthrs)) + 
  geom_histogram(binwidth = 4)) %>%  
  ggsave(filename = "../figs/hours_histogram.png", width = 1920, height = 1080, units = "px", dpi = 300)

# Analysis 

## Define training and test sets
holdout_indices <- createDataPartition(gss_tbl$mosthrs, p = .25, list=F) # I had this structure incorrect in Week 10 project; updated after outbrief
gss_holdout <- gss_tbl[holdout_indices,] 
gss_training <- gss_tbl[-holdout_indices,] 

## Run k-fold testing hyperparameters on training set
# I modified each of the models below by adding `system.time` and storing the results in a new variable, which I then added to my table2_tbl

mod1_tm <- system.time({ # added wrapper for system.time around all models, storing the variable for table2_tbl construction; experimented with library(tictoc) and library(microbenchmark)
  model1 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "lm", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})


local_cluster <- makeCluster(7) # using `detectCores()` I identified 8 cores, subtracting 1, I began the local cluster for parallelization
registerDoParallel(local_cluster) # activate cluster
mod1_tm_par <- system.time({
  model1 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "lm", 
    preProcess = c("medianImpute","center","nzv", "scale"),  
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})
stopCluster(local_cluster) # deactivate cluster
registerDoSEQ() # explicit registration of the backend of a cluster


mod2_tm <- system.time({
  model2 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "glmnet", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneGrid = expand.grid( 
      alpha = c(0,1), 
      lambda = seq(0.0001, 0.1, length = 10) 
    ),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})

local_cluster <- makeCluster(7) # activate cluster
registerDoParallel(local_cluster) # start cluster again
mod2_tm_par <- system.time({
  model2 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "glmnet", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneGrid = expand.grid( 
      alpha = c(0,1), 
      lambda = seq(0.0001, 0.1, length = 10) 
    ),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})
stopCluster(local_cluster) # end cluster
registerDoSEQ() # explicit registration of cluster backend

mod3_tm <- system.time({
  model3 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, # I revised my week 10 code to simplify with tuneLength rather than tuneGrid for code optimization
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})

local_cluster <- makeCluster(7) # activate cluster
registerDoParallel(local_cluster) # restart cluster
mod3_tm_par <- system.time({
  model3 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, # I revised my week 10 code to simplify with tuneLength rather than tuneGrid for code optimization
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})
stopCluster(local_cluster) # end cluster
registerDoSEQ() # explicit registration of cluster backend

mod4_tm <- system.time({ 
  model4 <- train(
    mosthrs ~., 
    gss_training, 
    na.action = na.pass, 
    method = "xgbTree", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, # I revised my week 10 code to simplify with tuneLength rather than tuneGrid for code optimization
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    ) 
  )
})

local_cluster <- makeCluster(7) # activate cluster
registerDoParallel(local_cluster) # restart cluster
mod4_tm_par <- system.time({ 
  model4 <- train(
    mosthrs ~., 
    gss_training, 
    na.action = na.pass, 
    method = "xgbTree", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, # I revised my week 10 code to simplify with tuneLength rather than tuneGrid for code optimization
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    ) 
  )
})
stopCluster(local_cluster) # end cluster
registerDoSEQ() # explicit registration of cluster backend

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
  ho_rsq = str_remove(round(c(postResample(pred1, gss_holdout$mosthrs)["Rsquared"], 
                              postResample(pred2, gss_holdout$mosthrs)["Rsquared"], 
                              postResample(pred3, gss_holdout$mosthrs)["Rsquared"], 
                              postResample(pred4, gss_holdout$mosthrs)["Rsquared"]), 2), "^0")
) %>% 
  write_csv(file = "../figs/table1.csv") 

# Table 2 created below using code for `original` and `parallelized` that extracts the [[3]] element, which is the elapsed time of the system.time() function. I obtained in two columns, one for original and the other for parallelized. 
table2_tbl <- tibble( 
  original = c(mod1_tm[[3]],mod2_tm[[3]],mod3_tm[[3]],mod4_tm[[3]]),
  parallelized = c(mod1_tm_par[[3]],mod2_tm_par[[3]],mod3_tm_par[[3]],mod4_tm_par[[3]])
) %>% 
  write_csv(file = "../figs/table2.csv") 

# The non-linear tree-based models benefited most from parallelization. This is because parallelization increased the logical processors that could conduct the simultaneous operations required to reduce the speed of more complex models. 
# The largest difference for the fastest model was a 155 second increase to speed in the xgbTree model. The OLS linear model which was parallelized was only 1.56 seconds faster. This is because xgbTree is very complex, where the unparallelized model has a 3.98 relative time compared to the parallelized model. 
# I would recommend the use of model4, the xgbTree model. While the xgbTree model has the second most variance explained (after Random Forest - RF) of all the models at the current hyperparameter tuning settings, its out of sample difference is a bit larger than RF. When parallelized, however the xgbTree model has a relative time 1.28 times faster than the RF parallelized model. Taken together the variance explained is higher at a much faster speed.  
