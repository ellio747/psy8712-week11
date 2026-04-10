# Script Settings and Resources
# removed working directory
library(dplyr) #changed out of tidyverse to dplyr
library(caret) 
library(glmnet) # had to add these two as I was beta testing in interactive mode
library(ranger) # had to add these two as I was beta testing in interactive mode
library(xgboost)
library(parallel) 
library(doParallel) 
# library(tictoc) #added tictoc to benchmark my simpler file with tuneLength = 1 and resulted in 87.079 seconds elapsed
set.seed(123) 

# tic()
# Data Import and Cleaning
gss_import_tbl <- haven::read_sav("../data/GSS2016.sav", user_na = T) %>% 
  haven::zap_missing() %>% 
  filter(!is.na(mosthrs))
gss_tbl <- gss_import_tbl %>% 
  select(-c(hrs1, hrs2)) %>% 
  select(which(colMeans(is.na(.)) < 0.75)) %>% 
  mutate(across(everything(), as.numeric)) %>% 
  as_tibble() 

# Analysis 

## Define training and test sets
holdout_indices <- createDataPartition(gss_tbl$mosthrs, p = .25, list=F) 
gss_holdout <- gss_tbl[holdout_indices,] 
gss_training <- gss_tbl[-holdout_indices,] 

mod1_tm <- system.time({ 
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


local_cluster <- makeCluster(31) # added 31 cores
registerDoParallel(local_cluster) 
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
stopCluster(local_cluster) 
registerDoSEQ() 


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

local_cluster <- makeCluster(31) # added 31 cores
registerDoParallel(local_cluster) 
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
stopCluster(local_cluster) 
registerDoSEQ() 

mod3_tm <- system.time({
  model3 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, #updated tuneLength of 3 for MSI
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})

local_cluster <- makeCluster(31) # added 31 cores
registerDoParallel(local_cluster) 
mod3_tm_par <- system.time({
  model3 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, #updated tuneLength of 3 for MSI
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})
stopCluster(local_cluster) 
registerDoSEQ() 

mod4_tm <- system.time({ 
  model4 <- train(
    mosthrs ~., 
    gss_training, 
    na.action = na.pass, 
    method = "xgbTree", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, #updated tuneLength of 3 for MSI
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    ) 
  )
})

local_cluster <- makeCluster(31) # added 31 cores
registerDoParallel(local_cluster) 
mod4_tm_par <- system.time({ 
  model4 <- train(
    mosthrs ~., 
    gss_training, 
    na.action = na.pass, 
    method = "xgbTree", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, #updated tuneLength of 3 for MSI 
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    ) 
  )
})
stopCluster(local_cluster) 
registerDoSEQ() 

## Examine k-fold CV results
summary(resamples(list(model1,model2, model3, model4))) 

## Compare to holdout CV results
pred1 <- predict(model1, newdata = gss_holdout, na.action = na.pass) 
pred2 <- predict(model2, newdata = gss_holdout, na.action = na.pass)
pred3 <- predict(model3, newdata = gss_holdout, na.action = na.pass)
pred4 <- predict(model4, newdata = gss_holdout, na.action = na.pass)

# Publication
table3_tbl <- tibble( 
  algo = c(model1$method, model2$method, model3$method, model4$method), 
  cv_rsq = round(c(getTrainPerf(model1)$TrainRsquared, 
                              getTrainPerf(model2)$TrainRsquared, 
                              getTrainPerf(model3)$TrainRsquared,
                              getTrainPerf(model4)$TrainRsquared), 2),
  ho_rsq = round(c(postResample(pred1, gss_holdout$mosthrs)["Rsquared"], 
                              postResample(pred2, gss_holdout$mosthrs)["Rsquared"], 
                              postResample(pred3, gss_holdout$mosthrs)["Rsquared"], 
                              postResample(pred4, gss_holdout$mosthrs)["Rsquared"]), 2)
)
write.csv(table3_tbl, file = "../out/table3.csv") #changed to utils function rather than readr for simplicity


table4_tbl <- tibble( 
  supercomputer = round(c(mod1_tm[[3]],mod2_tm[[3]],mod3_tm[[3]],mod4_tm[[3]]), 2),
  supercomputer_31 = round(c(mod1_tm_par[[3]],mod2_tm_par[[3]],mod3_tm_par[[3]],mod4_tm_par[[3]]), 2) #msismall has 128 max cores with 1 max node - this is what I will request
) 
write.csv(table4_tbl, file = "../out/table4.csv") #changed to utils function rather than readr for simplicity

# The tree-based models benefited most from moving to supercomputing. This is because many more logical processors were conducting simultaneous operations. 
# In tree-based models (random forest and xgbTree) as the number of cores increases, the processing time decreases. For the linear models there is not nearly the same relationship and it appears from my trial that as cores increase, time stays the same.
# I would recommend the supercomputer-based xgbTree modell. While using the supercomputer the xgbTree accounted for less variance explained than the RF model, its out of sample variance explained was quite good. It also had a very fast processing time using a supercomputer and this was only using 31 cores and without tuning many hyperparameters. In both cases these could be boosted for a better model overall. 

# toc()