# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse) 
library(caret) 
library(xgboost)
library(parallel) 
library(doParallel) 
set.seed(123) 

# Data Import and Cleaning
gss_import_tbl <- haven::read_sav("../data/GSS2016.sav", user_na = T) %>% 
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


local_cluster <- makeCluster(7) 
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

local_cluster <- makeCluster(7) 
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
    tuneLength = 3, 
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    )
  )
})

local_cluster <- makeCluster(7) 
registerDoParallel(local_cluster) 
mod3_tm_par <- system.time({
  model3 <- train(
    mosthrs ~ ., 
    gss_training, 
    na.action = na.pass, 
    method = "ranger", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, 
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
    tuneLength = 3, 
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    ) 
  )
})

local_cluster <- makeCluster(7) 
registerDoParallel(local_cluster) 
mod4_tm_par <- system.time({ 
  model4 <- train(
    mosthrs ~., 
    gss_training, 
    na.action = na.pass, 
    method = "xgbTree", 
    preProcess = c("medianImpute","center","nzv", "scale"), 
    tuneLength = 3, 
    trControl=trainControl(
      method="cv", number=10, verboseIter=T 
    ) 
  )
})
stopCluster(local_cluster) 
registerDoSEQ() 

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


table2_tbl <- tibble( 
  original = str_remove(round(c(mod1_tm[[3]],mod2_tm[[3]],mod3_tm[[3]],mod4_tm[[3]]), 2), "^0"),
  parallelized = str_remove(round(c(mod1_tm_par[[3]],mod2_tm_par[[3]],mod3_tm_par[[3]],mod4_tm_par[[3]]), 2), "^0"),
) %>% 
  write_csv(file = "../figs/table2.csv") 