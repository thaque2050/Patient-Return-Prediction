library(dplyr)
library(ggplot2)
library(reshape2)
library(caTools)
library(xgboost)
library(mice)


dtrain=read.csv('level1_dtrain.csv')

set.seed(1234)
#index=sample.split(dtrain2$RETURNYes,SplitRatio = 0.75)
#dtrain_train=dtrain2[index,]
#dtrain_test=dtrain2[!index,]


#XGB
l=ncol(dtrain)
X_train<-as.matrix(dtrain[,1:(l-1)])
Y_train<-dtrain[,l]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)

# Grid for model training
xgb_grid <- expand.grid(max_depth=c(50,100,500,1000),
                        min_child_weight = c(0,5,10,15,20,30,100,200),
                        subsample=c(0.1,0.5,0.625,0.7,1.0),
                        eta = c(0.01,0.03,0.05,0.09,0.1,0.5,0.7), 
                        colsample_bytree=c(0.1,0.5,0.7,1.0),
                        lambda = c(0,0.1,0.5,0.7,1.0), 
                        alpha = c(0,0.1,0.5,0.7,1.0),
                        gamma = c(0,5,10,15,20,30),
                        max_delta_step=c(0,5,10,15,20,30))

print(paste('number of combinations of parameters to be tested for best parameter selection are: ',nrow(xgb_grid)))

watchlist <- list(train=train_matrix)
cv.results <- data.frame(xgb_grid)
cv.results$nrounds = 0
cv.results$auc = 0

for (ind in 1:dim(xgb_grid)[1]){
  # Model parameters
  max_depth <- xgb_grid[ind,1]
  min_child_weight <-xgb_grid[ind,2]
  subsample<-xgb_grid[ind,3]
  eta<-xgb_grid[ind,4]
  colsample_bytree<-xgb_grid[ind,5]
  lambda<-xgb_grid[ind,6]
  alpha<-xgb_grid[ind,7]
  gamma<-xgb_grid[ind,8]
  max_delta_step<-xgb_grid[ind,9]
  
  #
  param <- list(booster="gbtree",
                objective = "binary:logistic",
                eval_metric="error",
                eta = eta,
                max_depth = max_depth,
                min_child_weight = min_child_weight,
                subsample = subsample, 
                colsample_bytree = colsample_bytree,
                lambda = lambda,
                alpha = alpha,
                gamma=gamma,
                max_delta_step=max_delta_step)
  # 5-fold CV
  set.seed(11111)
  fit_cv <- xgb.cv(params=param,
                   data=train_matrix,
                   nrounds=300,
                   watchlist=watchlist,
                   nfold=5,
                   early_stopping_rounds = 3)
  
  cv.results[ind,10] <- fit_cv$best_iteration
  cv.results[ind,11] <- fit_cv$evaluation_log[fit_cv$best_iteration][[4]]
  cat("Trained ", ind, " of ", dim(xgb_grid)[1], "\n")
}

write.csv(cv.results,'XGB_Hyperparameters_Opt.csv')




##RUN FULL MODEL BASED ON BEST PARAMETERS
#ind.max<-which.min(cv.results$auc)


#a<-cv.results[ind.max,]

## xgb style matrices
#xgb_params <- list(booster="gbtree",
#                   objective = "binary:logistic",
#                   eval_metric = "error",
#                   eta=a$eta,
#                   subsample=a$subsample,
#                   max_depth=a$max_depth,
#                   alpha=a$alpha,
#                   lambda=a$lambda,
#                   gamma=a$gamma,
#                   min_child_weight=a$min_child_weight,
#                   max_delta_step = a$max_delta_step,
#                   colsample_bytree=a$colsample_bytree)

#bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = a$nrounds)


#X_dtrain_test<-as.matrix(dtrain_test[,1:(l-1)])
#X_dtrain_test_matrix<-xgb.DMatrix(data=X_dtrain_test)
#dtrain_test_predict<-predict(bst,X_dtrain_test_matrix)
#test_pred=ifelse(dtrain_test_predict>0.5,'Yes','No')
#cm_xgb=table(dtrain_test$RETURNYes,test_pred,dnn = c('Actual','Predicted'))
#acc_xgb=(cm_xgb[1,1]+cm_xgb[2,2])/sum(cm_xgb)
#print(paste('The accuracy of the model at 0.5 cutoff is:',acc_xgb))

