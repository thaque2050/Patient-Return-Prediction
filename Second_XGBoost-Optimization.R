library(dplyr)
library(ggplot2)
library(reshape2)
library(caTools)
library(xgboost)
library(mice)

dtrain1=read.csv('Hospitals_Train.csv',nrows = 38221)
dtrain1$RETURN=as.character(dtrain1$RETURN)
dtrain1[dtrain1=='#N/A']=NA
remove_row=rownames(dtrain1[is.na(dtrain1$RETURN),])
remove_row=as.numeric(remove_row)

dtrain=read.csv('level1_all.csv')
dtrain$index=NULL
dtrain$RETURN=as.character(dtrain$RETURN)
dtrain=dtrain[-remove_row,]

dtrain$RETURN=ifelse(dtrain$RETURN=='Yes',1,0)
df_train=dtrain[1:38080,]
df_test=dtrain[38081:50112,]

#1XGB1
df_train2=df_train[1:19110,]
df_train3=df_train[19111:38080,]


#XGB
l=ncol(df_train3)
X_train<-as.matrix(df_train3[,1:(l-1)])
Y_train<-df_train3[,l]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)

# Grid for model training
xgb_grid <- expand.grid(max_depth=c(5,10),
                        min_child_weight = c(0,5,10),
                        subsample=c(0.3,0.5,0.625),
                        eta = c(0.01,0.03,0.3), 
                        colsample_bynode=c(0.1,0.5,1.0),
                        lambda = c(0,0.1,0.7), 
                        alpha = c(0,0.1,0.7),
                        gamma = c(0,5,100),
                        max_delta_step=c(5,30))

print(paste('number of combinations of parameters to be tested for best parameter selection are: ',nrow(xgb_grid)))

watchlist <- list(train=train_matrix)
cv.results <- data.frame(xgb_grid)
cv.results$nrounds = 0
cv.results$error = 0

for (ind in 1:dim(xgb_grid)[1]){
  # Model parameters
  max_depth <- xgb_grid[ind,1]
  min_child_weight <-xgb_grid[ind,2]
  subsample<-xgb_grid[ind,3]
  eta<-xgb_grid[ind,4]
  colsample_bynode<-xgb_grid[ind,5]
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
                colsample_bynode = colsample_bynode,
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

write.csv(cv.results,'XGB_Hyperparameters_Opt_second_level.csv')










#FINAL PREDICTIONS
#XGB
l=ncol(df_train3)
X_train<-as.matrix(df_train3[,1:(l-1)])
Y_train<-df_train3[,l]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)


# xgb hyperparameters
xgb_params <- list(booster="gbtree",objective = "binary:logistic",eval_metric = "error",
                   eta=0.01,
                   subsample=0.5,
                   max_depth=5,
                   alpha=0.1,
                   lambda=0.7,
                   gamma=0,
                   min_child_weight=0,
                   max_delta_step = 5,
                   colsample_bynode=0.1)

bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = 20)



#prediction 
X_dtrain_test<-as.matrix(df_test[,1:(l-1)])
X_dtrain_test_matrix<-xgb.DMatrix(data=X_dtrain_test)
dtrain_test_predict<-predict(bst,X_dtrain_test_matrix)
test_pred=ifelse(dtrain_test_predict>0.5,'Yes','No')

index=seq(1,12032,1)
df_predicted=data.frame(cbind(index,test_pred))
colnames(df_predicted)=c('INDEX','RETURN')           

write.csv(df_predicted,'submission_TEAM_10.csv.',row.names = FALSE)


