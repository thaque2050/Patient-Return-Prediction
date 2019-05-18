library(ranger)
library(h2o)
library(xgboost)
library(dplyr)
library(mice)
library(caTools)

#DATA PROCESSING
dtrain=read.csv('Hospitals_Train.csv',nrows = 38221)
dtest=read.csv('Hospitals_Test_X.csv')
dtest$RETURN=rep(NA,nrow(dtest))

#Combine two datasets
df=rbind(dtrain,dtest)%>%data.frame()

#Check missing values
sapply(df, function(x)sum(is.na(x)))

#Replace systemic missing values and delete unnecessary columns
df$INDEX=NULL
df$CONSULT_IN_ED[is.na(df$CONSULT_IN_ED)]=0
df$CONSULT_IN_ED=as.factor(df$CONSULT_IN_ED)

df$RISK=as.character(df$RISK)
df$RISK[df$RISK=='']='Not Measured'
df$RISK=as.factor(df$RISK)


df$SEVERITY=as.character(df$SEVERITY)
df$SEVERITY[df$SEVERITY=='']='Not Measured'
df$SEVERITY=as.factor(df$SEVERITY)


df$ADMIT_RESULT=as.character(df$ADMIT_RESULT)
df$ADMIT_RESULT[df$ADMIT_RESULT=='']='Others'
df$ADMIT_RESULT=as.factor(df$ADMIT_RESULT)

#CHANGE VARIABLE TYPES
df$SAME_DAY=as.factor(df$SAME_DAY)
df$CONSULT_ORDER=as.factor(df$CONSULT_ORDER)
df$CONSULT_CHARGE=as.factor(df$CONSULT_CHARGE)


#df$WEEKDAY_ARR=ifelse((df$WEEKDAY_ARR==1 | df$WEEKDAY_ARR==7),'Weekend','Weekday')
df$WEEKDAY_ARR=as.factor(df$WEEKDAY_ARR)
df$WEEKDAY_DEP=NULL


#df$MONTH_ARR=ifelse((df$MONTH_ARR==12|df$MONTH_ARR==1 |df$MONTH_ARR==2),'Winter',ifelse((df$MONTH_ARR==3 |df$MONTH_ARR==4 |df$MONTH_ARR==5),'Spring',ifelse((df$MONTH_ARR==6 |df$MONTH_ARR==7 |df$MONTH_ARR==8),'Summer','Autumn')))
df$MONTH_ARR=as.factor(df$MONTH_ARR)


#df$MONTH_DEP=ifelse((df$MONTH_DEP==12|df$MONTH_DEP==1 |df$MONTH_DEP==2),'Winter',ifelse((df$MONTH_DEP==3 |df$MONTH_DEP==4 |df$MONTH_DEP==5),'Spring',ifelse((df$MONTH_DEP==6 |df$MONTH_DEP==7 |df$MONTH_DEP==8),'Summer','Autumn')))
df$MONTH_DEP=as.factor(df$MONTH_DEP)


#df$HOUR_ARR=ifelse((df$HOUR_ARR<18 | df$HOUR_ARR<8),'Daytime','Evening')
#df$HOUR_DEP=ifelse((df$HOUR_DEP<18 | df$HOUR_DEP<8),'Daytime','Evening')
df$HOUR_DEP=NULL

df$CHARGES=as.numeric(as.character(df$CHARGES))
#df$CHARGES[is.na(df$CHARGES) & df$ED_RESULT=='Arrived in Error']=0
#df$CHARGES[is.na(df$CHARGES)]=median(df$CHARGES,na.rm = TRUE)

df$ED_RESULT=as.character(df$ED_RESULT)
#df$ED_RESULT[df$ED_RESULT=='']='Discharge'
df$ED_RESULT=as.factor(df$ED_RESULT)


df$DC_RESULT=as.character(df$DC_RESULT)
df$DC_RESULT=ifelse(df$DC_RESULT=='Home or Self Care','Home or Self Care',
                    ifelse(df$DC_RESULT=='LEFT W/O BEING SEEN AFTER TRIAGE','LEFT W/O BEING SEEN AFTER TRIAGE',
                           ifelse(df$DC_RESULT=='LEFT W/O BEING SEEN BEFORE TRIAGE','LEFT W/O BEING SEEN BEFORE TRIAGE',
                                  ifelse(df$DC_RESULT=='LEFT W/O COMPLETED TREATMENT','LEFT W/O COMPLETED TREATMENT',
                                         ifelse(df$DC_RESULT=='Rehab Facility','Rehab Facility',
                                                ifelse(df$DC_RESULT=='Home Health Care Svc','Home Health Care Svc',
                                                       ifelse(df$DC_RESULT=='Skilled Nursing Facility','Skilled Nursing Facility',
                                                              ifelse(df$DC_RESULT=='Expired','Expired',
                                                                     ifelse(df$DC_RESULT=='Left Against Medical Advice','Left Against Medical Advice',
                                                                            ifelse(df$DC_RESULT=='',NA,'Others'))))))))))
df$DC_RESULT=as.factor(df$DC_RESULT)





#Convert different type of missing values as NAs
df[df=='']=NA
df[df=='#N/A']=NA
df[df=='#VALUE!']=NA
df=droplevels(df)

#Missing values by columns
sapply(df, function(x)sum(is.na(x)))




#Impute missing values using mice with 11 possible imputations
n=5
df_imp=mice(df,m=n,maxit=5,meth='pmm',seed=500)




index=seq(1,50253,1)
pred_final_X=data.frame(index)



for (i in 1:n) {
  
df_complete <- complete(df_imp,i)
df_complete=droplevels(df_complete)

#Convert data with dummy variables
df_dummy=model.matrix(~.-1,data = df_complete)%>%data.frame()

df_train=df_dummy[1:38221,]
df_test=df_dummy[38222:50253,]

#1XGB1
df_train2=df_train[1:19110,]
df_train3=df_train[19111:38221,]


l=ncol(df_train2)
X_train<-as.matrix(df_train2[,1:(l-1)])
Y_train<-df_train2[,l]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)
xgb_params <- list(booster="gbtree",
                   objective = "binary:logistic",
                   eval_metric = "error",
                   eta=0.15,
                   subsample=1,
                   max_depth=500,
                   alpha=0.95,
                   lambda=0.08,
                   gamma=3,
                   min_child_weight=10,
                   max_delta_step = 5,
                   colsample_bytree=0.8)

bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = 31)

X_dtrain_test<-as.matrix(df_test[,1:(l-1)])
X_dtrain_test_matrix<-xgb.DMatrix(data=X_dtrain_test)
pred_test=predict(bst,X_dtrain_test_matrix)

X_dtrain_train<-as.matrix(df_train3[,1:(l-1)])
X_dtrain_train_matrix<-xgb.DMatrix(data=X_dtrain_train)
pred_train2=predict(bst,X_dtrain_train_matrix)
pred_train1=predict(bst,X_train)
pred_f=c(pred_train1,pred_train2,pred_test)

pred_final_X=data.frame(cbind(pred_final_X,pred_f))


#2XGB2

l=ncol(df_train2)
X_train<-as.matrix(df_train2[,1:(l-1)])
Y_train<-df_train2[,l]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)
xgb_params <- list(booster="gbtree",
                   objective = "binary:logistic",
                   eval_metric = "error",
                   eta=0.2,
                   subsample=1,
                   max_depth=500,
                   alpha=0.9,
                   lambda=0.05,
                   gamma=2,
                   min_child_weight=15,
                   max_delta_step = 10,
                   colsample_bytree=0.7)

bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = 25)

X_dtrain_test<-as.matrix(df_test[,1:(l-1)])
X_dtrain_test_matrix<-xgb.DMatrix(data=X_dtrain_test)
pred_test=predict(bst,X_dtrain_test_matrix)

X_dtrain_train<-as.matrix(df_train3[,1:(l-1)])
X_dtrain_train_matrix<-xgb.DMatrix(data=X_dtrain_train)
pred_train2=predict(bst,X_dtrain_train_matrix)
pred_train1=predict(bst,X_train)

pred_f=c(pred_train1,pred_train2,pred_test)
pred_final_X=data.frame(cbind(pred_final_X,pred_f))



#3RF

df_train2$RETURNYes=as.factor(df_train2$RETURNYes)

model.rf=ranger(formula = RETURNYes~.,data = df_train2,num.trees = 5000,
                mtry = 35,min.node.size = 35,sample.fraction = 1,seed = 123,verbose = FALSE,probability = TRUE)


pred_test=predict(model.rf,df_test,type='response',num.trees = model.rf$num.trees)$prediction[,1]

pred_train1=predict(model.rf,df_train2,type='response',num.trees = model.rf$num.trees)$prediction[,1]
pred_train2=predict(model.rf,df_train3,type='response',num.trees = model.rf$num.trees)$prediction[,1]

pred_f=c(pred_train1,pred_train2,pred_test)
pred_final_X=data.frame(cbind(pred_final_X,pred_f))


#4GBM
h2o.init()

Y='RETURNYes' #response
X=colnames(df_train2%>%dplyr::select(-'RETURNYes')) #predictors

df_train2[[Y]] <- as.factor(df_train2[[Y]])
df_train3[[Y]] <- as.factor(df_train3[[Y]])
df_test[[Y]] <- as.factor(df_test[[Y]])

df_train2.h2o=as.h2o(df_train2)
df_train3.h2o=as.h2o(df_train3)
df_test.h2o=as.h2o(df_test)

model.gbm <- h2o.gbm(x = X,y = Y,training_frame = df_train2.h2o,
                     max_depth = 7,
                     min_rows = 1,
                     learn_rate = 0.1,
                     learn_rate_annealing = 0.99,
                     sample_rate = 0.5,
                     col_sample_rate = 1,
                     ntrees=1000,
                     #col_sample_rate_per_tree=b$col_sample_rate_per_tree,
                     #max_after_balance_size=b$max_after_balance_size,
                     seed = 123)

pred_h2o <- predict(model.gbm, df_test.h2o)
pred_final=as.data.frame(pred_h2o)
pred_test=pred_final$p1

pred_h2o <- predict(model.gbm,df_train2.h2o)
pred_final=as.data.frame(pred_h2o)
pred_train1=pred_final$p1

pred_h2o <- predict(model.gbm,df_train3.h2o)
pred_final=as.data.frame(pred_h2o)
pred_train2=pred_final$p1

pred_f=c(pred_train1,pred_train2,pred_test)
pred_final_X=data.frame(cbind(pred_final_X,pred_f))





#5XGB3
df_train2$RETURNYes=as.numeric(as.character(df_train2$RETURNYes))

l=ncol(df_train2)
X_train<-as.matrix(df_train2[,1:(l-1)])
Y_train<-df_train2[,l]
train_matrix<-xgb.DMatrix(data=X_train, label=Y_train)
xgb_params <- list(booster="gbtree",
                   objective = "binary:logistic",
                   eval_metric = "error",
                   eta=0.1,
                   subsample=1,
                   max_depth=500,
                   alpha=0.7,
                   lambda=0.1,
                   gamma=1,
                   min_child_weight=15,
                   max_delta_step = 1,
                   colsample_bytree=0.7)

bst<-xgboost(params = xgb_params,data=X_train,label =Y_train,nrounds = 62)

X_dtrain_test<-as.matrix(df_test[,1:(l-1)])
X_dtrain_test_matrix<-xgb.DMatrix(data=X_dtrain_test)
pred_test=predict(bst,X_dtrain_test_matrix)

X_dtrain_train<-as.matrix(df_train3[,1:(l-1)])
X_dtrain_train_matrix<-xgb.DMatrix(data=X_dtrain_train)
pred_train2=predict(bst,X_dtrain_train_matrix)
pred_train1=predict(bst,X_train)

pred_f=c(pred_train1,pred_train2,pred_test)
pred_final_X=data.frame(cbind(pred_final_X,pred_f))

}


#Fix from here-------------------------------------
RETURN=complete(df_imp,1)$RETURN
pred_final_X$RETURN=RETURN

#Build New dataframe
write.csv(pred_final_X,'level1_all.csv',row.names = FALSE)



