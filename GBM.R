#GBM with H20
library(h2o)
library(dplyr)
library(caTools)

#DATA PROCESSING


df<-read.csv('Hospitals_Train.csv',nrows = 38221)
str(df)

dtrain=df

dtrain$INDEX=NULL
dtrain$CONSULT_IN_ED[is.na(dtrain$CONSULT_IN_ED)]=0
dtrain$CONSULT_IN_ED=as.factor(dtrain$CONSULT_IN_ED)

dtrain$RISK=as.character(dtrain$RISK)
dtrain$RISK[dtrain$RISK=='']='Not Measured'
dtrain$RISK=as.factor(dtrain$RISK)


dtrain$SEVERITY=as.character(dtrain$SEVERITY)
dtrain$SEVERITY[dtrain$SEVERITY=='']='Not Measured'
dtrain$SEVERITY=as.factor(dtrain$SEVERITY)


dtrain$ADMIT_RESULT=as.character(dtrain$ADMIT_RESULT)
dtrain$ADMIT_RESULT[dtrain$ADMIT_RESULT=='']='Others'
dtrain$ADMIT_RESULT=as.factor(dtrain$ADMIT_RESULT)

dtrain$ACUITY_ARR=as.character(dtrain$ACUITY_ARR)
dtrain=dtrain[!(dtrain$ACUITY_ARR=='5 Purple'),]
dtrain$ACUITY_ARR=as.factor(dtrain$ACUITY_ARR)


dtrain$SAME_DAY=as.factor(dtrain$SAME_DAY)
dtrain$CONSULT_ORDER=as.factor(dtrain$CONSULT_ORDER)
dtrain$CONSULT_CHARGE=as.factor(dtrain$CONSULT_CHARGE)


#dtrain$WEEKDAY_ARR=ifelse((dtrain$WEEKDAY_ARR==1 | dtrain$WEEKDAY_ARR==7),'Weekend','Weekday')
dtrain$WEEKDAY_ARR=as.factor(dtrain$WEEKDAY_ARR)
dtrain$WEEKDAY_DEP=NULL


#dtrain$MONTH_ARR=ifelse((dtrain$MONTH_ARR==12|dtrain$MONTH_ARR==1 |dtrain$MONTH_ARR==2),'Winter',ifelse((dtrain$MONTH_ARR==3 |dtrain$MONTH_ARR==4 |dtrain$MONTH_ARR==5),'Spring',ifelse((dtrain$MONTH_ARR==6 |dtrain$MONTH_ARR==7 |dtrain$MONTH_ARR==8),'Summer','Autumn')))
dtrain$MONTH_ARR=as.factor(dtrain$MONTH_ARR)


#dtrain$MONTH_DEP=ifelse((dtrain$MONTH_DEP==12|dtrain$MONTH_DEP==1 |dtrain$MONTH_DEP==2),'Winter',ifelse((dtrain$MONTH_DEP==3 |dtrain$MONTH_DEP==4 |dtrain$MONTH_DEP==5),'Spring',ifelse((dtrain$MONTH_DEP==6 |dtrain$MONTH_DEP==7 |dtrain$MONTH_DEP==8),'Summer','Autumn')))
dtrain$MONTH_DEP=as.factor(dtrain$MONTH_DEP)


#dtrain$HOUR_ARR=ifelse((dtrain$HOUR_ARR<18 | dtrain$HOUR_ARR<8),'Daytime','Evening')
#dtrain$HOUR_DEP=ifelse((dtrain$HOUR_DEP<18 | dtrain$HOUR_DEP<8),'Daytime','Evening')
dtrain$HOUR_DEP=NULL

dtrain$CHARGES=as.numeric(as.character(dtrain$CHARGES))
#dtrain$CHARGES[is.na(dtrain$CHARGES) & dtrain$ED_RESULT=='Arrived in Error']=0
#dtrain$CHARGES[is.na(dtrain$CHARGES)]=median(dtrain$CHARGES,na.rm = TRUE)

dtrain$ED_RESULT=as.character(dtrain$ED_RESULT)
#dtrain$ED_RESULT[dtrain$ED_RESULT=='']='Discharge'
dtrain$ED_RESULT=as.factor(dtrain$ED_RESULT)


dtrain$DC_RESULT=as.character(dtrain$DC_RESULT)
dtrain$DC_RESULT=ifelse(dtrain$DC_RESULT=='Home or Self Care','Home or Self Care',
                        ifelse(dtrain$DC_RESULT=='LEFT W/O BEING SEEN AFTER TRIAGE','LEFT W/O BEING SEEN AFTER TRIAGE',
                               ifelse(dtrain$DC_RESULT=='LEFT W/O BEING SEEN BEFORE TRIAGE','LEFT W/O BEING SEEN BEFORE TRIAGE',
                                      ifelse(dtrain$DC_RESULT=='LEFT W/O COMPLETED TREATMENT','LEFT W/O COMPLETED TREATMENT',
                                             ifelse(dtrain$DC_RESULT=='Rehab Facility','Rehab Facility',
                                                    ifelse(dtrain$DC_RESULT=='Home Health Care Svc','Home Health Care Svc',
                                                           ifelse(dtrain$DC_RESULT=='Skilled Nursing Facility','Skilled Nursing Facility',
                                                                  ifelse(dtrain$DC_RESULT=='Expired','Expired',
                                                                         ifelse(dtrain$DC_RESULT=='Left Against Medical Advice','Left Against Medical Advice','Others')))))))))
dtrain$DC_RESULT=as.factor(dtrain$DC_RESULT)



dtrain[dtrain=='']=NA
dtrain[dtrain=='#N/A']=NA
dtrain[dtrain=='#VALUE!']=NA
dtrain=na.omit(dtrain)
dtrain=droplevels(dtrain)


#GBM
set.seed(1234)

dtrain2=model.matrix(~.-1,data = dtrain)%>%data.frame()
index=sample.split(dtrain2$RETURNYes,SplitRatio = 0.75)
dtrain_train=dtrain2[index,]
dtrain_test=dtrain2[!index,]

set.seed(4567)
index2=sample.split(dtrain_train$RETURNYes,SplitRatio = 0.75)
dtrain_valid=dtrain_train[!index2,]
dtrain_train2=dtrain_train[index2,]


#Launch H2o cluster on localhost
h2o.init()


#Define input
Y='RETURNYes' #response
X=colnames(dtrain_train%>%dplyr::select(-'RETURNYes')) #predictors

## the response variable is an integer, we will turn it into a categorical/factor for binary classification
dtrain_train2[[Y]] <- as.factor(dtrain_train2[[Y]])
dtrain_train[[Y]] <- as.factor(dtrain_train[[Y]])
dtrain_valid[[Y]] <- as.factor(dtrain_valid[[Y]])
dtrain_test[[Y]] <- as.factor(dtrain_test[[Y]])

train2.h2o=as.h2o(dtrain_train2)
validation.h2o=as.h2o(dtrain_valid)
train.h2o=as.h2o(dtrain_train)
test.h2o=as.h2o(dtrain_test)



# create hyperparameter grid
hyper_grid <- list(
  max_depth = c(5,6,7,8,10),
  min_rows = 1,
  learn_rate = c(0.01, 0.05, 0.06,0.1),
  learn_rate_annealing = c(0.97,0.98,.99,1.0),
  sample_rate = c(.5, .625,.75, 1),
  col_sample_rate = c(0.9,0.95,1.0),
  ntrees=c(1000,5000)
#  col_sample_rate_per_tree=seq(0.1,1.0,0.1),
#  max_after_balance_size=seq(0.1,20,0.5)
)



# perform grid search 
grid <- h2o.grid(
  hyper_params = hyper_grid,
  algorithm = "gbm",
  grid_id = "gbm_grid1",
  x = X, 
  y = Y, 
  training_frame = train2.h2o,
  validation_frame = validation.h2o,
  search_criteria = list(strategy = "Cartesian"),
  seed = 123
)

a=grid@summary_table
b=a[a$logloss==min(a$logloss),]

write.csv(a,'GBM_H2o_Parameters.csv')


## We only provide the required parameters, everything else is default
model.gbm <- h2o.gbm(x = X,y = Y,training_frame = train.h2o,
                     max_depth = as.numeric(b$max_depth),
                     min_rows = as.numeric(b$min_rows),
                     learn_rate = as.numeric(b$learn_rate),
                     learn_rate_annealing = as.numeric(b$learn_rate_annealing),
                     sample_rate = as.numeric(b$sample_rate),
                     col_sample_rate = as.numeric(b$col_sample_rate),
                     ntrees=as.numeric(b$ntrees),
#                     col_sample_rate_per_tree=b$col_sample_rate_per_tree,
#                     max_after_balance_size=b$max_after_balance_size,
                     seed = 123)



## Get the accuracy on the test set
pred_h2o <- predict(model.gbm, test.h2o)
pred_final=as.data.frame(pred_h2o)
test_pred=ifelse(pred_final$p1>0.5,'Yes','No')
cm_gbm=table(dtrain_test$RETURNYes,test_pred,dnn = c('Actual','Predicted'))
acc_gbm=(cm_gbm[1,1]+cm_gbm[2,2])/sum(cm_gbm)
print(paste('The accuracy of the model at 0.5 cutoff is:',acc_gbm))

