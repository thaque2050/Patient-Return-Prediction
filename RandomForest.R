library(ranger)

#DATA PREPROCESSING
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


#RANDOM FOREST
set.seed(1234)

dtrain2=model.matrix(~.-1,data = dtrain)%>%data.frame()
index=sample.split(dtrain2$RETURNYes,SplitRatio = 0.75)
dtrain_train=dtrain2[index,]
dtrain_test=dtrain2[!index,]


#Conver variable to factor if you want to run classification tree
dtrain_train$RETURNYes=as.factor(dtrain_train$RETURNYes)

#Build grid for cartesian search
hyper_grid <- expand.grid(
  n_tree=c(500,1000,5000),
  mtry       = seq(5, 106, by = 10),
  min_node_size  = c(5,10,15,20,30,35),
  sampe_size = c(0.2,0.3,0.45,0.55,0.625,0.65,0.80,0.90,1.0),
  OOB_Misclassification=0
)


#Loop through to get best hyperparameters
for(i in 1:nrow(hyper_grid)) {
  model <- ranger(
    formula         = RETURNYes ~ ., 
    data            = dtrain_train, 
    num.trees       = hyper_grid$n_tree[i],
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min_node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123,
    verbose = FALSE,
    probability = TRUE
  )
  
  print(i)
  hyper_grid$OOB_Misclassification[i] <- model$prediction.error
}

best_params=hyper_grid[hyper_grid$OOB_Misclassification==min(hyper_grid$OOB_Misclassification),]

write.csv(hyper_grid,'rf_parameter_optimization.csv')

#Run model on full training dataset with best hyperparameters
model.rf=ranger(formula = RETURNYes~.,data = dtrain_train,num.trees = best_params$n_tree,
                mtry = best_params$mtry,min.node.size = best_params$min_node_size,
                sample.fraction = best_params$sampe_size,seed = 123,verbose = FALSE,probability = TRUE)

#prediction
pred=predict(model.rf,dtrain_test,type='response',num.trees = model.rf$num.trees)
pred=ifelse(pred$predictions[,2]>0.5,1,0)
cm_rf=table(dtrain_test$RETURNYes,pred)

acc_rf=(cm_rf[1,1]+cm_rf[2,2])/sum(cm_rf)
print(paste('The accuracy of the model at 0.5 cutoff is:',acc_rf))
