# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/lightgbm-vs-xgboost-which-algorithm-win-the-race-1ff7dd4917d

"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#import lightgbm and xgboost 
import lightgbm as lgbm
import xgboost as xgb 

df_train = pd.read_csv('data/adult_train.csv', header = None)
df_test = pd.read_csv('data/adult_test.csv', header = None)

df_train.columns = ['age', 'work_class', 'fnl_wgt', 'education', 'education-num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week',
           'native_country', 'Income']

df_test.columns = ['age', 'work_class', 'fnl_wgt', 'education', 'education-num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week',
           'native_country', 'Income']



# LabelEncoding our predictor variable 
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder() 
l.fit(df_train.Income) 
df_train.Income=Series(l.transform(df_train.Income))  
df_train.Income.value_counts()


#One_Hot_Encoding for the Categorical features in the dataset
one_hot_work_class=pd.get_dummies(df_train.work_class) 
one_hot_education=pd.get_dummies(df_train.education) 
one_hot_marital_status=pd.get_dummies(df_train.marital_status) 
one_hot_occupation=pd.get_dummies(df_train.occupation)
one_hot_relationship=pd.get_dummies(df_train.relationship) 
one_hot_race=pd.get_dummies(df_train.race) 
one_hot_sex=pd.get_dummies(df_train.sex) 
one_hot_native_country=pd.get_dummies(df_train.native_country)


#removing categorical features 
df_train.drop(['work_class','education','marital_status','occupation','relationship',
         'race','sex','native_country'],axis=1,inplace=True)

#Merging one hot encoded features with our dataset 'data' 
df_train=pd.concat([df_train, 
                    one_hot_work_class,
                    one_hot_education,one_hot_marital_status,
                    one_hot_occupation,one_hot_relationship,
                    one_hot_race,one_hot_sex,one_hot_native_country],axis=1)

_, i = np.unique(df_train.columns, return_index=True) #removing duplicates
df_train = df_train.iloc[:, i]

#Here our target variable is 'Income' with values as 1 or 0.  
#Separating our data into features dataset x and our target dataset y 
x=df_train.drop('Income',axis=1) 
y=df_train.Income

#Imputing missing values in our target variable 
y.fillna(y.mode()[0],inplace=True)


#splitting our dataset into test and train 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

#DMatrix object 
#label is used to define our outcome variable
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)

#setting parameters for xgboost #hyperparameter-tuning
parameters={'max_depth':7, 
            'silent':1,
            'objective':'binary:logistic',
            'eval_metric':'auc',
            'learning_rate':.05}

#training our model 
num_round=50
from datetime import datetime 
start = datetime.now() 
xg=xgb.train(parameters,dtrain,num_round) 
stop = datetime.now()

#Execution time of the model 
execution_time_xgb = stop-start 
execution_time_xgb

#datetime.timedelta( , , ) representation => (days , seconds , microseconds) 
#now predicting our model on test set 
ypred=xg.predict(dtest) 

#Converting probabilities into 1 or 0  
for i in range(0,9769): 
    if ypred[i]>=.5:       # setting threshold to .5 
       ypred[i]=1 
    else: 
       ypred[i]=0

from sklearn.metrics import accuracy_score 
accuracy_xgb = accuracy_score(y_test,ypred) 
accuracy_xgb


#Modelo lightgbm
train_data=lgbm.Dataset(x_train,label=y_train)


#setting parameters for lightgbm
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']

#training our model using light gbm
num_round=50
start=datetime.now()
lgbm=lgbm.train(param,train_data,num_round)
stop=datetime.now()

#Execution time of the model
execution_time_lgbm = stop-start
execution_time_lgbm

#predicting on test set
ypred2=lgbm.predict(x_test)

for i in range(0,9769):
    if ypred2[i]>=.5:       # setting threshold to .5
       ypred2[i]=1
    else:  
       ypred2[i]=0

#calculating accuracy
accuracy_lgbm = accuracy_score(ypred2,y_test)
accuracy_lgbm

from sklearn.metrics import roc_auc_score

#calculating roc_auc_score for xgboost
auc_xgb =  roc_auc_score(y_test,ypred)
auc_xgb

#calculating roc_auc_score for light gbm. 
auc_lgbm = roc_auc_score(y_test,ypred2)
auc_lgbm_comparison_dict = {'accuracy score':(accuracy_lgbm,accuracy_xgb),'auc score':(auc_lgbm,auc_xgb),'execution time':(execution_time_lgbm,execution_time_xgb)}


#Creating a dataframe ‘comparison_df’ for comparing the performance of Lightgbm and xgb. 
comparison_df = DataFrame(auc_lgbm_comparison_dict) 
comparison_df.index= ['LightGBM','xgboost'] 
comparison_df

