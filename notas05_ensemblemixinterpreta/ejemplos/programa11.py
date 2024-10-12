import lime
import lime.lime_tabular

import pandas as pd
import numpy as np
import lightgbm as lgb

# For converting textual categories to integer labels 
from sklearn.preprocessing import LabelEncoder

# for creating train test split
from sklearn.model_selection import train_test_split

# specify your configurations as a dict
lgb_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'binary',
    'metric':'binary_logloss',
    'metric': {'l2', 'auc'},
    'num_leaves': 20,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': None,
    'num_iteration':100,
    'num_threads':7,
    'max_depth':6,
    'min_data_in_leaf':20,
    'alpha':0.2}


# reading the titanic data
df_bank = pd.read_csv(r'data/GermanData.csv')

# data preparation
df_bank.fillna(0,inplace=True)

le = LabelEncoder()

df_bank.columns

feat = ['quaCheckaccount_le', 
        'numDuration', 
        'quaJob_le',
        'numAge', 
        'quaStatSex_le',
        'numCredits']


# label encoding textual data
df_bank['quaCheckaccount_le'] = le.fit_transform(df_bank['quaCheckaccount'])
df_bank['quaJob_le'] = le.fit_transform(df_bank['quaJob'])
df_bank['quaStatSex_le'] = le.fit_transform(df_bank['quaStatSex'])

# using train test split to create validation set
X_train,X_test,y_train,y_test = train_test_split(df_bank[feat],
                                                 df_bank[['target']],
                                                 test_size=0.3,
                                                 random_state=155)


# def lgb_model(X_train,y_train,X_test,y_test,lgb_params):
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

# training the lightgbm model
model = lgb.train(lgb_params,
                  lgb_train,
                  num_boost_round=20,
                  valid_sets=lgb_eval)


explainer = lime.lime_tabular.LimeTabularExplainer(df_bank[model.feature_name()].astype(int).values,  
                                                   mode='classification',
                                                   training_labels=df_bank['target'],
                                                   feature_names=model.feature_name())


def prob(data):
     return np.array(list(zip(1-model.predict(data),model.predict(data))))


#Se supone un punto de corte de 0.31 ya que:
np.sum(y_train)/len(y_train)



# asking for explanation for LIME model
i = 0
exp = explainer.explain_instance(X_test.iloc[i].astype(int).values, 
                                 prob, 
                                 num_features=6)

X_test.iloc[i]

exp.save_to_file('explanation0.html')

y_test.iloc[i]


i = 2
exp = explainer.explain_instance(X_test.iloc[i].astype(int).values, 
                                 prob, 
                                 num_features=6)

X_test.iloc[i]

exp.save_to_file('explanation2.html')

y_test.iloc[2]


#Se obtienen las predicciones reales
prob(X_test)
