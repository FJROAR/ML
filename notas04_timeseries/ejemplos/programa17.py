###############################################################################
########################### import modules ####################################
###############################################################################
import pandas as pd
import numpy as np
#from matplotlib import pyplot
#import statistics
#import math
#import sklearn.metrics
#import matplotlib.pyplot as plt
import plotly.offline as py
#import plotly.graph_objs as go
py.init_notebook_mode()
#import ipywidgets
#import scipy.stats
#from scipy.stats import boxcox
#specific fbprophet
import distutils
print(distutils.__file__)
#import pystan
#import plotly
#import fbprophet 
from prophet import Prophet

#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
#from fbprophet.plot import add_changepoints_to_plot
#from fbprophet.diagnostics import cross_validation
#from fbprophet.diagnostics import performance_metrics
#from fbprophet.plot import plot_cross_validation_metric
import itertools

###############################################################################
############################### import data ################################
###############################################################################
df = pd.read_csv('data/prueba_11.csv', sep = ',')
df_test = df[(len(df) - 10): len(df)]
df_train = df[0: (len(df)-10)]

date = pd.date_range('10-01-19', '02-29-20', freq='d')
df_train['ds'] = date
df_train=df_train[['ds','n_datos']]
df_train.columns=['ds','y']


###############################################################################
############################## grid search ####################################
###############################################################################


param_grid = {  
    'seasonality_mode': ('multiplicative', 'additive'),
    'n_changepoints': [15, 25, 35],
    'daily_seasonality': [True, False],
    'weekly_seasonality': [True, False]
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(df_train)  # Fit model with given params
    
    future = m.make_future_dataframe(freq='d', periods= 10) #includes by default dates from the history
    forecast = m.predict(future)
    pred = forecast[['yhat']].tail(10) 
    rmses.append(np.mean((pred['yhat'] - df_test['n_datos'])**2)**0.5)


# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

# Python
best_params = all_params[np.argmin(rmses)]
print(best_params)


#Predicción
model= Prophet(n_changepoints = best_params['n_changepoints'],
               seasonality_mode = best_params['seasonality_mode'],
               daily_seasonality = best_params['daily_seasonality'], 
               weekly_seasonality = best_params['weekly_seasonality'])
model.fit(df_train)
future = model.make_future_dataframe(freq='d', periods= 10) 
forecast = model.predict(future)

#Comparación

reales = df_test['n_datos']
pred = forecast['yhat'][(len(df) - 10): len(df)]

eam1 = np.mean(abs(np.array(reales) - np.array(pred)))
eam1
