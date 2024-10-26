#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:37:17 2020

@author: anacorraljodar
"""

###############################################################################
########################### import modules ####################################
###############################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot
#import statistics
#import math
#import sklearn.metrics
import matplotlib.pyplot as plt
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


###############################################################################
############################### import data ################################
###############################################################################
data_set = pd.read_csv("data/prueba_11.csv")
date=pd.date_range('10-01-19', '03-10-20', freq='d')
data_set['ds'] = date
data_set = data_set.drop(columns='Unnamed: 0')
data_set=data_set[['ds','n_datos']]
data_set.columns=['ds','y']

data_set['day_of_week']=data_set['ds'].dt.weekday
data_set['outliers'] = (data_set['y'] <= 19.28) | (data_set['y'] >= 99.94)

###############################################################################
############################### tranform data ################################
###############################################################################

def imputacion_ant_post(indices, df, column_ind_name):
    for i in indices:
        if i < 7:
            x2 = df[df[column_ind_name] == i+7]['y'].values[0]
            df.loc[df[column_ind_name] == i, 'y'] = x2
        elif i > (df.shape[0]-7):
            x1 = df[df[column_ind_name] == i-7]['y'].values[0]
            df.loc[df[column_ind_name] == i, 'y'] = x1
        else:
            x2 = df[df[column_ind_name] == i+7]['y'].values[0]
            x1 = df[df[column_ind_name] == i-7]['y'].values[0]
            df.loc[df[column_ind_name] == i, 'y'] = np.mean([x1,x2])
            
    return df

# creamos columna indices 
    

data_set['ind'] = range(0, data_set.shape[0])
indices = data_set[data_set.outliers == True].ind.values
data_set2 = imputacion_ant_post(indices=indices, df=data_set, column_ind_name='ind')

plt.hist(data_set2['y'])
#data_set['n_datos'], lam = boxcox(data_set['n_datos'])

data_set2.describe()
plt.hist(data_set2['y'])
data_set2 = data_set2.drop(columns='day_of_week')
data_set2 = data_set2.drop(columns= 'outliers')
data_set2 = data_set2.drop(columns= 'ind')
pyplot.plot(data_set2)
pyplot.show()
print(data_set.isnull().sum())

###############################################################################
############################## modelo 1  ######################################
###############################################################################
model= Prophet(daily_seasonality=True, 
               weekly_seasonality=True)
model.fit(data_set2)
future = model.make_future_dataframe(freq='d', periods= 7) #includes by default dates from the history
forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower', 'yhat_upper']].tail() #lower and upper are uncertainty intervals

fig4=model.plot_components(forecast)
fig3= model.plot(forecast)


###############################################################################
############################## modelo 2  ######################################
###############################################################################
model= Prophet(seasonality_mode='multiplicative', 
               daily_seasonality=7, 
               yearly_seasonality=False, 
               weekly_seasonality=True, 
               changepoint_prior_scale=0.05, 
               seasonality_prior_scale=0.05)
model.add_country_holidays(country_name='ES')
model.fit(data_set2)

future = model.make_future_dataframe(freq='d', periods=7) #includes by default dates from the history
future.tail()

forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower', 'yhat_upper']].tail()

fig3= model.plot(forecast)
fig4=model.plot_components(forecast)

###############################################################################
############################## modelo 3  ######################################
###############################################################################
model=  Prophet(
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode='multiplicative', 
        changepoint_prior_scale=0.025, 
        changepoint_range=0.9,
        seasonality_prior_scale=0.05
        ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=12
        ).add_seasonality(
                name='daily',
                period=1,
                fourier_order=15,
        ).add_seasonality(
                name='weekly',
                period=7,
                fourier_order=20)
model.add_country_holidays(country_name='ES')
model.fit(data_set2)
future = model.make_future_dataframe(freq='d', periods=7) #includes by default dates from the history
future.tail()
forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower', 'yhat_upper']].tail()
fig3= model.plot(forecast)
fig4=model.plot_components(forecast)

###############################################################################
############################## modelo 4  ######################################
###############################################################################
data_set_copy1 = data_set2.copy()
def is_weekend(data_set2):
    date = pd.to_datetime(data_set2)
    if date.dayofweek >= 5:
        return 1
    else:
        return 0



data_set_copy1['is_weekend'] = data_set_copy1['ds'].apply(is_weekend)


model= Prophet(seasonality_mode='multiplicative', weekly_seasonality= False, daily_seasonality=7, yearly_seasonality=True, changepoint_prior_scale=5)
model.add_seasonality(name='weekly_is_weekend', period=7, fourier_order=10, condition_name='is_weekend')
model.add_country_holidays(country_name='ES')

#!!!!!!!!!!!!!!!!Aquí hay un fallo, no se encuentra en lo que sigue 
#is_weekend ¿Cómo se arreglaría? 
#Nota: Resolver si hay tiempo como ejercicio

forecast = model.fit(data_set_copy1).predict(future)
forecast[['ds','yhat','yhat_lower', 'yhat_upper']].tail()
fig = model.plot_components(forecast)
fig= model.plot(forecast)



###############################################################################
############################## modelo 5  ######################################
###############################################################################

model=  Prophet(
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode='multiplicative', 
        changepoint_prior_scale=5, 
        changepoint_range=0.9,
        ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=12
        ).add_seasonality(
                name='daily',
                period=1,
                fourier_order=15
        ).add_seasonality(
                name='weekly',
                period=7,
                fourier_order=20
        ).add_seasonality(
                name='quarterly',
                period=365.25/4,
                fourier_order=5,
                prior_scale=15)


###############################################################################
############################## modelo 6  ######################################
###############################################################################

data_set3 = data_set2.copy()
def is_weekend_(data_set3):
    date = pd.to_datetime(data_set3)
    if date.dayofweek >= 5:
        return 1
    else:
        return 0
data_set3['is_weekend_'] = data_set3['ds'].apply(is_weekend_)

model = Prophet(seasonality_mode='multiplicative', daily_seasonality=7, changepoint_prior_scale=3, yearly_seasonality=True, weekly_seasonality=True)
model.add_regressor('is_weekend_', mode='multiplicative')
model.add_country_holidays(country_name='ES')
model.fit(data_set3)

future['is_weekend_'] = future['ds'].apply(is_weekend_)

forecast[['ds','yhat','yhat_lower', 'yhat_upper']].tail()
forecast = model.predict(future)
fig = model.plot_components(forecast)
fig= model.plot(forecast)

