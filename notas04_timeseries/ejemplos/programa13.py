import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/pernoctaciones.csv',
                 sep = ';')
df['fch'] = 100*df['fch'].str.slice(0, 4).astype(int) + df['fch'].str.slice(6, 7).astype(int)


#Se crean regresores para el modelo

df['cte'] = 1


df['trend'] = np.arange(0, len(df['fch']))
df['cos1'] = np.cos((np.pi * 1 / 12)*df['trend'])
df['cos2'] = np.cos((np.pi * 2 / 12)*df['trend'])
df['cos3'] = np.cos((np.pi * 3 / 12)*df['trend'])
df['cos4'] = np.cos((np.pi * 4 / 12)*df['trend'])
df['cos5'] = np.cos((np.pi * 5 / 12)*df['trend'])
df['cos6'] = np.cos((np.pi * 6 / 12)*df['trend'])



df['sin1'] = np.sin((np.pi * 1 / 12)*df['trend'])
df['sin2'] = np.sin((np.pi * 2 / 12)*df['trend'])
df['sin3'] = np.sin((np.pi * 3 / 12)*df['trend'])
df['sin4'] = np.sin((np.pi * 4 / 12)*df['trend'])
df['sin5'] = np.sin((np.pi * 5 / 12)*df['trend'])
df['sin6'] = np.sin((np.pi * 6 / 12)*df['trend'])



df['agosto'] = np.where(df['fch'] % 100 == 8, 1, 0)



df_test = df[df['fch'] >= 201900].reset_index(drop = True)
df_train = df[df['fch'] < 201900].reset_index(drop = True)

#Se crea un modelo de regresión lineal en el training a ver qué variables
#resultan significativas

y = df_train['pernoctaciones']
X = df_train[['cte', 'trend', 'cos1', 'cos2', 'cos3', 'cos4', 'cos5', 'cos6',
        'sin1', 'sin2', 'sin3',  'sin4', 'sin5', 'sin6','agosto']]



import statsmodels.api as sm
mod1 = sm.OLS(y, X).fit()  

mod1.summary()

#Modelo 2 Se eliminan las no significativas
y2 = df_train['pernoctaciones']
X2 = df_train[['cte', 'trend', 'cos2', 'cos4', 'cos6',
        'sin2',  'sin4', 'sin6','agosto']]



mod2 = sm.OLS(y2, X2).fit()  
mod2.summary()


#Los datos tienen autocorrelación residual, cabe proporner alguna transformación
#a ver si se corrige

y3 = (y2 - y2.shift(-1))
#Se imputa el primer valor al haber muchos en la serie, el impacto
#es mínimo
y3 = y3.fillna(0)

mod31 = sm.OLS(y3, X2).fit()  
mod31.summary()


mod32 = sm.OLS(y3, df_train[['cte', 'cos2', 'cos4', 'cos6',
        'sin2',  'sin4', 'sin6','agosto']]).fit()  
mod32.summary()


#Predicción
incr_pred = mod32.predict(df_test[['cte', 'cos2', 'cos4', 'cos6',
        'sin2',  'sin4', 'sin6','agosto']])

df_test['pred'] = incr_pred

pred0 = df_train['pernoctaciones'][0] + incr_pred[14]
pred0

pred1 = pred0 + incr_pred[13]
pred1

pred2 = pred1 + incr_pred[12]
pred2

pred3 = pred2 + incr_pred[11]
pred3

pred4 = pred3 + incr_pred[10]
pred4

pred5 = pred4 + incr_pred[9]
pred5

pred6 = pred5 + incr_pred[8]
pred6