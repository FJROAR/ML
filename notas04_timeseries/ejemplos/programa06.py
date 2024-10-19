#https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

import numpy as np
import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm


series = pd.read_csv('data/daily-min-temperatures.csv', 
                     header=0, 
                     index_col=0,
                     sep = ';')
series.plot()
pyplot.show()


#Estructura de autocorrelaciones
plot_acf(series, lags = 20)
pyplot.show()

plot_pacf(series, lags = 20)
pyplot.show()


#Cálculo de las 2 primeras FAT (excluyendo la 0 que es lógicamente la varianza)
sum1 = series - np.mean(series)
sum2 = series.shift(1) - np.mean(series)
autocov1 = np.sum(sum1 * sum2)/len(series)
rho1t = autocov1 / np.var(series)

rho1t

sum1 = series - np.mean(series)
sum2 = series.shift(2) - np.mean(series)
autocov2 = np.sum(sum1 * sum2)/len(series)
rho2t = autocov2 / np.var(series)

rho2t


#Cálculo bajo statmodels
acf(series)[1]
acf(series)[2]

#Estructura de autocorrelaciones parciales
plot_pacf(series, lags=20)
pyplot.show()



#Cálculo de las 2 primeras FAP (excluyendo la 0 que es lógicamente la varianza)

#Modelo lineal 1 (bajo steatsmodels)
n = len(series)
y = series[1:(n+1)]
x1 = series.shift(1)[1:(n+1)]
x1 = sm.add_constant(x1)

mod1 = sm.OLS(y, x1).fit()
mod1.params[1]


#Modelo lineal 2 (bajo steatsmodels)
n = len(series)
y = series[2:(n+1)]
x1 = series.shift(1)[2:(n+1)]
x2 = series.shift(2)[2:(n+1)]
x = pd.DataFrame({'x1': x1['Temp'], 'x2': x2['Temp']})
x = sm.add_constant(x)
mod2 = sm.OLS(y, x).fit()
mod2.params[2]


#Cálculo bajo statmodels
pacf(series)[1]
pacf(series)[2]

plot(pacf(series))





