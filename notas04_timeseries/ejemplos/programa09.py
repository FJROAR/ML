#https://setscholars.net/a-step-by-step-modelling-approach-to-the-forecasting-of-bangladesh-population-using-box-jenkins-method/

from pandas import read_csv
#from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

df = read_csv('data/BD_pop_1960_2019.csv', sep = ';')
df = df.set_index('Year')


#Paso un: Visualización de la Serie Temporal


df.plot(figsize = (8,6))

fig = pyplot.figure(figsize = (8,6))
autocorrelation_plot(df)
pyplot.show()


#En este caso no se observa tendencia en varianza, a lo sumo
#en media y se analizará por diferenciación


from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(8,6), 'figure.dpi':120})

plot_acf(df)
pyplot.show()

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(df)
pyplot.show()

#Test de estacionariedad

# =============================================================
# Prepared by: SETScholars (https://setscholars.net)
# =============================================================

from statsmodels.tsa.stattools import adfuller

# ADF Test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(); print(f'ADF Statistic: {result[0]}')
    print();  print(f'n_lags: {result[1]}')
    print();  print(f'p-value: {result[1]}')

    print(); print('Critial Values:')
    for key, value in result[4].items():
        print(f'   {key}, {value}')   

adf_test(df["Population"])


from statsmodels.tsa.stattools import kpss

def kpss_test(series, **kw):    
    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    
    # Format Output
    print(); print(f'KPSS Statistic: {statistic}')
    print(); print(f'p-value: {p_value}')
    print(); print(f'num lags: {n_lags}')
    print(); print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    
kpss_test(df["Population"])


#Buscando el orden d

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(16,12), 'figure.dpi':120})

df.reset_index(drop=True, inplace=True)

# Original Series
fig, axes = plt.subplots(4, 2, sharex=True)
axes[0, 0].plot(df.values); axes[0, 0].set_title('Original Series')
plot_acf(df.values, ax=axes[0, 1], lags = len(df)-1)

# 1st Differencing
df1 = df["Population"].diff()
axes[1, 0].plot(df1); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df1.dropna(), ax=axes[1, 1], lags = len(df1.dropna())-1)

# 2nd Differencing
df2 = df["Population"].diff().diff()
axes[2, 0].plot(df2); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df2.dropna(), ax=axes[2, 1], lags = len(df2.dropna())-1)

# 3rd Differencing
df3 = df["Population"].diff().diff().diff()
axes[3, 0].plot(df3); axes[3, 0].set_title('3rd Order Differencing')
plot_acf(df3.dropna(), ax=axes[3, 1], lags = len(df3.dropna())-1)

plt.show()

#Buscando el orden AR
from statsmodels.graphics.tsaplots import plot_pacf
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(14,10), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)

df2 = df["Population"].diff().diff()

axes[0].plot(df2); axes[0].set_title('2nd Differencing')
axes[1].set(ylim=(-5,5))
plot_pacf(df2.dropna(), ax=axes[1]) #PACF

plt.show()


#Buscando el orden MA
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize':(14,10), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)

df2 = df["Population"].diff().diff()
axes[0].plot(df2); axes[0].set_title('2nd Differencing')
#axes[1].set(ylim=(0,1.2))
plot_acf(df2.dropna(), ax=axes[1], lags = len(df2.dropna())-1) # ACF

plt.show()

## ADF test
adf_test(df2.dropna())

#Se construye el modelo
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams.update({'figure.figsize':(12,6), 'figure.dpi':220})

df = read_csv('data/BD_pop_1960_2019.csv', sep = ";")
df = df.set_index('Year')

# 2,2,2 ARIMA Model
model = ARIMA(df["Population"], order=(2,2,2))
model_fit = model.fit()
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
predictions = model_fit.get_prediction(dynamic=False)

# Obtén los intervalos de confianza
pred_ci = predictions.conf_int()

# Grafica los datos originales y las predicciones
plt.figure(figsize=(10, 6))
plt.plot(df["Population"], label="Datos reales")
plt.plot(predictions.predicted_mean, label="Predicción", color="orange")

# Grafica los intervalos de confianza
plt.fill_between(pred_ci.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color="lightgrey", alpha=0.5)

plt.legend()
plt.show()

