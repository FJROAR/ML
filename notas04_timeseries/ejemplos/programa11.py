#https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

#Se importan los datos
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
print(df.shape)  # (123, 8)
df.tail()

#Visualización de las series temporales a analizar, casi todas presentan el
#mismo patrón salvo gdfce y gdfim

fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();

#Se aplican los Test de Causalidad de Granger. Su H0 es que los coeficientes de
#vaolores pasados en la ecuación de regresión es 0

#Las filas son la respuesta Y y las columnas son las variables explicativas
#X, se observan los p-valores de las series en la matriz que se genera

#Si el p-valor < 0.05, entonces la serie X causa a la serie Y, en este caso
#"todas las series se causan"
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(df, variables = df.columns)   

#Análisis de co-integración: Permite establecer una conexión estadísticamente 
#significativa entre 2 o más series temporales
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)


#Sub-división del dataset en training-test
nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]

# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)  # (4, 8)


#Se trata de analizar y convertir las series en estacionarias

#Análisis de la estacionariedad
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Realiza la prueba ADF para verificar la estacionariedad de una serie y muestra un informe."""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']
    
    def adjust(val, length=6): 
        return str(val).ljust(length)

    # Imprimir resumen
    print(f'    Prueba ADF para la columna "{name}"', "\n   ", '-'*47)
    print(f' Hipótesis nula: Los datos tienen raíz unitaria. No estacionaria.')
    print(f' Nivel de significancia = {signif}')
    print(f' Estadístico de prueba  = {output["test_statistic"]}')
    print(f' Número de lags elegidos = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Valor crítico {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => Valor p = {p_value}. Rechazamos la hipótesis nula.")
        print(" => La serie es estacionaria.")
    else:
        print(f" => Valor p = {p_value}. Evidencia débil para rechazar la hipótesis nula.")
        print(" => La serie no es estacionaria.")

# Prueba ADF en cada columna de df_train
for name, column in df_train.items():
    adfuller_test(column, name=name)
    print('\n')
    
#Conversión por primeras diferencia en estacionarias y vuelta a aplicar los 
#tests ADF
df_differenced = df_train.diff().dropna()

for name, column in df_differenced.items():
    adfuller_test(column, name=column.name)
    print('\n')
    

#Si se vuelve a aplicar una nueva diferenciación, entonces ya sí se consigue
#que todas las series sean estacionarias
df_differenced = df_differenced.diff().dropna()

#Selección (por criterios de complejidad) del orden correcto del VAR
model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

#Bajo el AIC, el valor más bajo se alcanza en p=4
#Otro modo sería con:
    #x = model.select_order(maxlags=12)
    #x.summary()

#Se entrena con el modelo escogido VAR(4)
model_fitted = model.fit(4)
model_fitted.summary()


#Análisis de la correlación residual
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(col, ':', round(val, 2))

#Al ofrecer valores en torno a 2, parece que las autocorrelaciones residuales
#están adecuadamente controladas

#Predicción a futuro
lag_order = model_fitted.k_ar
print(lag_order)

forecast_input = df_differenced.values[-lag_order:]
forecast_input

fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast

#Se eliminan las transformaciones para obtener los valores de predicción adecuados
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
df_results.loc[:, ['rgnp_forecast', 'pgnp_forecast', 'ulc_forecast', 'gdfco_forecast',
                   'gdf_forecast', 'gdfim_forecast', 'gdfcf_forecast', 'gdfce_forecast']]

#Testing: Forecast vs Real
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();

#Evaluación analítica de predicciones
from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})



# Convertir las columnas en arrays unidimensionales de NumPy
rgnp_forecast = np.array(df_results['rgnp_forecast']).flatten()
rgnp_test = np.array(df_test['rgnp']).flatten()

# Lista de nombres de las columnas para calcular la precisión del pronóstico
variables = ['rgnp', 'pgnp', 'ulc', 'gdfco', 'gdf', 'gdfim', 'gdfcf', 'gdfce']

# Iterar sobre cada variable, calcular precisión y mostrar los resultados
for var in variables:
    forecast_column = f'{var}_forecast'
    
    # Convertir las columnas en arrays unidimensionales de NumPy
    forecast_values = np.array(df_results[forecast_column]).flatten()
    actual_values = np.array(df_test[var]).flatten()
    
    # Calcular la precisión del pronóstico
    print(f'\nForecast Accuracy of: {var}')
    accuracy_prod = forecast_accuracy(forecast_values, actual_values)
    
    # Imprimir cada métrica de precisión
    for k, v in accuracy_prod.items():
        print(f'{k}: {round(v, 4)}')