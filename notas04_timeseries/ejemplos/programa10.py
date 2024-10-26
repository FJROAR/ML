import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

df_parinf = pd.read_csv('data/paroinfla.csv', sep = ';', decimal = ',')

#Dado que se sabía que existía autocorrelación, se va a tratar de asociar
#un modelo SARIMA al tener período trimestral = 4
# (1, 1, 0) x (1, 0, 0) con la variable explicativa PARO

df_exog = df_parinf['PARO'][0: (len(df_parinf) - 4)]
df_endog = df_parinf['INFLACION'][0: (len(df_parinf) - 4)]

mod = ARIMA(df_endog, 
            df_exog,
            order = (1, 1, 0),
            seasonal_order = (1, 0, 0, 4))

res = mod.fit()
print(res.summary())

plot_acf(res.resid)
plot_pacf(res.resid)

for_exog = df_parinf['PARO'][(len(df_parinf) - 4) :(len(df_parinf) - 0)]
forecasting = res.forecast(4, exog = for_exog)



from plotly.offline import plot
import plotly.graph_objects as go

y_pred = y_pred = pd.concat([df_endog, pd.Series(forecasting)])

x_pred = pd.concat([df_parinf['obs'], 
                    pd.Series(['1990Q1', '1990Q2', '1990Q3', '1990Q4'])], 
                   ignore_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x = x_pred,
                         y = y_pred, 
                         mode = 'lines',
                         name = 'Prediccion con ARIMA'))
fig.add_trace(go.Scatter(x = df_parinf['obs'],
                         y = df_endog, 
                         mode = 'lines',
                         name = 'Serie real'))

plot(fig)


real = df_endog = df_parinf['INFLACION'][(len(df_parinf) - 4): (len(df_parinf))]
pred = forecasting

df_acc = pd.DataFrame({'real' : real,
                       'pred' : pred})
