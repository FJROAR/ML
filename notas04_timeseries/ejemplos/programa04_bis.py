#https://www.ine.es/jaxiT3/Datos.htm?t=13896
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go

#Para el alisado exponencial simple
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#from   statsmodels.tsa.holtwinters import  Holt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

df_hip = pd.read_csv('data/pernoctaciones.csv', sep = ';')
df_hip = df_hip.sort_values(by = 'fch', ascending = True).reset_index(drop = True)

x = df_hip['fch']
y = df_hip['pernoctaciones']

model = ExponentialSmoothing(y, 
                             trend = 'add',
                             damped = False,
                             seasonal = 'add',
                             seasonal_periods = 12).fit()

y_pred = y.append(model.forecast(12))
x_pred = pd.Series(x).append(pd.Series(['2020M04','2020M05','2020M06','2020M07','2020M08','2020M09',
                '2020M10','2020M11','2020M12','2021M01','2021M02','2021M03']))

#Efecto alisado
fig = go.Figure()
fig.add_trace(go.Scatter(x = x_pred,
                         y = y_pred, 
                         mode = 'lines',
                         name = 'Prediccion con Holt-Winter Estacional'))
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['pernoctaciones'], 
                         mode = 'lines',
                         name = 'Serie real'))

plot(fig)

