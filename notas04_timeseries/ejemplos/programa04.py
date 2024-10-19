#https://www.ine.es/jaxiT3/Datos.htm?t=13896
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go

#Para el alisado exponencial simple
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing
#from   statsmodels.tsa.holtwinters import  Holt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

df_hip = pd.read_csv('data/hipotec.csv', sep = ';')
df_hip = df_hip.sort_values(by = 'fch', ascending = True).reset_index(drop = True)

x = df_hip['fch']
y = df_hip['valor']

model = ExponentialSmoothing(y, 
                             trend = 'add',
                             damped = False,
                             seasonal = 'add',
                             seasonal_periods = 12).fit()

y_pred = y.append(model.forecast(12))
x_pred = pd.Series(x).append(pd.Series(['2021M07','2021M08','2021M09','2021M10','2021M11','2021M12',
                '2022M01','2022M02','2022M03','2022M04','2022M05','2022M06']))

#Efecto alisado
fig = go.Figure()
fig.add_trace(go.Scatter(x = x_pred,
                         y = y_pred, 
                         mode = 'lines',
                         name = 'Prediccion con Holt-Winter Estacional'))
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor'], 
                         mode = 'lines',
                         name = 'Serie real'))

plot(fig)

