#https://www.ine.es/jaxiT3/Datos.htm?t=13896
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go

df_hip = pd.read_csv('data/hipotec.csv', sep = ';')
df_hip = df_hip.sort_values(by = 'fch', ascending = True)

x = df_hip['fch']
y = df_hip['valor']

fig = go.Figure()
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor'], 
                         mode = 'lines'))
              
plot(fig)


#Cálculo de una media móvil no centrada de orden 5

window=5
valor_ajust = df_hip['valor'].rolling(window=window, 
                                      center = False).mean()
df_hip['valor_m5_0c'] = valor_ajust


#Efecto alisado
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor'], 
                         mode = 'lines',
                         name = 'Serie real'))
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor_m5_0c'], 
                         mode = 'lines',
                         name = 'Media móvil 5 non-center'))

plot(fig)


#Cálculo de una media móvil centrada de orden 5

window=7
valor_ajust = df_hip['valor'].rolling(window=window, center = True).mean()
df_hip['valor_m5_1c'] = valor_ajust


#Efecto alisado
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor'], 
                         mode = 'lines',
                         name = 'Serie real'))
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor_m5_1c'], 
                         mode = 'lines',
                         name = 'Media móvil 5 center'))

plot(fig)
