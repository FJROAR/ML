#https://www.ine.es/jaxiT3/Datos.htm?t=13896
import pandas as pd
from plotly.offline import plot
import plotly.graph_objects as go

df_hip = pd.read_csv('data/hipotec.csv', sep = ';')
df_hip = df_hip.sort_values(by = 'fch', ascending = True).reset_index(drop = True)

x = df_hip['fch']
y = df_hip['valor']


def SuavizacionExponencialSimple(series, alpha):
    #series = y
    #alpha = 0.3
    result = [series[0]]
    for n in range(1, len(series)) :
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

df_hip['Alisada_simple'] = SuavizacionExponencialSimple(y, 0.3)


#Efecto alisado
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor'], 
                         mode = 'lines',
                         name = 'Serie real'))
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['Alisada_simple'], 
                         mode = 'lines',
                         name = 'Alisada simple alfa = 0.3'))

plot(fig)


def SuavizacionHolt(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


df_hip['Alisada_doble'] = SuavizacionHolt(y, 0.3, 0.1)[0:(len(y))]


#Efecto alisado
fig = go.Figure()
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['valor'], 
                         mode = 'lines',
                         name = 'Serie real'))
fig.add_trace(go.Scatter(x = df_hip['fch'],
                         y = df_hip['Alisada_doble'], 
                         mode = 'lines',
                         name = 'Alisada doble alfa = 0.3, beta = 0.1'))

plot(fig)