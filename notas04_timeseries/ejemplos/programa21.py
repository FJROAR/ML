#https://pycaret.gitbook.io/docs/learn-pycaret/official-blog/time-series-anomaly-detection-with-pycaret

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Cargar el conjunto de datos de taxis de Nueva York (usa un archivo CSV descargado)
# Asegúrate de cambiar la ruta del archivo según corresponda
url = 'data/nyc_taxi.csv'
data = pd.read_csv(url)


# selecciona los viajes
data_hourly = data['value']

# Visualizar los datos de series temporales
plt.figure(figsize=(14, 7))
data_hourly.plot(title='Número de viajes de taxi por hora', xlabel='Fecha', ylabel='Número de viajes', color='blue')
plt.axhline(data_hourly.mean(), color='red', linestyle='--', label='Media')
plt.legend()
plt.show()

# Aplicar Isolation Forest para detectar anomalías
model = IsolationForest(contamination=0.05)  # Ajustar según sea necesario
data_hourly = data_hourly.values.reshape(-1, 1)  # Reshape para el modelo
model.fit(data_hourly)

# Predecir anomalías
predictions = model.predict(data_hourly)
anomalies = np.where(predictions == -1)[0]  # Índices de las anomalías

# Visualizar las anomalías en la serie temporal
plt.figure(figsize=(14, 7))
plt.plot(data_hourly, label='Número de viajes', color='blue')
plt.scatter(anomalies, data_hourly[anomalies], color='red', label='Anomalías', s=100)
plt.title('Detección de Anomalías en el Número de Viajes de Taxi')
plt.xlabel('Hora')
plt.ylabel('Número de viajes')
plt.legend()
plt.show()
