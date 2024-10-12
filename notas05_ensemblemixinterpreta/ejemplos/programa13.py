import pandas as pd
import numpy as np
import dalex as dx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  # Importar matplotlib
%matplotlib qt


# Cargar el conjunto de datos California Housing
california_data = fetch_california_housing()
X = pd.DataFrame(california_data.data, columns=california_data.feature_names)
y = california_data.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Crear un objeto de modelo de Dalex
explainer = dx.Explainer(model, X_train, y_train, label='Random Forest')

# Explicaciones globales
# Obtener la importancia de las características
global_explanation = explainer.model_parts()

importance_df = global_explanation.result
print("Importancia de las variables:")
print(importance_df)

df_explanation0 = explainer.predict_parts(X_test.iloc[0]).result

explainer.predict_parts(X_test.iloc[0]).plot()
plt.show()

