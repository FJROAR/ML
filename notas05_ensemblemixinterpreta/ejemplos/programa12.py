# Importar las librerías necesarias
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar el dataset de Titanic
titanic = fetch_openml('titanic', version=1)

# Convertir a DataFrame
X = pd.DataFrame(titanic.data, columns=titanic.feature_names)
y = titanic.target

# Seleccionar solo 5 variables relevantes
X = X[['pclass', 'sex', 'age', 'sibsp', 'fare']]
X['sex'] = X['sex'].map({'male': 0, 'female': 1})  # Convertir variable categórica

# Rellenar valores nulos en 'age' con la media y en 'fare' con la media
X['age'].fillna(X['age'].mean(), inplace=True)
X['fare'].fillna(X['fare'].mean(), inplace=True)  # Asegurarse de que no haya valores nulos en 'fare'

# Eliminar filas con valores nulos en 'pclass' y 'sibsp' (si existen)
X.dropna(subset=['pclass', 'sibsp'], inplace=True)

# Asegurarse de que la variable objetivo y las características tengan el mismo número de muestras
y = y.loc[X.index]  # Alinear la variable objetivo con las características

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y.astype(int), 
                                                    test_size=0.2, 
                                                    random_state=155)

# Entrenar un modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=155)
model.fit(X_train, y_train)

# Crear un explainer de SHAP
explainer = shap.TreeExplainer(model)

# Generar los valores SHAP
shap_values = explainer.shap_values(X_test)


shap_df = pd.DataFrame(shap_values[:, :, 1], columns=X.columns)


# Calcular la importancia promedio de las características
importance = shap_df.abs().mean().sort_values(ascending=False)

# Crear un gráfico de barras usando Plotly
fig = px.bar(importance, x=importance.index, y=importance.values,
             labels={'x': 'Características', 'y': 'Importancia promedio'},
             title='Importancia de las Características según SHAP',
             color=importance.values)

# Guardar el gráfico en un archivo HTML
fig.write_html('shap_plotly_summary.html')

print("El gráfico de SHAP se ha guardado como 'shap_plotly_summary.html'")


#Visualización total


def plot_shap_summary_seaborn(shap_values, feature_names, title="SHAP Summary Plot"):
    """
    Crea un gráfico de resumen de SHAP usando Seaborn.

    Parameters:
    - shap_values: array-like, valores SHAP.
    - feature_names: lista de nombres de las características.
    - title: Título del gráfico.
    """
    # Crear un DataFrame con los valores SHAP y las características
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Derretir el DataFrame para crear un formato largo
    shap_long = shap_df.melt(var_name='Feature', value_name='SHAP Value')

    # Calcular el valor absoluto de SHAP para ordenarlos
    shap_long['Abs SHAP Value'] = shap_long['SHAP Value'].abs()

    # Ordenar por valor absoluto de SHAP
    shap_long = shap_long.sort_values(by='Abs SHAP Value', ascending=False)

    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, len(feature_names) * 0.5))
    sns.stripplot(data=shap_long, x='SHAP Value', y='Feature', hue='Abs SHAP Value', 
                   palette='viridis', size=5, jitter=True, alpha=0.7)

    plt.title(title)
    plt.xlabel('Valor SHAP')
    plt.ylabel('Características')
    plt.legend(title='Valor Absoluto de SHAP', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
    
    
plot_shap_summary_seaborn(shap_values[:, :, 1], X.columns)


shap.plots.violin(shap_values[:, :, 1],
                  feature_names = X.columns)





