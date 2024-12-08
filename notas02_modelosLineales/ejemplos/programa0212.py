#https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
#conda install -c sebp scikit-survival

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxnetSurvivalAnalysis


data = pd.read_csv('data/lung.csv')
data.head()

#Se eliminan filas con datos nulos
data = data.dropna(subset=['time', 'status', 'age', 'sex', 'ph.ecog',
                           'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss'])
data.shape

y = data[['status', 'time']]

y['status'] = np.where(y['status'] == 1, True, False)


y = y.to_records(index = False)

X = data[['age', 'sex', 
          'ph.ecog', 'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss']]


#Se llama al método con regularización Lasso
coxReg = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
model = coxReg.fit(X, y)

# Obtener los coeficientes del modelo
coefficients = model.coef_

# Obtener la última columna de los coeficientes
final_coefficients = coefficients[:, -1]

# Crear el DataFrame con los coeficientes de las variables
coef_df = pd.DataFrame({'Coeficientes': final_coefficients}, index=X.columns)

print(coef_df)

# Convertir los coeficientes en un DataFrame para crear una tabla
coef_df = pd.DataFrame(coefficients.T, 
                       columns=['Coeficientes'], 
                       index=[f"Variable_{i}" for i in range(X.shape[1])])


def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, name + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


coefficients_elastic_net = pd.DataFrame(
    coxReg.coef_,
    index=X.columns,
    columns=np.round(coxReg.alphas_, 5)
)

plot_coefficients(coefficients_elastic_net, n_highlight=5)
plt.show()